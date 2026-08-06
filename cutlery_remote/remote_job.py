from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any


class RemoteExecutionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RemoteExecutionJobError(RuntimeError):
    """Raised when a remote execution job receives an invalid lifecycle transition."""


class RemoteExecutionCallbackError(RuntimeError):
    """Raised after every callback in one cleanup phase has had a chance to run."""

    def __init__(self, phase: str, errors: list[BaseException]):
        self.phase = phase
        self.errors = tuple(errors)
        super().__init__(f"Remote execution {phase} callbacks failed: {len(errors)} error(s).")


RemoteExecutionCallback = Callable[["RemoteExecutionJob"], Awaitable[None] | None]


class RemoteExecutionJob:
    """Own one local prompt's remote execution lifecycle.

    Callbacks receive this job, so adapters can inspect the local and remote
    prompt ids without coupling this primitive to a queue, HTTP client, or
    process implementation. A cancellation always attempts every abort,
    terminate, peer-interrupt, and cleanup callback in that order.
    """

    def __init__(self, local_prompt_id: str, remote_prompt_ids: tuple[str, ...] = ()):
        normalized_local_id = _required_id(local_prompt_id, "local_prompt_id")
        self.local_prompt_id = normalized_local_id
        self.cancellation_event = asyncio.Event()
        self._remote_prompt_ids = {_required_id(prompt_id, "remote_prompt_id") for prompt_id in remote_prompt_ids}
        self._state = RemoteExecutionState.PENDING
        self._result: Any = None
        self._error: BaseException | None = None
        self._callbacks: dict[str, list[RemoteExecutionCallback]] = {
            "abort": [],
            "terminate": [],
            "peer_interrupt": [],
            "cleanup": [],
        }
        self._state_guard = threading.RLock()
        self._cleanup_lock = asyncio.Lock()
        self._cleaned = False
        self._cleanup_error: RemoteExecutionCallbackError | None = None

    @property
    def state(self) -> RemoteExecutionState:
        with self._state_guard:
            return self._state

    def check_cancelled(self) -> None:
        if self.cancellation_event.is_set():
            raise asyncio.CancelledError(f"Remote execution job {self.local_prompt_id!r} was cancelled.")

    @property
    def result(self) -> Any:
        with self._state_guard:
            return self._result

    @property
    def error(self) -> BaseException | None:
        with self._state_guard:
            return self._error

    @property
    def remote_prompt_ids(self) -> frozenset[str]:
        with self._state_guard:
            return frozenset(self._remote_prompt_ids)

    @property
    def cleaned(self) -> bool:
        with self._state_guard:
            return self._cleaned

    def add_remote_prompt_id(self, remote_prompt_id: str) -> None:
        with self._state_guard:
            if self._cleaned:
                raise RemoteExecutionJobError("Cannot add a remote prompt id after cleanup.")
            self._remote_prompt_ids.add(_required_id(remote_prompt_id, "remote_prompt_id"))

    def start(self) -> None:
        with self._state_guard:
            if self._state is not RemoteExecutionState.PENDING:
                raise RemoteExecutionJobError(f"Cannot start a job in {self._state.value} state.")
            self._state = RemoteExecutionState.RUNNING

    def register_abort(self, callback: RemoteExecutionCallback) -> None:
        self._register_callback("abort", callback)

    def register_terminate(self, callback: RemoteExecutionCallback) -> None:
        self._register_callback("terminate", callback)

    def register_peer_interrupt(self, callback: RemoteExecutionCallback) -> None:
        self._register_callback("peer_interrupt", callback)

    def register_cleanup(self, callback: RemoteExecutionCallback) -> None:
        self._register_callback("cleanup", callback)

    async def succeed(self, result: Any) -> None:
        with self._state_guard:
            self._require_active("succeed")
            self._result = result
            self._state = RemoteExecutionState.SUCCEEDED
        await self.cleanup()

    async def fail(self, error: BaseException) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("error must be an exception instance.")
        with self._state_guard:
            self._require_active("fail")
            self._error = error
            self._state = RemoteExecutionState.FAILED
        await self.cleanup()

    async def cancel(self) -> bool:
        with self._state_guard:
            if self._state in _TERMINAL_STATES:
                return False
            self._state = RemoteExecutionState.CANCELLED
            self.cancellation_event.set()

        errors: list[BaseException] = []
        for phase in ("abort", "terminate", "peer_interrupt"):
            errors.extend(await self._run_callbacks(phase))
        try:
            await self.cleanup()
        except RemoteExecutionCallbackError as exc:
            errors.extend(exc.errors)
        if errors:
            raise RemoteExecutionCallbackError("cancel", errors)
        return True

    async def cleanup(self) -> bool:
        """Run cleanup callbacks at most once, including under concurrent callers."""

        async with self._cleanup_lock:
            with self._state_guard:
                if self._cleaned:
                    if self._cleanup_error is not None:
                        raise self._cleanup_error
                    return False
                self._cleaned = True
            errors = await self._run_callbacks("cleanup")
            if errors:
                cleanup_error = RemoteExecutionCallbackError("cleanup", errors)
                with self._state_guard:
                    self._cleanup_error = cleanup_error
                raise cleanup_error
            return True

    def _register_callback(self, phase: str, callback: RemoteExecutionCallback) -> None:
        if not callable(callback):
            raise TypeError(f"{phase} callback must be callable.")
        with self._state_guard:
            if self._cleaned:
                raise RemoteExecutionJobError("Cannot register callbacks after cleanup.")
            self._callbacks[phase].append(callback)

    def _require_active(self, action: str) -> None:
        if self._state in _TERMINAL_STATES:
            raise RemoteExecutionJobError(f"Cannot {action} a job in {self._state.value} state.")

    async def _run_callbacks(self, phase: str) -> list[BaseException]:
        with self._state_guard:
            callbacks = tuple(self._callbacks[phase])
        errors: list[BaseException] = []
        for callback in callbacks:
            try:
                outcome = callback(self)
                if inspect.isawaitable(outcome):
                    await outcome
            except BaseException as exc:
                errors.append(exc)
        return errors


class RemoteExecutionJobRegistry:
    """Thread-safe lookup of live remote jobs by the owning local prompt id."""

    def __init__(self):
        self._jobs: dict[str, RemoteExecutionJob] = {}
        self._guard = threading.RLock()

    def register(self, job: RemoteExecutionJob) -> RemoteExecutionJob:
        if not isinstance(job, RemoteExecutionJob):
            raise TypeError("job must be a RemoteExecutionJob.")
        with self._guard:
            if job.local_prompt_id in self._jobs:
                raise RemoteExecutionJobError(f"A job already owns local prompt {job.local_prompt_id!r}.")
            job.register_cleanup(lambda completed_job: self.remove(completed_job.local_prompt_id))
            self._jobs[job.local_prompt_id] = job
        return job

    def create(self, local_prompt_id: str, remote_prompt_ids: tuple[str, ...] = ()) -> RemoteExecutionJob:
        return self.register(RemoteExecutionJob(local_prompt_id, remote_prompt_ids))

    def get(self, local_prompt_id: str) -> RemoteExecutionJob | None:
        with self._guard:
            return self._jobs.get(str(local_prompt_id))

    def remove(self, local_prompt_id: str) -> None:
        with self._guard:
            self._jobs.pop(str(local_prompt_id), None)

    async def cancel(self, local_prompt_id: str) -> bool:
        job = self.get(local_prompt_id)
        if job is None:
            return False
        return await job.cancel()


_TERMINAL_STATES = frozenset(
    {
        RemoteExecutionState.SUCCEEDED,
        RemoteExecutionState.FAILED,
        RemoteExecutionState.CANCELLED,
    }
)


def _required_id(value: object, name: str) -> str:
    identifier = str(value or "").strip()
    if not identifier:
        raise ValueError(f"{name} must be a non-empty string.")
    return identifier
