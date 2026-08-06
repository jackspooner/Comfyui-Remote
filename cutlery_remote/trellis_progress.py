from __future__ import annotations

import math
from pathlib import Path
import sys
import threading
from types import ModuleType
from typing import Any, Callable


_TRELLIS_NODE_DIRECTORY = Path(__file__).resolve().parents[2] / "ComfyUI-Trellis2"
_PATCH_LOCK = threading.RLock()
_PATCH_SESSIONS: dict[Path, _TrellisTqdmPatchSession] = {}
_ACTIVE_TRELLIS_PROGRESS: dict[str, TrellisTqdmProgress] = {}


def _finite_total(value: object) -> float | None:
    try:
        total = float(value)
    except (TypeError, ValueError):
        return None
    return total if math.isfinite(total) and total > 0 else None


class _TrellisTqdmProxy:
    def __init__(self, progress: "TrellisTqdmProgress", bar: Any):
        self._progress = progress
        self._bar = bar
        self._node_key = progress.start(bar)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bar, name)

    def __iter__(self):
        first_item = True
        try:
            for item in self._bar:
                if not first_item:
                    self._sync()
                first_item = False
                yield item
        finally:
            self._sync()

    def __enter__(self):
        self._bar.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return self._bar.__exit__(exc_type, exc_value, traceback)
        finally:
            self._sync()

    def update(self, value: float = 1):
        result = self._bar.update(value)
        self._sync()
        return result

    def close(self):
        try:
            return self._bar.close()
        finally:
            self._sync()

    def _sync(self) -> None:
        if self._node_key is not None:
            self._progress.update(self._bar, self._node_key)


class _NodeProgress:
    def __init__(self, progress_bar: Any):
        self.progress_bar = progress_bar
        self.active_bar: Any = None
        self.offset = 0.0


class _TrellisTqdmPatchSession:
    def __init__(self, trellis_directory: Path):
        self.trellis_directory = trellis_directory
        self.patched: list[tuple[ModuleType, Callable[..., Any], Callable[..., Any]]] = []
        self.users = 0

    def patch_loaded_modules(self) -> None:
        for module in tuple(sys.modules.values()):
            if not _is_trellis_module(module, self.trellis_directory):
                continue
            tqdm = getattr(module, "tqdm", None)
            if not callable(tqdm):
                continue

            def wrapped_tqdm(*args, _tqdm=tqdm, **kwargs):
                bar = _tqdm(*args, **kwargs)
                progress = _progress_for_executing_node()
                return _TrellisTqdmProxy(progress, bar) if progress is not None else bar

            setattr(module, "tqdm", wrapped_tqdm)
            self.patched.append((module, tqdm, wrapped_tqdm))

    def restore(self) -> None:
        for module, original, wrapped in reversed(self.patched):
            if getattr(module, "tqdm", None) is wrapped:
                setattr(module, "tqdm", original)
        self.patched.clear()


def _is_trellis_module(module: ModuleType | None, trellis_directory: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        Path(module_file).resolve().relative_to(trellis_directory)
    except (OSError, ValueError):
        return False
    return True


def _progress_for_executing_node() -> TrellisTqdmProgress | None:
    from comfy_execution.utils import get_executing_context

    executing_context = get_executing_context()
    if executing_context is None:
        return None
    with _PATCH_LOCK:
        return _ACTIVE_TRELLIS_PROGRESS.get(str(executing_context.prompt_id))


class TrellisTqdmProgress:
    """Temporarily mirror finite Trellis tqdm counters into ComfyUI node progress."""

    def __init__(self, prompt_id: str, *, trellis_directory: Path = _TRELLIS_NODE_DIRECTORY):
        self._prompt_id = prompt_id
        self._trellis_directory = trellis_directory.resolve()
        self._node_progress: dict[tuple[str, str], _NodeProgress] = {}

    def __enter__(self):
        with _PATCH_LOCK:
            if self._prompt_id in _ACTIVE_TRELLIS_PROGRESS:
                raise RuntimeError(f"Trellis progress is already active for prompt {self._prompt_id!r}.")
            _ACTIVE_TRELLIS_PROGRESS[self._prompt_id] = self
            session = _PATCH_SESSIONS.get(self._trellis_directory)
            if session is None:
                try:
                    session = _TrellisTqdmPatchSession(self._trellis_directory)
                    session.patch_loaded_modules()
                    _PATCH_SESSIONS[self._trellis_directory] = session
                except Exception:
                    del _ACTIVE_TRELLIS_PROGRESS[self._prompt_id]
                    raise
            session.users += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        with _PATCH_LOCK:
            if _ACTIVE_TRELLIS_PROGRESS.get(self._prompt_id) is not self:
                raise RuntimeError(f"Trellis progress registration was lost for prompt {self._prompt_id!r}.")
            del _ACTIVE_TRELLIS_PROGRESS[self._prompt_id]
            session = _PATCH_SESSIONS[self._trellis_directory]
            session.users -= 1
            if session.users == 0:
                session.restore()
                del _PATCH_SESSIONS[self._trellis_directory]
        return False

    def start(self, bar: Any) -> tuple[str, str] | None:
        total = _finite_total(getattr(bar, "total", None))
        if total is None:
            return None
        from comfy_execution.utils import get_executing_context

        executing_context = get_executing_context()
        if executing_context is None:
            return None
        node_key = (str(executing_context.prompt_id), str(executing_context.node_id))
        state = self._node_progress.get(node_key)
        if state is None:
            from comfy.utils import ProgressBar

            state = _NodeProgress(ProgressBar(total, node_id=executing_context.node_id))
            self._node_progress[node_key] = state
        if state.active_bar is not None:
            previous_total = _finite_total(getattr(state.active_bar, "total", None))
            if previous_total is not None:
                state.offset += previous_total
        state.active_bar = bar
        state.progress_bar.update_absolute(state.offset, total=state.offset + total)
        return node_key

    def update(self, bar: Any, node_key: tuple[str, str]) -> None:
        state = self._node_progress.get(node_key)
        if state is None or bar is not state.active_bar:
            return
        total = _finite_total(getattr(bar, "total", None))
        if total is None:
            return
        value = float(getattr(bar, "n", 0))
        if not math.isfinite(value):
            return
        value = min(max(value, 0.0), total)
        state.progress_bar.update_absolute(state.offset + value, total=state.offset + total)
