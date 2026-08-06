from __future__ import annotations

import logging
import http.client
import queue
import subprocess
import threading
import time
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar


LOGGER = logging.getLogger("cutlery.interrupt")
DEFAULT_POLL_INTERVAL_S = 0.25
DEFAULT_TERMINATE_TIMEOUT_S = 10.0
HTTP_RESPONSE_READ_CHUNK_SIZE = 64 * 1024

T = TypeVar("T")


@dataclass(frozen=True)
class HttpResponseBytes:
    status: int
    reason: str
    headers: Any
    body: bytes


class HttpRequestAbortHandle:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._connection: http.client.HTTPConnection | None = None

    def set_connection(self, connection: http.client.HTTPConnection) -> None:
        with self._lock:
            self._connection = connection

    def clear_connection(self, connection: http.client.HTTPConnection) -> None:
        with self._lock:
            if self._connection is connection:
                self._connection = None

    def close(self) -> None:
        with self._lock:
            connection = self._connection
        if connection is not None:
            connection.close()


def _response_content_length(response: Any) -> int | None:
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    value = getter("Content-Length") if callable(getter) else None
    if value is None:
        getheader = getattr(response, "getheader", None)
        value = getheader("Content-Length") if callable(getheader) else None
    if value is None or not str(value).strip():
        return None
    try:
        content_length = int(str(value).strip())
    except ValueError as exc:
        raise RuntimeError("HTTP response has an invalid Content-Length header.") from exc
    if content_length < 0:
        raise RuntimeError("HTTP response has an invalid Content-Length header.")
    return content_length


def read_response_bytes(response: Any, *, max_response_bytes: int | None = None) -> bytes:
    """Read an HTTP response in bounded chunks before it reaches a JSON decoder."""

    if max_response_bytes is not None:
        if not isinstance(max_response_bytes, int) or isinstance(max_response_bytes, bool) or max_response_bytes < 0:
            raise ValueError("max_response_bytes must be a non-negative integer or None.")
        content_length = _response_content_length(response)
        if content_length is not None and content_length > max_response_bytes:
            raise RuntimeError(
                f"HTTP response declares {content_length} bytes, exceeding the {max_response_bytes}-byte limit."
            )

    chunks: list[bytes] = []
    total_bytes = 0
    while True:
        chunk = response.read(HTTP_RESPONSE_READ_CHUNK_SIZE)
        if not chunk:
            return b"".join(chunks)
        payload = bytes(chunk)
        total_bytes += len(payload)
        if max_response_bytes is not None and total_bytes > max_response_bytes:
            raise RuntimeError(f"HTTP response exceeds the {max_response_bytes}-byte limit.")
        chunks.append(payload)


def throw_if_interrupted() -> None:
    try:
        import comfy.model_management as model_management  # type: ignore
    except Exception:
        return
    throw_interrupted = getattr(model_management, "throw_exception_if_processing_interrupted", None)
    if callable(throw_interrupted):
        throw_interrupted()


def terminate_process(
    process: subprocess.Popen[Any],
    *,
    terminate_timeout_s: float = DEFAULT_TERMINATE_TIMEOUT_S,
    kill_timeout_s: float = DEFAULT_TERMINATE_TIMEOUT_S,
) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=max(0.1, float(terminate_timeout_s)))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=max(0.1, float(kill_timeout_s)))


def run_interruptible_worker(
    worker: Callable[[], T],
    *,
    abort: Callable[[], None],
    description: str,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    logger: logging.Logger | None = None,
) -> T:
    throw_if_interrupted()
    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            result_queue.put(("result", worker()))
        except BaseException as exc:
            result_queue.put(("error", exc))

    thread = threading.Thread(target=run, name="CutleryInterruptibleWorker", daemon=True)
    thread.start()
    poll_interval = max(0.01, float(poll_interval_s))
    active_logger = logger or LOGGER

    while True:
        try:
            kind, value = result_queue.get(timeout=poll_interval)
        except queue.Empty:
            try:
                throw_if_interrupted()
            except BaseException:
                active_logger.info("Cancellation detected; aborting %s", description)
                try:
                    abort()
                except Exception:
                    active_logger.warning("Failed to abort %s after cancellation", description, exc_info=True)
                thread.join(timeout=1)
                raise
            continue

        if kind == "error":
            raise value
        return value


def _request_bytes_once(
    method: str,
    url: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = 300.0,
    max_response_bytes: int | None = None,
    abort_handle: HttpRequestAbortHandle | None = None,
) -> HttpResponseBytes:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(f"Invalid HTTP URL: {url!r}")

    target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_class(parsed.netloc, timeout=max(1.0, float(timeout_s)))
    if abort_handle is not None:
        abort_handle.set_connection(connection)
    try:
        connection.request(method.upper(), target, body=body, headers=headers or {})
        response = connection.getresponse()
        response_body = read_response_bytes(response, max_response_bytes=max_response_bytes)
        return HttpResponseBytes(
            status=int(getattr(response, "status", 0) or 0),
            reason=str(getattr(response, "reason", "") or ""),
            headers=getattr(response, "headers", {}),
            body=response_body,
        )
    finally:
        if abort_handle is not None:
            abort_handle.clear_connection(connection)
        connection.close()


def request_bytes(
    method: str,
    url: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = 300.0,
    max_response_bytes: int | None = None,
    description: str | None = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    logger: logging.Logger | None = None,
    on_cancel: Callable[[], None] | None = None,
) -> HttpResponseBytes:
    abort_handle = HttpRequestAbortHandle()

    def abort_request() -> None:
        abort_handle.close()
        if on_cancel is not None:
            on_cancel()

    return run_interruptible_worker(
        lambda: _request_bytes_once(
            method,
            url,
            body=body,
            headers=headers,
            timeout_s=timeout_s,
            max_response_bytes=max_response_bytes,
            abort_handle=abort_handle,
        ),
        abort=abort_request,
        description=description or f"{method.upper()} {url}",
        poll_interval_s=poll_interval_s,
        logger=logger,
    )


def request_bytes_uninterruptible(
    method: str,
    url: str,
    *,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_s: float = 10.0,
    max_response_bytes: int | None = None,
) -> HttpResponseBytes:
    """Issue a bounded cleanup request even while ComfyUI's interrupt flag is set."""

    return _request_bytes_once(
        method,
        url,
        body=body,
        headers=headers,
        timeout_s=timeout_s,
        max_response_bytes=max_response_bytes,
    )


def run_interruptible_subprocess(
    command: Sequence[str],
    *,
    description: str,
    timeout_s: float | None = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    logger: logging.Logger | None = None,
    on_cancel: Callable[[], None] | None = None,
    **popen_kwargs: Any,
) -> subprocess.Popen[Any]:
    throw_if_interrupted()
    active_logger = logger or LOGGER
    process = subprocess.Popen(list(command), **popen_kwargs)
    started = time.monotonic()
    poll_interval = max(0.01, float(poll_interval_s))

    while process.poll() is None:
        try:
            throw_if_interrupted()
        except BaseException:
            active_logger.info("Cancellation detected; terminating %s pid=%s", description, process.pid)
            terminate_process(process)
            if on_cancel is not None:
                on_cancel()
            raise
        if timeout_s is not None and time.monotonic() - started > float(timeout_s):
            terminate_process(process)
            raise TimeoutError(f"{description} timed out after {float(timeout_s):.0f} seconds.")
        time.sleep(poll_interval)

    return process
