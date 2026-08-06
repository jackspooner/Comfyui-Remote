from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.parse

try:
    import psutil
except ImportError:  # Optional: worker lifecycle remains available without process accounting.
    psutil = None

from .target import TrustedRemoteTarget

try:
    from ..cutlery_gpu_memory import get_nvidia_process_memory
    from ..cutlery_vram import register_external_model_cache
except ImportError:  # pragma: no cover - supports direct module imports in tests.
    from cutlery_gpu_memory import get_nvidia_process_memory
    from cutlery_vram import register_external_model_cache


LOGGER = logging.getLogger("cutlery.remote.worker")
_STARTUP_TIMEOUT_SECONDS = 180.0


class LocalWorkerLease:
    def __init__(self, manager: "LocalWorkerManager", target: TrustedRemoteTarget):
        self._manager = manager
        self._target = target
        self._released = False

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.release()

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._manager.release(self._target)


class _WorkerState:
    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.active_leases = 0
        self.shutdown_timer: threading.Timer | None = None
        self.idle_deadline: float | None = None
        self.log_handle = None


class LocalWorkerManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._states: dict[str, _WorkerState] = {}

    @staticmethod
    def _address(target: TrustedRemoteTarget) -> tuple[str, int]:
        parsed = urllib.parse.urlsplit(target.base_url)
        return str(parsed.hostname), int(parsed.port)

    @classmethod
    def _is_listening(cls, target: TrustedRemoteTarget) -> bool:
        host, port = cls._address(target)
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return True
        except OSError:
            return False

    def acquire(self, target: TrustedRemoteTarget) -> LocalWorkerLease:
        if not target.worker_python:
            return LocalWorkerLease(self, target)
        with self._lock:
            state = self._states.setdefault(target.name, _WorkerState())
            if state.shutdown_timer is not None:
                state.shutdown_timer.cancel()
                state.shutdown_timer = None
                state.idle_deadline = None
            if state.process is not None and state.process.poll() is not None:
                self._close_log(state)
                state.process = None
            if not self._is_listening(target):
                self._start(target, state)
            state.active_leases += 1
        return LocalWorkerLease(self, target)

    def _start(self, target: TrustedRemoteTarget, state: _WorkerState) -> None:
        python_path = Path(str(target.worker_python)).resolve()
        comfy_root = Path(str(target.worker_comfy_root)).resolve()
        if not python_path.is_file():
            raise RuntimeError(f"Remote worker Python does not exist: {python_path}")
        if not (comfy_root / "main.py").is_file():
            raise RuntimeError(f"Remote worker ComfyUI root does not contain main.py: {comfy_root}")
        host, port = self._address(target)
        log_path = comfy_root / "user" / "__cutlery" / "logs" / f"remote-worker-{target.name}.log"
        user_directory = comfy_root / "user" / f"remote-{target.name}"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        user_directory.mkdir(parents=True, exist_ok=True)
        state.log_handle = log_path.open("ab")
        env = os.environ.copy()
        env["CUTLERY_REMOTE_SERVER_ENABLED"] = "1"
        env["CUTLERY_REMOTE_PROXY_NODES_ENABLED"] = "0"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            str(python_path),
            str(comfy_root / "main.py"),
            "--listen",
            host,
            "--port",
            str(port),
            "--user-directory",
            str(user_directory),
            "--database-url",
            f"sqlite:///{(user_directory / 'comfyui.db').as_posix()}",
            "--disable-all-custom-nodes",
            "--whitelist-custom-nodes",
            "ComfyUI-Trellis2",
            "Cutlery-Remote",
        ]
        LOGGER.info("[Cutlery Remote] Starting local worker target=%s port=%s", target.name, port)
        state.process = subprocess.Popen(
            command,
            cwd=comfy_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=state.log_handle,
            stderr=subprocess.STDOUT,
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            ),
        )
        deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if state.process.poll() is not None:
                exit_code = state.process.returncode
                self._close_log(state)
                state.process = None
                raise RuntimeError(
                    f"Local remote worker {target.name!r} exited during startup with code {exit_code}; see {log_path}."
                )
            if self._is_listening(target):
                LOGGER.info("[Cutlery Remote] Local worker ready target=%s port=%s", target.name, port)
                return
            time.sleep(0.25)
        self._stop_state(target.name, state)
        raise TimeoutError(f"Local remote worker {target.name!r} did not listen within {_STARTUP_TIMEOUT_SECONDS:g}s.")

    def release(self, target: TrustedRemoteTarget) -> None:
        if not target.worker_python:
            return
        with self._lock:
            state = self._states.get(target.name)
            if state is None:
                return
            state.active_leases = max(0, state.active_leases - 1)
            if state.active_leases == 0 and state.process is not None:
                timer = threading.Timer(
                    target.worker_idle_seconds,
                    self._stop_if_idle,
                    args=(target.name, state),
                )
                timer.daemon = True
                state.shutdown_timer = timer
                state.idle_deadline = time.monotonic() + target.worker_idle_seconds
                timer.start()

    def _stop_if_idle(self, name: str, state: _WorkerState) -> None:
        with self._lock:
            if state.active_leases == 0:
                self._stop_state(name, state)

    def _stop_state(self, name: str, state: _WorkerState) -> None:
        if state.shutdown_timer is not None:
            state.shutdown_timer.cancel()
            state.shutdown_timer = None
        state.idle_deadline = None
        process = state.process
        state.process = None
        if process is not None and process.poll() is None:
            LOGGER.info("[Cutlery Remote] Stopping idle local worker target=%s", name)
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    LOGGER.warning("[Cutlery Remote] Worker process tree did not exit target=%s pid=%s", name, process.pid)
            else:
                process.terminate()
                try:
                    process.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        self._close_log(state)

    @staticmethod
    def _close_log(state: _WorkerState) -> None:
        if state.log_handle is not None:
            state.log_handle.close()
            state.log_handle = None

    @staticmethod
    def _process_tree_pids(pid: int) -> list[int]:
        if psutil is None:
            return [pid]
        try:
            process = psutil.Process(pid)
        except (psutil.Error, OSError):
            return []
        try:
            children = process.children(recursive=True)
        except (psutil.Error, OSError):
            children = []
        result = []
        for item in [process, *children]:
            try:
                if item.is_running():
                    result.append(item.pid)
            except (psutil.Error, OSError):
                continue
        return result

    @staticmethod
    def _process_tree_rss_bytes(process_pids: list[int]) -> int | None:
        if psutil is None:
            return None
        total = 0
        found = False
        for pid in process_pids:
            try:
                process = psutil.Process(pid)
                if process.is_running():
                    total += process.memory_info().rss
                    found = True
            except (psutil.Error, OSError):
                continue
        return total if found else None

    def status(self) -> dict[str, object]:
        """Return safe local-worker telemetry without exposing config paths or tokens."""
        with self._lock:
            workers = []
            now = time.monotonic()
            for name, state in sorted(self._states.items()):
                process = state.process
                running = process is not None and process.poll() is None
                process_pids = self._process_tree_pids(process.pid) if running else []
                process_memory = get_nvidia_process_memory(process_pids) if process_pids else {
                    "available": False,
                    "source": "nvidia-smi",
                    "processes": [],
                    "used_bytes": None,
                    "error": "worker-not-running",
                }
                workers.append(
                    {
                        "name": name,
                        "running": running,
                        "pid": process.pid if running else None,
                        "process_pids": process_pids,
                        "active_leases": state.active_leases,
                        "idle": running and state.active_leases == 0,
                        "idle_shutdown_in_seconds": (
                            max(0, int(state.idle_deadline - now))
                            if state.idle_deadline is not None
                            else None
                        ),
                        "ram_used_bytes": self._process_tree_rss_bytes(process_pids) if running else None,
                        "vram_used_bytes": process_memory["used_bytes"],
                        "vram_source": process_memory["source"],
                        "vram_error": process_memory["error"],
                    }
                )
        return {"workers": workers}

    def stop_all(self) -> dict[str, object]:
        """Force-stop owned local workers, including active leased executions."""
        with self._lock:
            stopped = []
            active = []
            for name, state in self._states.items():
                process = state.process
                if process is not None and process.poll() is None:
                    stopped.append(name)
                    if state.active_leases:
                        active.append(name)
                self._stop_state(name, state)
        if stopped:
            LOGGER.info(
                "[Cutlery Remote] Force-stopped local workers targets=%s active_targets=%s",
                ",".join(stopped),
                ",".join(active) or "none",
            )
        return {"stopped_targets": stopped, "active_targets": active, "forced": bool(active)}


LOCAL_WORKERS = LocalWorkerManager()
register_external_model_cache(
    "cutlery_local_remote_workers",
    unload=LOCAL_WORKERS.stop_all,
    status=LOCAL_WORKERS.status,
)
atexit.register(LOCAL_WORKERS.stop_all)


def lease_remote_target(target: TrustedRemoteTarget) -> LocalWorkerLease:
    return LOCAL_WORKERS.acquire(target)
