from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections.abc import Iterable
from typing import Any


LOGGER = logging.getLogger("cutlery.gpu_memory")
BYTES_PER_MIB = 1024**2
DEVICE_MEMORY_CACHE_SECONDS = 1.0

_DEVICE_MEMORY_CACHE_LOCK = threading.Lock()
_device_memory_cache: tuple[float, dict[str, Any]] | None = None


def _mib_to_bytes(value: str) -> int | None:
    value = value.strip()
    if not value or value.upper() == "N/A":
        return None
    try:
        return int(value) * BYTES_PER_MIB
    except ValueError:
        return None


def parse_nvidia_gpu_memory(output: str) -> list[dict[str, Any]]:
    """Parse the nounits CSV emitted by the device-memory nvidia-smi query."""
    devices = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        devices.append(
            {
                "index": index,
                "uuid": fields[1] or None,
                "name": fields[2] or None,
                "total_bytes": _mib_to_bytes(fields[3]),
                "used_bytes": _mib_to_bytes(fields[4]),
                "free_bytes": _mib_to_bytes(fields[5]),
            }
        )
    return devices


def _run_nvidia_smi(arguments: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["nvidia-smi", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.debug("NVIDIA GPU telemetry query failed: %s", exc)
        return None


def _device_memory_snapshot() -> dict[str, Any]:
    result = _run_nvidia_smi(
        [
            "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None:
        return {
            "available": False,
            "source": "nvidia-smi",
            "devices": [],
            "error": "nvidia-smi-unavailable",
        }
    if result.returncode != 0:
        LOGGER.debug("NVIDIA GPU telemetry query exited with code %s", result.returncode)
        return {
            "available": False,
            "source": "nvidia-smi",
            "devices": [],
            "error": "nvidia-smi-failed",
        }

    devices = parse_nvidia_gpu_memory(result.stdout)
    if not devices:
        return {
            "available": False,
            "source": "nvidia-smi",
            "devices": [],
            "error": "nvidia-smi-returned-no-gpus",
        }
    return {
        "available": True,
        "source": "nvidia-smi",
        "devices": devices,
        "error": None,
    }


def get_gpu_memory_snapshot(*, force_refresh: bool = False) -> dict[str, Any]:
    """Return cached physical GPU memory totals from nvidia-smi.

    Device totals are system-wide and are the only values suitable for a shared
    GPU headline. They must not be combined with process-local allocator stats.
    """
    global _device_memory_cache

    now = time.monotonic()
    with _DEVICE_MEMORY_CACHE_LOCK:
        if not force_refresh and _device_memory_cache is not None:
            cached_at, cached = _device_memory_cache
            if now - cached_at < DEVICE_MEMORY_CACHE_SECONDS:
                return {**cached, "cached": True}

        snapshot = _device_memory_snapshot()
        if snapshot["available"]:
            _device_memory_cache = (now, snapshot)
        return {**snapshot, "cached": False}


def _parse_nvidia_compute_processes(output: str, process_ids: set[int]) -> tuple[list[dict[str, int]], bool]:
    processes = []
    memory_unavailable = False
    for line in output.splitlines():
        if line.strip().upper() == "N/A":
            memory_unavailable = True
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        if pid not in process_ids:
            continue
        used_bytes = _mib_to_bytes(fields[1])
        if used_bytes is None:
            memory_unavailable = True
            continue
        processes.append({"pid": pid, "used_bytes": used_bytes})
    return processes, memory_unavailable


def get_nvidia_process_memory(process_pids: Iterable[int]) -> dict[str, Any]:
    """Return NVIDIA compute-process memory without misreporting WDDM as zero.

    NVIDIA does not expose compute-process memory for all Windows WDDM driver
    configurations. In that case ``used_bytes`` is ``None`` and the explicit
    error leaves callers free to use a Windows performance-counter fallback.
    """
    requested_pid_set = set()
    for pid in process_pids:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            continue
        if pid > 0:
            requested_pid_set.add(pid)
    requested_pids = sorted(requested_pid_set)
    if not requested_pids:
        return {
            "available": False,
            "source": "nvidia-smi",
            "processes": [],
            "used_bytes": None,
            "error": "no-process-ids",
        }

    result = _run_nvidia_smi(
        [
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if result is None:
        return {
            "available": False,
            "source": "nvidia-smi",
            "processes": [],
            "used_bytes": None,
            "error": "nvidia-smi-unavailable",
        }
    if result.returncode != 0:
        LOGGER.debug("NVIDIA process-memory query exited with code %s", result.returncode)
        return {
            "available": False,
            "source": "nvidia-smi",
            "processes": [],
            "used_bytes": None,
            "error": "nvidia-smi-process-query-failed",
        }

    processes, memory_unavailable = _parse_nvidia_compute_processes(result.stdout, set(requested_pids))
    if memory_unavailable:
        return {
            "available": False,
            "source": "nvidia-smi",
            "processes": [],
            "used_bytes": None,
            "error": "nvidia-smi-process-memory-unavailable-wddm",
        }
    return {
        "available": True,
        "source": "nvidia-smi",
        "processes": processes,
        "used_bytes": sum(item["used_bytes"] for item in processes),
        "error": None,
    }


def _reset_gpu_memory_cache() -> None:
    """Reset cached device telemetry for focused tests."""
    global _device_memory_cache
    with _DEVICE_MEMORY_CACHE_LOCK:
        _device_memory_cache = None


__all__ = [
    "BYTES_PER_MIB",
    "DEVICE_MEMORY_CACHE_SECONDS",
    "get_gpu_memory_snapshot",
    "get_nvidia_process_memory",
    "parse_nvidia_gpu_memory",
]
