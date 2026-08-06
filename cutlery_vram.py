from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from typing import Any

import torch


BYTES_PER_GIB = 1024**3


class _ExternalCache:
    def __init__(self, unload: Callable[[], Any], status: Callable[[], Any] | None = None) -> None:
        self.unload = unload
        self.status = status


_EXTERNAL_CACHES: dict[str, _ExternalCache] = {}


def _model_management():
    try:
        import comfy.model_management as model_management
    except Exception:
        return None
    return model_management


def resolve_external_device(requested: str, *, cuda_index_style: bool = False) -> str:
    value = str(requested or "auto").strip().lower()
    if value in {"", "auto", "default"}:
        model_management = _model_management()
        device = None
        if model_management is not None:
            try:
                device = model_management.get_torch_device()
            except Exception:
                device = None
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if getattr(device, "type", str(device)) == "cuda":
            index = getattr(device, "index", None)
            if cuda_index_style:
                return f"cuda:{0 if index is None else index}"
            return "cuda"
        return str(device)
    if value == "cuda" and cuda_index_style:
        return "cuda:0"
    return value


def _is_cuda_device(device: str) -> bool:
    return str(device or "").strip().lower().startswith("cuda")


def prepare_external_model_load(device: str, *, minimum_free_vram_gib: float = 0.0, reason: str = "") -> None:
    if not _is_cuda_device(device):
        return
    model_management = _model_management()
    if model_management is None:
        return

    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch_device.index is None:
        try:
            comfy_device = model_management.get_torch_device()
            if getattr(comfy_device, "type", None) == "cuda":
                torch_device = comfy_device
        except Exception:
            if torch.cuda.is_available():
                torch_device = torch.device("cuda", torch.cuda.current_device())
    required_bytes = max(0, int(float(minimum_free_vram_gib or 0.0) * BYTES_PER_GIB))
    try:
        cleanup_models = getattr(model_management, "cleanup_models", None)
        if callable(cleanup_models):
            cleanup_models()
        if required_bytes > 0:
            extra_reserved = getattr(model_management, "extra_reserved_memory", lambda: 0)
            try:
                required_bytes += int(extra_reserved())
            except Exception:
                pass
            model_management.free_memory(required_bytes, torch_device)
        soft_empty_cache = getattr(model_management, "soft_empty_cache", None)
        if callable(soft_empty_cache):
            soft_empty_cache()
    except Exception as exc:
        logging.warning("[Cutlery] Could not prepare ComfyUI VRAM before external model load%s: %s", f" ({reason})" if reason else "", exc)


def move_to_cpu(value: Any) -> None:
    cpu_method = getattr(value, "cpu", None)
    if callable(cpu_method):
        cpu_method()
        return

    to_method = getattr(value, "to", None)
    if callable(to_method):
        try:
            to_method("cpu")
        except TypeError:
            to_method(torch.device("cpu"))


def collect_and_empty_cache() -> None:
    gc.collect()
    model_management = _model_management()
    if model_management is not None:
        soft_empty_cache = getattr(model_management, "soft_empty_cache", None)
        if callable(soft_empty_cache):
            try:
                soft_empty_cache(force=True)
            except TypeError:
                soft_empty_cache()
            except Exception:
                pass


def register_external_model_cache(name: str, unload: Callable[[], Any], status: Callable[[], Any] | None = None) -> None:
    _EXTERNAL_CACHES[str(name)] = _ExternalCache(unload=unload, status=status)


def unload_external_model_caches() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name, cache in list(_EXTERNAL_CACHES.items()):
        try:
            results[name] = cache.unload()
        except Exception as exc:
            results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    collect_and_empty_cache()
    return results


def external_model_cache_status() -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    for name, cache in list(_EXTERNAL_CACHES.items()):
        if cache.status is None:
            statuses[name] = {"registered": True}
            continue
        try:
            statuses[name] = cache.status()
        except Exception as exc:
            statuses[name] = {"registered": True, "error": f"{type(exc).__name__}: {exc}"}
    return statuses
