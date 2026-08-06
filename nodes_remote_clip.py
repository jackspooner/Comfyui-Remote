from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
import hashlib
import http.client as http_client
import inspect
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
import urllib.parse
import uuid

try:
    from .cutlery_config import REMOTE_CLIP_SERVER_ENV
    from .cutlery_features import feature_disabled_response
    from .cutlery_interrupt import (
        read_response_bytes,
        request_bytes as interruptible_request_bytes,
        throw_if_interrupted,
    )
    from .cutlery_lora_chain import CUTLERY_LORA_CHAIN, lora_chain_entries
    from .cutlery_remote.auth import build_auth_headers, configured_remote_token, is_authorized
    from .cutlery_remote.dotenv import env_value
    from .cutlery_remote.lora_materialization import materialize_remote_lora_file
    from .cutlery_remote.serialization import decode_value_bundle, encode_value_bundle
    from .cutlery_remote.target import resolve_trusted_remote_target
    from .cutlery_clip_gguf import list_clip_text_encoder_names, load_gguf_clip, resolve_clip_text_encoder_path
except ImportError:
    from cutlery_config import REMOTE_CLIP_SERVER_ENV
    from cutlery_features import feature_disabled_response
    from cutlery_interrupt import read_response_bytes, request_bytes as interruptible_request_bytes, throw_if_interrupted
    from cutlery_lora_chain import CUTLERY_LORA_CHAIN, lora_chain_entries
    from cutlery_remote.auth import build_auth_headers, configured_remote_token, is_authorized
    from cutlery_remote.dotenv import env_value
    from cutlery_remote.lora_materialization import materialize_remote_lora_file
    from cutlery_remote.serialization import decode_value_bundle, encode_value_bundle
    from cutlery_remote.target import resolve_trusted_remote_target
    from cutlery_clip_gguf import list_clip_text_encoder_names, load_gguf_clip, resolve_clip_text_encoder_path

try:
    from aiohttp import web
    from server import PromptServer
except Exception:
    web = None
    PromptServer = None


LOGGER = logging.getLogger("cutlery.remote.clip")
CATEGORY = "Cutlery/Remote"
REMOTE_CLIP_MODE_ENV = "CUTLERY_REMOTE_CLIP_MODE"
REMOTE_CLIP_BASE_URL_ENV = "CUTLERY_REMOTE_CLIP_BASE_URL"
REMOTE_CLIP_TIMEOUT_ENV = "CUTLERY_REMOTE_CLIP_TIMEOUT_S"
REMOTE_CLIP_ENCODE_TIMEOUT_ENV = "CUTLERY_REMOTE_CLIP_ENCODE_TIMEOUT_S"
REMOTE_CLIP_LORA_UPLOAD_LIMIT_MB_ENV = "CUTLERY_REMOTE_CLIP_LORA_UPLOAD_LIMIT_MB"
REMOTE_CLIP_FILE_UPLOAD_LIMIT_MB_ENV = "CUTLERY_REMOTE_CLIP_FILE_UPLOAD_LIMIT_MB"
REMOTE_CLIP_RESPONSE_LIMIT_MB_ENV = "CUTLERY_REMOTE_CLIP_RESPONSE_LIMIT_MB"
REMOTE_TOKEN_ENV = "CUTLERY_REMOTE_TOKEN"
REMOTE_CLIP_MODE_DIRECT = "direct"
REMOTE_CLIP_MODE_REMOTE = "remote"
REMOTE_CLIP_MATERIALIZED_LORA_DIR = "cutlery_remote"
REMOTE_CLIP_MATERIALIZED_CLIP_DIR = "cutlery_remote"
REMOTE_CLIP_MATERIALIZED_QWEN_IMAGE_DIR = "cutlery_remote/qwen"
NONE_CHOICE = "None"
CONFIGURE_REMOTE_CHOICE = "Configure Remote CLIP target"
LOADING_REMOTE_CHOICES = "Loading remote CLIP choices"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_ENCODE_TIMEOUT_SECONDS = 600.0
DEFAULT_LORA_UPLOAD_LIMIT_MB = 4096
DEFAULT_CLIP_UPLOAD_LIMIT_MB = 2048
DEFAULT_RESPONSE_LIMIT_MB = 256
REMOTE_CLIP_LORA_UPLOAD_LIMIT_BYTES = DEFAULT_LORA_UPLOAD_LIMIT_MB * 1024 * 1024
REMOTE_CLIP_FILE_UPLOAD_LIMIT_BYTES = DEFAULT_CLIP_UPLOAD_LIMIT_MB * 1024 * 1024
REMOTE_CLIP_RESPONSE_LIMIT_BYTES = DEFAULT_RESPONSE_LIMIT_MB * 1024 * 1024
REMOTE_CLIP_CONDITIONING_CACHE_MAX_ENTRIES = 32
INVENTORY_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_SECONDS
WIDGET_INVENTORY_TIMEOUT_SECONDS = 5.0
LORA_HASH_CHUNK_SIZE = 1024 * 1024
REMOTE_CLIP_UPLOAD_CHUNK_SIZE = 1024 * 1024
REMOTE_CLIP_TRANSFER_LOG_INTERVAL_BYTES = 64 * 1024 * 1024
REMOTE_CLIP_IMAGE_BUNDLE_SCHEMA = "cutlery.remote.image_file_bundle.v1"
REMOTE_CLIP_IMAGE_FILE_REF_BUNDLE_SCHEMA = "cutlery.remote.image_file_ref_bundle.v1"
CLIP_TYPES = [
    "stable_diffusion",
    "stable_cascade",
    "sd3",
    "stable_audio",
    "hunyuan_dit",
    "flux",
    "mochi",
    "hunyuan_video",
    "ltxv",
    "pixart",
    "cosmos",
    "lumina2",
    "wan",
    "hidream",
    "chroma",
    "ace",
    "omnigen2",
    "qwen_image",
    "hunyuan_image",
    "hunyuan_video_15",
    "flux2",
    "ovis",
    "kandinsky5",
    "kandinsky5_image",
    "newbie",
    "longcat_image",
    "cogvideox",
    "lens",
    "pixeldit",
    "krea2",
]

_CLIP_CACHE: dict[tuple[Any, ...], Any] = {}
_LORA_CACHE: dict[str, tuple[Any, Any]] = {}
_VAE_CACHE: dict[str, Any] = {}
_REMOTE_CLIP_CONDITIONING_CACHE: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_ACTIVE_CLIP_KEY: tuple[Any, ...] | None = None
_LORA_HASH_CACHE: dict[str, tuple[int, int, str]] = {}
REMOTE_CLIP_JOB_NODE_ID = "1"
REMOTE_CLIP_JOB_UI_KEY = "cutlery_remote_clip"


class RemoteClipUploadTooLarge(ValueError):
    pass


def _json_response(payload: dict[str, Any], status: int = 200):
    if web is None:
        return payload
    return web.json_response(payload, status=status)


def _remote_clip_server_disabled_response():
    return feature_disabled_response(
        "remote_clip_server",
        code="remote_clip_server_disabled",
        env_var=REMOTE_CLIP_SERVER_ENV,
        web_module=web,
    )


def _request_headers(request: Any) -> dict[str, str]:
    headers = getattr(request, "headers", None)
    return dict(headers or {})


def _header_value(headers: dict[str, str], name: str) -> str:
    wanted = name.casefold()
    for key, value in headers.items():
        if str(key).casefold() == wanted:
            return str(value)
    return ""


async def _request_json(request: Any) -> dict[str, Any]:
    json_fn = getattr(request, "json", None)
    if not callable(json_fn):
        return {}
    payload = json_fn()
    if inspect.isawaitable(payload):
        payload = await payload
    return payload if isinstance(payload, dict) else {}


def _authorized(request: Any) -> tuple[bool, dict[str, Any] | None, int]:
    token = configured_remote_token()
    if not token:
        LOGGER.warning("[Cutlery Remote CLIP] Request rejected because CUTLERY_REMOTE_TOKEN is not configured")
        return (
            False,
            {"ok": False, "error": "Cutlery remote token is not configured on this ComfyUI instance."},
            503,
        )
    if not is_authorized(_request_headers(request), token):
        LOGGER.warning("[Cutlery Remote CLIP] Request rejected because authorization failed")
        return False, {"ok": False, "error": "Unauthorized."}, 401
    return True, None, 200


def _clean_base_url(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "://" not in text:
        text = f"http://{text}"
    return text.rstrip("/")


def _remote_clip_mode() -> str:
    raw = str(env_value(REMOTE_CLIP_MODE_ENV, REMOTE_CLIP_MODE_DIRECT) or "").strip().lower().replace("-", "_")
    if raw in {"remote", "server", "remote_server", "this_is_remote"}:
        return REMOTE_CLIP_MODE_REMOTE
    return REMOTE_CLIP_MODE_DIRECT


def remote_clip_base_url() -> str:
    if _remote_clip_mode() == REMOTE_CLIP_MODE_REMOTE:
        return ""
    configured_target = env_value(REMOTE_CLIP_BASE_URL_ENV)
    if not configured_target:
        return ""
    return resolve_trusted_remote_target(configured_target).base_url


def _remote_clip_auth_token() -> str:
    return env_value(REMOTE_TOKEN_ENV) or configured_remote_token()


def _remote_clip_auth_headers() -> dict[str, str]:
    return build_auth_headers(_remote_clip_auth_token())


def _remote_clip_target_hint() -> str:
    if _remote_clip_mode() == REMOTE_CLIP_MODE_REMOTE:
        return (
            f"{REMOTE_CLIP_MODE_ENV}=remote marks this ComfyUI as the Remote CLIP server. "
            f"Set {REMOTE_CLIP_MODE_ENV}=direct and {REMOTE_CLIP_BASE_URL_ENV} on the client ComfyUI instead."
        )
    return f"Set {REMOTE_CLIP_BASE_URL_ENV} in the ComfyUI root .env file."


def _remote_clip_timeout(default: float = DEFAULT_TIMEOUT_SECONDS) -> float:
    raw = env_value(REMOTE_CLIP_TIMEOUT_ENV)
    if not raw:
        return default
    try:
        return max(0.25, float(raw))
    except ValueError:
        return default


def _remote_clip_encode_timeout(default: float = DEFAULT_ENCODE_TIMEOUT_SECONDS) -> float:
    raw = env_value(REMOTE_CLIP_ENCODE_TIMEOUT_ENV) or env_value(REMOTE_CLIP_TIMEOUT_ENV)
    if not raw:
        return default
    try:
        return max(0.25, float(raw))
    except ValueError:
        return default


def _remote_clip_lora_upload_limit_bytes(default: int = REMOTE_CLIP_LORA_UPLOAD_LIMIT_BYTES) -> int:
    raw = env_value(REMOTE_CLIP_LORA_UPLOAD_LIMIT_MB_ENV)
    if not raw:
        return default
    try:
        value_mb = int(str(raw).strip())
    except ValueError:
        return default
    if value_mb <= 0:
        return default
    return value_mb * 1024 * 1024


def _remote_clip_file_upload_limit_bytes(default: int = REMOTE_CLIP_FILE_UPLOAD_LIMIT_BYTES) -> int:
    raw = env_value(REMOTE_CLIP_FILE_UPLOAD_LIMIT_MB_ENV)
    if not raw:
        return default
    try:
        value_mb = int(str(raw).strip())
    except ValueError:
        return default
    if value_mb <= 0:
        return default
    return value_mb * 1024 * 1024


def _remote_clip_response_limit_bytes(default: int = REMOTE_CLIP_RESPONSE_LIMIT_BYTES) -> int:
    raw = env_value(REMOTE_CLIP_RESPONSE_LIMIT_MB_ENV)
    if not raw:
        return default
    try:
        value_mb = int(str(raw).strip())
    except ValueError:
        return default
    if value_mb <= 0:
        return default
    return value_mb * 1024 * 1024


def _raise_request_body_limit(request: Any, limit_bytes: int) -> None:
    current = int(getattr(request, "_client_max_size", 0) or 0)
    if current and current >= limit_bytes:
        return
    try:
        setattr(request, "_client_max_size", limit_bytes)
    except Exception:
        LOGGER.debug("[Cutlery Remote CLIP] Could not raise aiohttp request body limit", exc_info=True)


def _is_request_entity_too_large(error: Exception) -> bool:
    return (
        getattr(error, "status", None) == 413
        or getattr(error, "status_code", None) == 413
        or error.__class__.__name__ == "HTTPRequestEntityTooLarge"
    )


def _post_json(path: str, payload: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
    base_url = remote_clip_base_url()
    if not base_url:
        raise RuntimeError(_remote_clip_target_hint())
    url = f"{base_url}{path}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    headers.update(_remote_clip_auth_headers())
    data = json.dumps(payload).encode("utf-8")
    return _open_json("POST", url, data=data, headers=headers, timeout=timeout or _remote_clip_timeout())


def _post_bytes(path: str, payload: bytes, headers: dict[str, str], *, timeout: float | None = None) -> dict[str, Any]:
    base_url = remote_clip_base_url()
    if not base_url:
        raise RuntimeError(_remote_clip_target_hint())
    request_headers = {"Content-Type": "application/octet-stream", "Accept": "application/json"}
    request_headers.update(headers)
    request_headers.update(_remote_clip_auth_headers())
    return _open_json("POST", f"{base_url}{path}", data=payload, headers=request_headers, timeout=timeout or _remote_clip_timeout())


def _post_file(
    path: str,
    file_path: Path,
    headers: dict[str, str],
    *,
    source_name: str,
    progress: _RemoteClipUploadProgress | None = None,
    progress_label: str = "LoRA",
    timeout: float | None = None,
) -> dict[str, Any]:
    base_url = remote_clip_base_url()
    if not base_url:
        raise RuntimeError(_remote_clip_target_hint())
    url = urllib.parse.urljoin(f"{base_url.rstrip('/')}/", path.lstrip("/"))
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(f"Remote CLIP base URL is not a valid HTTP(S) URL: {base_url}")

    request_target = urllib.parse.urlunparse(("", "", parsed.path or "/", parsed.params, parsed.query, ""))
    request_headers = {
        "Content-Type": "application/octet-stream",
        "Accept": "application/json",
        "Content-Length": str(file_path.stat().st_size),
    }
    request_headers.update(headers)
    request_headers.update(_remote_clip_auth_headers())
    connection_cls = http_client.HTTPSConnection if parsed.scheme == "https" else http_client.HTTPConnection
    connection = connection_cls(parsed.hostname, parsed.port, timeout=timeout or _remote_clip_timeout())
    try:
        throw_if_interrupted()
        connection.putrequest("POST", request_target)
        for name, value in request_headers.items():
            connection.putheader(name, str(value))
        connection.endheaders()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(REMOTE_CLIP_UPLOAD_CHUNK_SIZE), b""):
                throw_if_interrupted()
                connection.send(chunk)
                if progress is not None:
                    progress.add(len(chunk), source_name=source_name, source_label=progress_label)
        throw_if_interrupted()
        response = connection.getresponse()
        raw = read_response_bytes(
            response,
            max_response_bytes=_remote_clip_response_limit_bytes(),
        ).decode("utf-8", errors="replace")
        if int(getattr(response, "status", 0) or 0) >= 400:
            raise RuntimeError(f"Remote CLIP request failed with HTTP {response.status}: {raw}")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise RuntimeError("Remote CLIP response was not a JSON object.")
        return payload
    except Exception as error:
        if isinstance(error, RuntimeError):
            raise
        raise RuntimeError(f"Remote CLIP request failed: {error}") from error
    finally:
        try:
            connection.close()
        except Exception:
            pass


def _get_json(path: str, *, timeout: float | None = None) -> dict[str, Any]:
    base_url = remote_clip_base_url()
    if not base_url:
        raise RuntimeError(_remote_clip_target_hint())
    headers = {"Accept": "application/json"}
    headers.update(_remote_clip_auth_headers())
    return _open_json("GET", f"{base_url}{path}", headers=headers, timeout=timeout or INVENTORY_TIMEOUT_SECONDS)


def _open_json(method: str, url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None, timeout: float) -> dict[str, Any]:
    try:
        response = interruptible_request_bytes(
            method,
            url,
            body=data,
            headers=headers or {},
            timeout_s=timeout,
            max_response_bytes=_remote_clip_response_limit_bytes(),
            description=f"Remote CLIP {method.upper()} {urllib.parse.urlparse(url).path or url}",
            logger=LOGGER,
        )
    except Exception as error:
        raise RuntimeError(f"Remote CLIP request failed: {error}") from error
    raw = response.body.decode("utf-8", errors="replace")
    if response.status >= 400:
        raise RuntimeError(f"Remote CLIP request failed with HTTP {response.status}: {raw}")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("Remote CLIP response was not a JSON object.")
    return payload


def _folder_filename_list(folder_name: str) -> list[str]:
    try:
        import folder_paths

        return sorted((str(name) for name in folder_paths.get_filename_list(folder_name)), key=str.casefold)
    except Exception:
        LOGGER.warning("[Cutlery Remote CLIP] Could not list ComfyUI folder %s", folder_name, exc_info=True)
        return []


def _vae_names() -> list[str]:
    vaes = list(_folder_filename_list("vae"))
    approx_vaes = _folder_filename_list("vae_approx")
    video_taes = ["taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5", "taeltx_2"]
    image_taes = ["taesd", "taesdxl", "taesd3", "taef1", "taef2"]
    have_img_encoder: set[str] = set()
    have_img_decoder: set[str] = set()

    for name in approx_vaes:
        parts = name.split("_", 1)
        if len(parts) != 2 or parts[0] not in image_taes:
            if any(name.startswith(prefix) for prefix in video_taes):
                vaes.append(name)
            continue
        if parts[1].startswith("encoder."):
            have_img_encoder.add(parts[0])
        elif parts[1].startswith("decoder."):
            have_img_decoder.add(parts[0])
    vaes.extend(name for name in have_img_decoder if name in have_img_encoder)
    vaes.append("pixel_space")
    return sorted(dict.fromkeys(str(name) for name in vaes if str(name).strip()), key=str.casefold)


def _normalize_lora_name(name: Any) -> str:
    return str(name or "").replace("\\", "/").strip().strip("/")


def _lora_full_path(lora_name: str) -> str:
    import folder_paths

    return folder_paths.get_full_path_or_raise("loras", _normalize_lora_name(lora_name))


def _lora_roots() -> list[Path]:
    try:
        import folder_paths

        return [Path(path) for path in folder_paths.get_folder_paths("loras")]
    except Exception:
        return []


def _primary_lora_root() -> Path:
    import folder_paths

    return Path(folder_paths.models_dir) / "loras"


def _normalize_clip_name(name: Any) -> str:
    return str(name or "").replace("\\", "/").strip().strip("/")


def _clip_full_path(clip_name: str) -> str:
    return resolve_clip_text_encoder_path(_normalize_clip_name(clip_name))


def _clip_roots() -> list[Path]:
    try:
        import folder_paths

        return [Path(path) for path in folder_paths.get_folder_paths("text_encoders")]
    except Exception:
        return []


def _primary_clip_root() -> Path:
    roots = _clip_roots()
    if not roots:
        raise RuntimeError("ComfyUI text_encoders folder is not configured.")
    return roots[0]


def _primary_input_root() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_input_directory())
    except Exception as error:
        raise RuntimeError("ComfyUI input folder is not configured.") from error


def _sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    stat = file_path.stat()
    cache_key = str(file_path.resolve())
    cached = _LORA_HASH_CACHE.get(cache_key)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(LORA_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _LORA_HASH_CACHE[cache_key] = (stat.st_mtime_ns, stat.st_size, value)
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _format_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    if size < 1024:
        return f"{int(size)} B"
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        size /= 1024.0
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
    return f"{size:.2f} TiB"


class _RemoteClipUploadProgress:
    def __init__(self, node_id: str | None, total_bytes: int) -> None:
        self.node_id = str(node_id) if node_id is not None else None
        self.total_bytes = max(0, int(total_bytes or 0))
        self.copied_bytes = 0
        self._last_logged_bytes = 0
        self._progress_bar = self._make_progress_bar()
        self._update_bar(0)

    def _make_progress_bar(self) -> Any:
        if not self.node_id:
            return None
        try:
            import comfy.utils as comfy_utils
        except Exception:
            return None
        progress_bar = getattr(comfy_utils, "ProgressBar", None)
        if progress_bar is None:
            return None
        total = max(1, self.total_bytes)
        try:
            return progress_bar(total, node_id=self.node_id)
        except TypeError:
            return progress_bar(total)

    def _update_bar(self, value: int) -> None:
        if self._progress_bar is None:
            return
        total = max(1, self.total_bytes)
        self._progress_bar.update_absolute(min(total, max(0, int(value))), total=total)

    def add(
        self,
        byte_count: int,
        *,
        lora_name: str | None = None,
        source_name: str | None = None,
        source_label: str = "LoRA",
    ) -> None:
        self.copied_bytes += max(0, int(byte_count))
        self._update_bar(self.copied_bytes)
        should_log = (
            self.copied_bytes >= self.total_bytes
            or self.copied_bytes - self._last_logged_bytes >= REMOTE_CLIP_TRANSFER_LOG_INTERVAL_BYTES
        )
        if should_log:
            self._last_logged_bytes = self.copied_bytes
            total = max(1, self.total_bytes)
            percent = min(100.0, (self.copied_bytes / total) * 100.0)
            LOGGER.info(
                "[Cutlery Remote CLIP] Remote %s copy progress name=%s copied=%s/%s (%.1f%%)",
                source_label,
                source_name or lora_name or "",
                _format_bytes(self.copied_bytes),
                _format_bytes(self.total_bytes),
                percent,
            )

    def finish(self) -> None:
        if self.total_bytes:
            self._update_bar(self.total_bytes)


def _lora_inventory_entry(name: str) -> dict[str, Any] | None:
    lora_name = _normalize_lora_name(name)
    if not lora_name:
        return None
    try:
        path = Path(_lora_full_path(lora_name))
        stat = path.stat()
        return {
            "name": lora_name,
            "sha256": _sha256_file(path),
            "size": stat.st_size,
            "materialized": lora_name.casefold().startswith(f"{REMOTE_CLIP_MATERIALIZED_LORA_DIR.casefold()}/"),
        }
    except Exception:
        LOGGER.warning("[Cutlery Remote CLIP] Could not hash LoRA name=%s", lora_name, exc_info=True)
        return None


def _local_lora_inventory() -> list[dict[str, Any]]:
    entries = [_lora_inventory_entry(name) for name in _folder_filename_list("loras")]
    return [entry for entry in entries if entry is not None]


def _lora_inventory_name_entry(name: str) -> dict[str, Any] | None:
    lora_name = _normalize_lora_name(name)
    if not lora_name:
        return None
    return {
        "name": lora_name,
        "sha256": "",
        "size": None,
        "materialized": lora_name.casefold().startswith(f"{REMOTE_CLIP_MATERIALIZED_LORA_DIR.casefold()}/"),
    }


def _local_lora_name_inventory() -> list[dict[str, Any]]:
    entries = [_lora_inventory_name_entry(name) for name in _folder_filename_list("loras")]
    return [entry for entry in entries if entry is not None]


def _normalize_lora_inventory_entry(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        name = _normalize_lora_name(raw)
        return {"name": name, "sha256": "", "size": None, "materialized": False} if name else None
    if not isinstance(raw, dict):
        return None
    name = _normalize_lora_name(raw.get("name") or raw.get("lora_name"))
    if not name:
        return None
    sha256 = str(raw.get("sha256") or "").strip().lower()
    size = raw.get("size")
    try:
        size = int(size) if size is not None else None
    except (TypeError, ValueError):
        size = None
    return {
        "name": name,
        "sha256": sha256,
        "size": size,
        "materialized": bool(raw.get("materialized")),
    }


def _merge_lora_inventories(local_entries: list[dict[str, Any]], remote_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    def key_for(entry: dict[str, Any], source: str) -> str:
        sha256 = str(entry.get("sha256") or "").strip().lower()
        if sha256:
            return f"sha:{sha256}"
        return f"{source}:{_normalize_lora_name(entry.get('name')).casefold()}"

    def add(entry: dict[str, Any], source: str) -> None:
        normalized = _normalize_lora_inventory_entry(entry)
        if normalized is None:
            return
        key = key_for(normalized, source)
        existing = merged.setdefault(
            key,
            {
                "display_name": normalized["name"],
                "sha256": normalized.get("sha256") or "",
                "size": normalized.get("size"),
                "local_name": None,
                "remote_name": None,
                "local_entry": None,
                "remote_entry": None,
            },
        )
        if source == "local":
            if existing.get("local_name") is None:
                existing["local_name"] = normalized["name"]
                existing["local_entry"] = normalized
                existing["display_name"] = normalized["name"]
            if normalized.get("sha256"):
                existing["sha256"] = normalized["sha256"]
            if normalized.get("size") is not None:
                existing["size"] = normalized["size"]
        else:
            if existing.get("remote_name") is None:
                existing["remote_name"] = normalized["name"]
                existing["remote_entry"] = normalized
            if existing.get("local_name") is None:
                existing["display_name"] = normalized["name"]
            if not existing.get("sha256") and normalized.get("sha256"):
                existing["sha256"] = normalized["sha256"]
            if existing.get("size") is None and normalized.get("size") is not None:
                existing["size"] = normalized["size"]

    for entry in local_entries:
        add(entry, "local")
    for entry in remote_entries:
        add(entry, "remote")

    return sorted(merged.values(), key=lambda entry: str(entry["display_name"]).casefold())


def _remote_lora_inventory_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_inventory = payload.get("lora_inventory")
    if isinstance(raw_inventory, list):
        entries = [_normalize_lora_inventory_entry(entry) for entry in raw_inventory]
        return [entry for entry in entries if entry is not None]
    entries = [_normalize_lora_inventory_entry(name) for name in payload.get("loras", [])]
    return [entry for entry in entries if entry is not None]


def _clip_inventory_entry(name: str) -> dict[str, Any] | None:
    clip_name = _normalize_clip_name(name)
    if not clip_name:
        return None
    try:
        path = Path(_clip_full_path(clip_name))
        stat = path.stat()
        return {
            "name": clip_name,
            "sha256": _sha256_file(path),
            "size": stat.st_size,
            "materialized": clip_name.casefold().startswith(f"{REMOTE_CLIP_MATERIALIZED_CLIP_DIR.casefold()}/"),
        }
    except Exception:
        LOGGER.warning("[Cutlery Remote CLIP] Could not hash CLIP/text encoder name=%s", clip_name, exc_info=True)
        return None


def _local_clip_inventory() -> list[dict[str, Any]]:
    entries = [_clip_inventory_entry(name) for name in list_clip_text_encoder_names()]
    return [entry for entry in entries if entry is not None]


def _clip_inventory_name_entry(name: str) -> dict[str, Any] | None:
    clip_name = _normalize_clip_name(name)
    if not clip_name:
        return None
    return {
        "name": clip_name,
        "sha256": "",
        "size": None,
        "materialized": clip_name.casefold().startswith(f"{REMOTE_CLIP_MATERIALIZED_CLIP_DIR.casefold()}/"),
    }


def _local_clip_name_inventory() -> list[dict[str, Any]]:
    entries = [_clip_inventory_name_entry(name) for name in list_clip_text_encoder_names()]
    return [entry for entry in entries if entry is not None]


def _normalize_clip_inventory_entry(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        name = _normalize_clip_name(raw)
        return {"name": name, "sha256": "", "size": None, "materialized": False} if name else None
    if not isinstance(raw, dict):
        return None
    name = _normalize_clip_name(raw.get("name") or raw.get("clip_name") or raw.get("text_encoder"))
    if not name:
        return None
    sha256 = str(raw.get("sha256") or "").strip().lower()
    size = raw.get("size")
    try:
        size = int(size) if size is not None else None
    except (TypeError, ValueError):
        size = None
    return {
        "name": name,
        "sha256": sha256,
        "size": size,
        "materialized": bool(raw.get("materialized")),
    }


def _merge_clip_inventories(local_entries: list[dict[str, Any]], remote_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    def key_for(entry: dict[str, Any], source: str) -> str:
        sha256 = str(entry.get("sha256") or "").strip().lower()
        if sha256:
            return f"sha:{sha256}"
        return f"{source}:{_normalize_clip_name(entry.get('name')).casefold()}"

    def add(entry: dict[str, Any], source: str) -> None:
        normalized = _normalize_clip_inventory_entry(entry)
        if normalized is None:
            return
        key = key_for(normalized, source)
        existing = merged.setdefault(
            key,
            {
                "display_name": normalized["name"],
                "sha256": normalized.get("sha256") or "",
                "size": normalized.get("size"),
                "local_name": None,
                "remote_name": None,
                "local_entry": None,
                "remote_entry": None,
            },
        )
        if source == "local":
            if existing.get("local_name") is None:
                existing["local_name"] = normalized["name"]
                existing["local_entry"] = normalized
                existing["display_name"] = normalized["name"]
            if normalized.get("sha256"):
                existing["sha256"] = normalized["sha256"]
            if normalized.get("size") is not None:
                existing["size"] = normalized["size"]
        else:
            if existing.get("remote_name") is None:
                existing["remote_name"] = normalized["name"]
                existing["remote_entry"] = normalized
            if existing.get("local_name") is None:
                existing["display_name"] = normalized["name"]
            if not existing.get("sha256") and normalized.get("sha256"):
                existing["sha256"] = normalized["sha256"]
            if existing.get("size") is None and normalized.get("size") is not None:
                existing["size"] = normalized["size"]

    for entry in local_entries:
        add(entry, "local")
    for entry in remote_entries:
        add(entry, "remote")

    return sorted(merged.values(), key=lambda entry: str(entry["display_name"]).casefold())


def _remote_clip_file_inventory_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_inventory = payload.get("clip_inventory")
    if isinstance(raw_inventory, list):
        entries = [_normalize_clip_inventory_entry(entry) for entry in raw_inventory]
        return [entry for entry in entries if entry is not None]
    entries = [_normalize_clip_inventory_entry(name) for name in payload.get("text_encoders", [])]
    return [entry for entry in entries if entry is not None]


def local_remote_clip_inventory(*, include_hashes: bool = True) -> dict[str, Any]:
    lora_inventory = _local_lora_inventory() if include_hashes else _local_lora_name_inventory()
    clip_inventory = _local_clip_inventory() if include_hashes else _local_clip_name_inventory()
    return {
        "ok": True,
        "text_encoders": [entry["name"] for entry in clip_inventory],
        "clip_inventory": clip_inventory,
        "loras": [entry["name"] for entry in lora_inventory],
        "lora_inventory": lora_inventory,
        "clip_types": list(CLIP_TYPES),
        "vaes": _vae_names(),
    }


def fetch_remote_clip_inventory(timeout: float | None = None, *, include_hashes: bool = True) -> dict[str, Any]:
    path = "/cutlery/remote/clip/inventory"
    if not include_hashes:
        path = f"{path}?include_hashes=0"
    payload = _get_json(path, timeout=timeout or INVENTORY_TIMEOUT_SECONDS)
    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("error") or "Remote CLIP inventory request failed."))
    lora_inventory = _remote_lora_inventory_from_payload(payload)
    clip_inventory = _remote_clip_file_inventory_from_payload(payload)
    return {
        "text_encoders": [entry["name"] for entry in clip_inventory] or [str(item) for item in payload.get("text_encoders", []) if str(item).strip()],
        "clip_inventory": clip_inventory,
        "loras": [entry["name"] for entry in lora_inventory] or [str(item) for item in payload.get("loras", []) if str(item).strip()],
        "lora_inventory": lora_inventory,
        "clip_types": [str(item) for item in payload.get("clip_types", []) if str(item).strip()] or list(CLIP_TYPES),
        "vaes": [str(item) for item in payload.get("vaes", []) if str(item).strip()],
    }


def remote_clip_widget_choices(timeout: float | None = None) -> dict[str, Any]:
    if _remote_clip_mode() == REMOTE_CLIP_MODE_REMOTE:
        inventory = local_remote_clip_inventory(include_hashes=False)
        text_encoders = [str(item) for item in inventory.get("text_encoders", []) if str(item).strip()]
        clip_types = [str(item) for item in inventory.get("clip_types", []) if str(item).strip()] or list(CLIP_TYPES)
        vaes = [str(item) for item in inventory.get("vaes", []) if str(item).strip()]
        return {"ok": True, "text_encoders": text_encoders, "clip_types": clip_types, "vaes": vaes}
    inventory = fetch_remote_clip_inventory(timeout=timeout or WIDGET_INVENTORY_TIMEOUT_SECONDS, include_hashes=False)
    text_encoders = [str(item) for item in inventory.get("text_encoders", []) if str(item).strip()]
    clip_types = [str(item) for item in inventory.get("clip_types", []) if str(item).strip()] or list(CLIP_TYPES)
    vaes = [str(item) for item in inventory.get("vaes", []) if str(item).strip()]
    return {"ok": True, "text_encoders": text_encoders, "clip_types": clip_types, "vaes": vaes}


def _inventory_for_widgets() -> dict[str, Any]:
    if _remote_clip_mode() != REMOTE_CLIP_MODE_REMOTE:
        placeholder = LOADING_REMOTE_CHOICES if remote_clip_base_url() else CONFIGURE_REMOTE_CHOICE
        return {"text_encoders": [placeholder], "loras": [], "clip_types": list(CLIP_TYPES), "vaes": [placeholder]}
    try:
        text_encoders = sorted(
            {str(name or "").strip() for name in list_clip_text_encoder_names() if str(name or "").strip()},
            key=str.casefold,
        )
    except Exception as error:
        LOGGER.warning("[Cutlery Remote CLIP] Falling back to placeholder widget inventory: %s", error)
        text_encoders = []
    return {"text_encoders": text_encoders or [CONFIGURE_REMOTE_CHOICE], "loras": [], "clip_types": list(CLIP_TYPES), "vaes": _vae_names()}


def _normalize_lora_chain_entries(lora_chain: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for raw_entry in lora_chain_entries(lora_chain):
        name = _normalize_lora_name(raw_entry.get("lora_name") or raw_entry.get("name"))
        if not name or name == NONE_CHOICE:
            continue
        strength = float(raw_entry.get("strength_clip", 1.0))
        if strength == 0:
            continue
        entries.append({"lora_name": name, "strength_clip": strength})
    return entries


def _mirrored_lora_relative_name(source_name: str) -> str:
    relative_name = str(source_name or "").replace("\\", "/").strip()
    if not relative_name or relative_name.startswith("/") or re.match(r"^[A-Za-z]:", relative_name):
        raise ValueError("LoRA name must be a relative path under the ComfyUI loras folder.")
    parts = relative_name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("LoRA name must not contain empty, current-directory, or parent-directory path components.")
    return "/".join(parts)


def _materialized_file_basename_key(name: Any) -> str:
    basename = Path(_normalize_lora_name(name)).name
    basename = re.sub(r"^[0-9a-fA-F]{12}-", "", basename)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._").casefold()


def _materialize_lora_bytes(
    source_name: str,
    payload: bytes,
    expected_sha256: str = "",
    limit_bytes: int | None = None,
) -> dict[str, Any]:
    effective_limit = _remote_clip_lora_upload_limit_bytes() if limit_bytes is None else int(limit_bytes)
    if len(payload) > effective_limit:
        raise RemoteClipUploadTooLarge(
            f"Uploaded LoRA exceeds the {effective_limit}-byte limit."
        )
    sha256 = _sha256_bytes(payload)
    expected = str(expected_sha256 or "").strip().lower()
    if expected and expected != sha256:
        raise ValueError("Uploaded LoRA SHA-256 did not match the expected hash.")

    relative_name = _mirrored_lora_relative_name(source_name)
    root = _primary_lora_root().resolve()
    target = (root / Path(relative_name)).resolve()
    if os.path.commonpath([str(root), str(target)]) != str(root):
        raise ValueError("LoRA path escaped the remote ComfyUI loras folder.")

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _sha256_file(target) == sha256:
            return {"ok": True, "name": relative_name, "sha256": sha256, "size": target.stat().st_size, "materialized": True}
        raise ValueError(f"Remote LoRA path already exists with different contents: {relative_name}")

    tmp_handle = tempfile.NamedTemporaryFile("wb", delete=False, dir=target.parent, prefix=".cutlery-upload-", suffix=".tmp")
    tmp_path = Path(tmp_handle.name)
    try:
        with tmp_handle as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        _LORA_HASH_CACHE.pop(str(target), None)
        LOGGER.info("[Cutlery Remote CLIP] Mirrored remote LoRA name=%s bytes=%s sha256=%s", relative_name, len(payload), sha256)
        return {"ok": True, "name": relative_name, "sha256": sha256, "size": len(payload), "materialized": True}
    finally:
        tmp_path.unlink(missing_ok=True)


async def _materialize_lora_upload(
    source_name: str,
    stream: Any,
    expected_sha256: str = "",
    limit_bytes: int | None = None,
) -> dict[str, Any]:
    effective_limit = _remote_clip_lora_upload_limit_bytes() if limit_bytes is None else int(limit_bytes)
    root = _primary_lora_root().resolve()
    relative_name = _mirrored_lora_relative_name(source_name)
    target = (root / Path(relative_name)).resolve()
    if os.path.commonpath([str(root), str(target)]) != str(root):
        raise ValueError("LoRA path escaped the remote ComfyUI loras folder.")
    target.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    total_size = 0
    tmp_handle = tempfile.NamedTemporaryFile("wb", delete=False, dir=target.parent, prefix=".cutlery-upload-", suffix=".tmp")
    tmp_path = Path(tmp_handle.name)
    try:
        with tmp_handle as handle:
            async for chunk in stream.iter_chunked(REMOTE_CLIP_UPLOAD_CHUNK_SIZE):
                data = bytes(chunk or b"")
                if not data:
                    continue
                if total_size + len(data) > effective_limit:
                    raise RemoteClipUploadTooLarge(
                        f"Uploaded LoRA exceeds the {effective_limit}-byte limit."
                    )
                handle.write(data)
                digest.update(data)
                total_size += len(data)
            handle.flush()
            os.fsync(handle.fileno())
        if total_size == 0:
            raise ValueError("Uploaded LoRA is empty.")

        sha256 = digest.hexdigest()
        expected = str(expected_sha256 or "").strip().lower()
        if expected and expected != sha256:
            raise ValueError("Uploaded LoRA SHA-256 did not match the expected hash.")
        if target.exists():
            if _sha256_file(target) == sha256:
                tmp_path.unlink(missing_ok=True)
                return {"ok": True, "name": relative_name, "sha256": sha256, "size": target.stat().st_size, "materialized": True}
            raise ValueError(f"Remote LoRA path already exists with different contents: {relative_name}")
        os.replace(tmp_path, target)
        _LORA_HASH_CACHE.pop(str(target), None)
        LOGGER.info(
            "[Cutlery Remote CLIP] Mirrored remote LoRA name=%s bytes=%s sha256=%s",
            relative_name,
            total_size,
            sha256,
        )
        return {"ok": True, "name": relative_name, "sha256": sha256, "size": total_size, "materialized": True}
    finally:
        tmp_path.unlink(missing_ok=True)


def _materialize_lora_to_remote(
    merged_entry: dict[str, Any],
    *,
    progress_node_id: str | None = None,
    progress: _RemoteClipUploadProgress | None = None,
) -> dict[str, Any]:
    local_name = str(merged_entry.get("local_name") or "").strip()
    if not local_name:
        raise RuntimeError("Selected LoRA is not installed locally and cannot be materialized.")
    local_path = Path(_lora_full_path(local_name))
    size = local_path.stat().st_size
    sha256 = str(merged_entry.get("sha256") or _sha256_file(local_path)).lower()
    upload_progress = progress or _RemoteClipUploadProgress(progress_node_id, size)
    LOGGER.info(
        "[Cutlery Remote CLIP] Copying LoRA to remote name=%s bytes=%s sha256=%s",
        local_name,
        _format_bytes(size),
        sha256,
    )
    response = materialize_remote_lora_file(
        remote_clip_base_url(),
        local_path,
        local_name,
        auth_headers=_remote_clip_auth_headers(),
        timeout_seconds=_remote_clip_timeout(),
        sha256=sha256,
        chunk_size=REMOTE_CLIP_UPLOAD_CHUNK_SIZE,
        check_cancelled=throw_if_interrupted,
        on_chunk=lambda copied: upload_progress.add(copied, source_name=local_name, source_label="LoRA"),
        max_response_bytes=_remote_clip_response_limit_bytes(),
    )
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "Remote LoRA materialization failed."))
    LOGGER.info(
        "[Cutlery Remote CLIP] Remote LoRA materialization complete source=%s remote_name=%s copied=%s sha256=%s",
        local_name,
        response.get("name"),
        _format_bytes(upload_progress.copied_bytes or size),
        sha256,
    )
    if progress is None:
        upload_progress.finish()
    return response


def _selected_local_lora_entry(selected: str) -> dict[str, Any] | None:
    local_name = _normalize_lora_name(selected)
    if not local_name:
        return None
    try:
        local_path = Path(_lora_full_path(local_name))
        stat = local_path.stat()
        return {"local_name": local_name, "sha256": _sha256_file(local_path), "size": stat.st_size}
    except Exception:
        return None


def _prepare_remote_lora_entries(entries: list[dict[str, Any]], *, progress_node_id: str | None = None) -> list[dict[str, Any]]:
    if not entries:
        return []
    remote_inventory = fetch_remote_clip_inventory(timeout=_remote_clip_encode_timeout(), include_hashes=False)

    remote_by_name: dict[str, str] = {}
    for remote_entry in remote_inventory.get("lora_inventory", []):
        remote_name = _normalize_lora_name(remote_entry.get("name") if isinstance(remote_entry, dict) else remote_entry)
        if not remote_name:
            continue
        remote_by_name.setdefault(remote_name.casefold(), remote_name)

    prepared: list[dict[str, Any]] = []
    for entry in entries:
        selected = _normalize_lora_name(entry.get("lora_name"))
        remote_name = ""
        if selected:
            remote_name = remote_by_name.get(selected.casefold(), "")
        if not remote_name:
            local_entry = _selected_local_lora_entry(selected)
            if local_entry is not None:
                materialized = _materialize_lora_to_remote(local_entry, progress_node_id=progress_node_id)
                remote_name = _normalize_lora_name(materialized.get("name"))
        prepared.append({"lora_name": str(remote_name or selected), "strength_clip": float(entry.get("strength_clip", 1.0))})
    return prepared


def _materialized_clip_relative_name(source_name: str, sha256: str) -> str:
    basename = Path(_normalize_clip_name(source_name)).name or "clip.safetensors"
    safe_basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._") or "clip.safetensors"
    return f"{REMOTE_CLIP_MATERIALIZED_CLIP_DIR}/{sha256[:12]}-{safe_basename}"


async def _materialize_clip_upload(
    source_name: str,
    stream: Any,
    expected_sha256: str = "",
    limit_bytes: int | None = None,
) -> dict[str, Any]:
    effective_limit = _remote_clip_file_upload_limit_bytes() if limit_bytes is None else int(limit_bytes)
    root = _primary_clip_root().resolve()
    materialized_root = (root / REMOTE_CLIP_MATERIALIZED_CLIP_DIR).resolve()
    materialized_root.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    total_size = 0
    tmp_handle = tempfile.NamedTemporaryFile("wb", delete=False, dir=materialized_root, prefix=".upload-", suffix=".tmp")
    tmp_path = Path(tmp_handle.name)
    try:
        with tmp_handle as handle:
            async for chunk in stream.iter_chunked(REMOTE_CLIP_UPLOAD_CHUNK_SIZE):
                data = bytes(chunk or b"")
                if not data:
                    continue
                if total_size + len(data) > effective_limit:
                    raise RemoteClipUploadTooLarge(
                        f"Uploaded CLIP/text encoder exceeds the {effective_limit}-byte limit."
                    )
                handle.write(data)
                digest.update(data)
                total_size += len(data)

        sha256 = digest.hexdigest()
        expected = str(expected_sha256 or "").strip().lower()
        if expected and expected != sha256:
            raise ValueError("Uploaded CLIP/text encoder SHA-256 did not match the expected hash.")

        relative_name = _materialized_clip_relative_name(source_name, sha256)
        target = (root / Path(relative_name)).resolve()
        if os.path.commonpath([str(materialized_root), str(target)]) != str(materialized_root):
            raise ValueError("Materialized CLIP/text encoder path escaped the remote materialized CLIP directory.")

        if target.exists() and _sha256_file(target) == sha256:
            tmp_path.unlink(missing_ok=True)
            return {"ok": True, "name": relative_name, "sha256": sha256, "size": target.stat().st_size, "materialized": True}

        tmp_path.replace(target)
        _LORA_HASH_CACHE.pop(str(target.resolve()), None)
        LOGGER.info(
            "[Cutlery Remote CLIP] Materialized remote CLIP/text encoder name=%s bytes=%s sha256=%s",
            relative_name,
            _format_bytes(total_size),
            sha256,
        )
        return {"ok": True, "name": relative_name, "sha256": sha256, "size": total_size, "materialized": True}
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _materialize_clip_to_remote(
    merged_entry: dict[str, Any],
    *,
    progress_node_id: str | None = None,
    progress: _RemoteClipUploadProgress | None = None,
) -> dict[str, Any]:
    local_name = str(merged_entry.get("local_name") or "").strip()
    if not local_name:
        raise RuntimeError("Selected CLIP/text encoder is not installed locally and cannot be materialized.")
    local_path = Path(_clip_full_path(local_name))
    size = local_path.stat().st_size
    sha256 = str(merged_entry.get("sha256") or _sha256_file(local_path)).lower()
    upload_progress = progress or _RemoteClipUploadProgress(progress_node_id, size)
    LOGGER.info(
        "[Cutlery Remote CLIP] Copying CLIP/text encoder to remote name=%s bytes=%s sha256=%s",
        local_name,
        _format_bytes(size),
        sha256,
    )
    response = _post_file(
        "/cutlery/remote/clip/clips/materialize",
        local_path,
        {
            "X-Cutlery-Clip-Name": urllib.parse.quote(local_name, safe="/.-_"),
            "X-Cutlery-Clip-SHA256": sha256,
        },
        source_name=local_name,
        progress=upload_progress,
        progress_label="CLIP",
    )
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "Remote CLIP/text encoder materialization failed."))
    LOGGER.info(
        "[Cutlery Remote CLIP] Remote CLIP/text encoder materialization complete source=%s remote_name=%s copied=%s sha256=%s",
        local_name,
        response.get("name"),
        _format_bytes(upload_progress.copied_bytes or size),
        sha256,
    )
    if progress is None:
        upload_progress.finish()
    return response


def _selected_local_clip_entry(selected: str) -> dict[str, Any] | None:
    local_name = _normalize_clip_name(selected)
    if not local_name:
        return None
    try:
        local_path = Path(_clip_full_path(local_name))
        stat = local_path.stat()
        return {"local_name": local_name, "sha256": _sha256_file(local_path), "size": stat.st_size}
    except Exception:
        return None


def _prepare_remote_clip_names(clip_names: list[str], *, progress_node_id: str | None = None) -> list[str]:
    normalized_names = [_normalize_clip_name(name) for name in clip_names]
    if not normalized_names:
        return []
    remote_inventory = fetch_remote_clip_inventory(timeout=_remote_clip_encode_timeout(), include_hashes=False)

    remote_by_name: dict[str, str] = {}
    remote_materialized_by_basename: dict[str, str] = {}
    for remote_entry in remote_inventory.get("clip_inventory", []):
        remote_name = _normalize_clip_name(remote_entry.get("name") if isinstance(remote_entry, dict) else remote_entry)
        if remote_name:
            remote_by_name.setdefault(remote_name.casefold(), remote_name)
            if remote_name.casefold().startswith(f"{REMOTE_CLIP_MATERIALIZED_CLIP_DIR.casefold()}/"):
                basename_key = _materialized_file_basename_key(remote_name)
                if basename_key:
                    remote_materialized_by_basename.setdefault(basename_key, remote_name)

    prepared: list[str] = []
    for selected in normalized_names:
        remote_name = ""
        if selected:
            remote_name = remote_by_name.get(selected.casefold(), "")
        if not remote_name:
            remote_name = remote_materialized_by_basename.get(_materialized_file_basename_key(selected), "")
        if not remote_name:
            local_entry = _selected_local_clip_entry(selected)
            if local_entry is not None:
                materialized = _materialize_clip_to_remote(local_entry, progress_node_id=progress_node_id)
                remote_name = _normalize_clip_name(materialized.get("name"))
        prepared.append(remote_name or selected)
    return prepared


def _clear_materialized_loras() -> dict[str, Any]:
    materialized_root = (_primary_lora_root() / REMOTE_CLIP_MATERIALIZED_LORA_DIR).resolve()
    deleted_count = 0
    if materialized_root.exists():
        for path in sorted(materialized_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            resolved = path.resolve()
            if os.path.commonpath([str(materialized_root), str(resolved)]) != str(materialized_root):
                continue
            if path.is_file():
                path.unlink()
                deleted_count += 1
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            materialized_root.rmdir()
        except OSError:
            pass
    _LORA_CACHE.clear()
    _clear_remote_conditioning_cache()
    _collect_and_empty_cache()
    LOGGER.info("[Cutlery Remote CLIP] Cleared materialized remote LoRAs deleted_count=%s", deleted_count)
    return {"ok": True, "deleted_count": deleted_count}


def clear_remote_materialized_loras() -> dict[str, Any]:
    if remote_clip_base_url():
        response = _post_json("/cutlery/remote/clip/loras/clear", {})
        if response.get("ok"):
            _clear_remote_conditioning_cache()
        return response
    return _clear_materialized_loras()


def _clear_materialized_clips() -> dict[str, Any]:
    global _ACTIVE_CLIP_KEY

    materialized_root = (_primary_clip_root() / REMOTE_CLIP_MATERIALIZED_CLIP_DIR).resolve()
    deleted_count = 0
    if materialized_root.exists():
        for path in sorted(materialized_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            resolved = path.resolve()
            if os.path.commonpath([str(materialized_root), str(resolved)]) != str(materialized_root):
                continue
            if path.is_file():
                path.unlink()
                deleted_count += 1
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            materialized_root.rmdir()
        except OSError:
            pass
    _CLIP_CACHE.clear()
    _clear_remote_conditioning_cache()
    _ACTIVE_CLIP_KEY = None
    _collect_and_empty_cache()
    LOGGER.info("[Cutlery Remote CLIP] Cleared materialized remote CLIP/text encoders deleted_count=%s", deleted_count)
    return {"ok": True, "deleted_count": deleted_count}


def clear_remote_materialized_clips() -> dict[str, Any]:
    if remote_clip_base_url():
        response = _post_json("/cutlery/remote/clip/clips/clear", {})
        if response.get("ok"):
            _clear_remote_conditioning_cache()
        return response
    return _clear_materialized_clips()


def _clear_materialized_qwen_images() -> dict[str, Any]:
    materialized_root = (_primary_input_root() / REMOTE_CLIP_MATERIALIZED_QWEN_IMAGE_DIR).resolve()
    deleted_count = 0
    if materialized_root.exists():
        for path in sorted(materialized_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            resolved = path.resolve()
            if os.path.commonpath([str(materialized_root), str(resolved)]) != str(materialized_root):
                continue
            if path.is_file():
                path.unlink()
                deleted_count += 1
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            materialized_root.rmdir()
        except OSError:
            pass
    _clear_remote_conditioning_cache()
    LOGGER.info("[Cutlery Remote CLIP] Cleared materialized remote Qwen input images deleted_count=%s", deleted_count)
    return {"ok": True, "deleted_count": deleted_count}


def clear_remote_materialized_qwen_images() -> dict[str, Any]:
    if remote_clip_base_url():
        response = _post_json("/cutlery/remote/clip/images/clear", {})
        if response.get("ok"):
            _clear_remote_conditioning_cache()
        return response
    return _clear_materialized_qwen_images()


def post_remote_clip_encode(payload: dict[str, Any]) -> dict[str, Any]:
    response = _post_json("/cutlery/remote/clip/text-encode", payload, timeout=_remote_clip_encode_timeout())
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "Remote CLIP text encode failed."))
    bundle = response.get("conditioning")
    if not isinstance(bundle, dict):
        raise RuntimeError("Remote CLIP text encode response did not include a conditioning bundle.")
    return bundle


def post_remote_dual_clip_encode(payload: dict[str, Any]) -> dict[str, Any]:
    response = _post_json("/cutlery/remote/clip/dual-text-encode", payload, timeout=_remote_clip_encode_timeout())
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "Remote dual CLIP text encode failed."))
    bundle = response.get("conditioning")
    if not isinstance(bundle, dict):
        raise RuntimeError("Remote dual CLIP text encode response did not include a conditioning bundle.")
    return bundle


def post_remote_qwen_image_edit_plus_encode(payload: dict[str, Any]) -> dict[str, Any]:
    response = _post_json("/cutlery/remote/clip/qwen-image-edit-plus", payload, timeout=_remote_clip_encode_timeout())
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "Remote Qwen image edit text encode failed."))
    bundle = response.get("conditioning")
    if not isinstance(bundle, dict):
        raise RuntimeError("Remote Qwen image edit text encode response did not include a conditioning bundle.")
    return bundle


def _normalized_loras_for_conditioning_cache(loras: Any) -> tuple[tuple[str, float], ...]:
    if not isinstance(loras, list):
        return ()
    normalized: list[tuple[str, float]] = []
    for entry in loras:
        if not isinstance(entry, dict):
            continue
        lora_name = _normalize_lora_name(entry.get("lora_name") or entry.get("name"))
        if not lora_name:
            continue
        normalized.append((lora_name, float(entry.get("strength_clip", 1.0))))
    return tuple(normalized)


def _normalized_qwen_images_for_conditioning_cache(images: Any) -> tuple[Any, ...]:
    if not isinstance(images, dict):
        return ()
    normalized: list[Any] = []
    for image_name in ("image1", "image2", "image3"):
        bundle = images.get(image_name)
        if not isinstance(bundle, dict):
            continue
        frames = []
        for frame in bundle.get("frames") or []:
            if not isinstance(frame, dict):
                continue
            frames.append(
                (
                    str(frame.get("name") or frame.get("filename") or ""),
                    str(frame.get("type") or ""),
                    str(frame.get("sha256") or ""),
                    int(frame.get("byte_count") or frame.get("size") or 0),
                )
            )
        normalized.append(
            (
                image_name,
                str(bundle.get("schema") or ""),
                int(bundle.get("width") or 0),
                int(bundle.get("height") or 0),
                tuple(frames),
            )
        )
    return tuple(normalized)


def _remote_conditioning_cache_key(kind: str, payload: dict[str, Any]) -> tuple[Any, ...]:
    if kind == "dual":
        encoder_key = (
            str(payload.get("clip_name1") or ""),
            str(payload.get("clip_name2") or ""),
        )
    elif kind == "qwen_image_edit_plus":
        encoder_key = (
            str(payload.get("text_encoder") or ""),
            str(payload.get("vae_name") or ""),
            _normalized_qwen_images_for_conditioning_cache(payload.get("images")),
        )
    else:
        encoder_key = (str(payload.get("text_encoder") or ""),)
    return (
        kind,
        _remote_clip_mode(),
        remote_clip_base_url(),
        *encoder_key,
        str(payload.get("clip_type") or ""),
        str(payload.get("prompt") or ""),
        _normalized_loras_for_conditioning_cache(payload.get("loras")),
    )


def _get_cached_remote_conditioning(cache_key: tuple[Any, ...]) -> dict[str, Any] | None:
    cached = _REMOTE_CLIP_CONDITIONING_CACHE.get(cache_key)
    if cached is None:
        return None
    _REMOTE_CLIP_CONDITIONING_CACHE.move_to_end(cache_key)
    return cached


def _remember_remote_conditioning(cache_key: tuple[Any, ...], bundle: dict[str, Any]) -> None:
    _REMOTE_CLIP_CONDITIONING_CACHE[cache_key] = bundle
    _REMOTE_CLIP_CONDITIONING_CACHE.move_to_end(cache_key)
    while len(_REMOTE_CLIP_CONDITIONING_CACHE) > REMOTE_CLIP_CONDITIONING_CACHE_MAX_ENTRIES:
        _REMOTE_CLIP_CONDITIONING_CACHE.popitem(last=False)


def _clear_remote_conditioning_cache() -> None:
    _REMOTE_CLIP_CONDITIONING_CACHE.clear()


def _clip_type(clip_type: str):
    import comfy.sd

    return getattr(comfy.sd.CLIPType, str(clip_type or "stable_diffusion").upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)


def _clip_cache_key(text_encoder: str, clip_type: str, device: str) -> tuple[str, str, str]:
    return (str(text_encoder), str(clip_type), str(device or "default"))


def _dual_clip_cache_key(clip_name1: str, clip_name2: str, clip_type: str, device: str) -> tuple[str, str, str, str, str]:
    return ("dual", str(clip_name1), str(clip_name2), str(clip_type), str(device or "default"))


def _collect_and_empty_cache() -> None:
    try:
        from .cutlery_vram import collect_and_empty_cache
    except ImportError:
        from cutlery_vram import collect_and_empty_cache

    collect_and_empty_cache()


def unload_remote_clip_cache() -> dict[str, Any]:
    global _ACTIVE_CLIP_KEY

    previous_key = _ACTIVE_CLIP_KEY
    had_cache = bool(_CLIP_CACHE or _LORA_CACHE or _VAE_CACHE or _REMOTE_CLIP_CONDITIONING_CACHE or previous_key is not None)
    _CLIP_CACHE.clear()
    _LORA_CACHE.clear()
    _VAE_CACHE.clear()
    _clear_remote_conditioning_cache()
    _ACTIVE_CLIP_KEY = None
    _collect_and_empty_cache()
    LOGGER.info("[Cutlery Remote CLIP] Unloaded text encoder cache previous_key=%s", previous_key)
    return {
        "ok": True,
        "unloaded": had_cache,
        "previous_clip_key": list(previous_key) if previous_key is not None else None,
    }


def _ensure_active_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    global _ACTIVE_CLIP_KEY

    if _ACTIVE_CLIP_KEY is not None and _ACTIVE_CLIP_KEY != key:
        LOGGER.info(
            "[Cutlery Remote CLIP] Switching text encoder old_key=%s new_key=%s; unloading previous cache first",
            _ACTIVE_CLIP_KEY,
            key,
        )
        unload_remote_clip_cache()
    return key


def _ensure_active_clip_key(text_encoder: str, clip_type: str, device: str) -> tuple[str, str, str]:
    return _ensure_active_key(_clip_cache_key(text_encoder, clip_type, device))


def _ensure_active_dual_clip_key(clip_name1: str, clip_name2: str, clip_type: str, device: str) -> tuple[str, str, str, str, str]:
    return _ensure_active_key(_dual_clip_cache_key(clip_name1, clip_name2, clip_type, device))


def _load_clip_for_remote_encode(text_encoder: str, clip_type: str, device: str = "default"):
    import comfy.sd
    import folder_paths
    import torch

    key = _clip_cache_key(text_encoder, clip_type, device)
    cached = _CLIP_CACHE.get(key)
    if cached is not None:
        LOGGER.info("[Cutlery Remote CLIP] Reusing cached text encoder name=%s type=%s", text_encoder, clip_type)
        return cached

    clip_path = resolve_clip_text_encoder_path(text_encoder)
    model_options: dict[str, Any] = {}
    if device == "cpu":
        model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
    is_gguf = clip_path.lower().endswith(".gguf")
    LOGGER.info(
        "[Cutlery Remote CLIP] Loading text encoder name=%s type=%s path=%s gguf=%s",
        text_encoder,
        clip_type,
        clip_path,
        is_gguf,
    )
    if is_gguf:
        clip = load_gguf_clip([clip_path], _clip_type(clip_type))
        _CLIP_CACHE[key] = clip
        return clip

    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=_clip_type(clip_type),
        model_options=model_options,
    )
    _CLIP_CACHE[key] = clip
    return clip


def _load_dual_clip_for_remote_encode(clip_name1: str, clip_name2: str, clip_type: str, device: str = "default"):
    import comfy.sd
    import folder_paths
    import torch

    key = _dual_clip_cache_key(clip_name1, clip_name2, clip_type, device)
    cached = _CLIP_CACHE.get(key)
    if cached is not None:
        LOGGER.info(
            "[Cutlery Remote CLIP] Reusing cached dual text encoder clip1=%s clip2=%s type=%s",
            clip_name1,
            clip_name2,
            clip_type,
        )
        return cached

    clip_paths = (_clip_full_path(clip_name1), _clip_full_path(clip_name2))
    model_options: dict[str, Any] = {}
    if device == "cpu":
        model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")
    uses_gguf = any(path.lower().endswith(".gguf") for path in clip_paths)
    LOGGER.info(
        "[Cutlery Remote CLIP] Loading dual text encoder clip1=%s clip2=%s type=%s paths=%s gguf=%s",
        clip_name1,
        clip_name2,
        clip_type,
        clip_paths,
        uses_gguf,
    )
    if uses_gguf:
        clip = load_gguf_clip(clip_paths, _clip_type(clip_type))
        _CLIP_CACHE[key] = clip
        return clip

    clip = comfy.sd.load_clip(
        ckpt_paths=list(clip_paths),
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=_clip_type(clip_type),
        model_options=model_options,
    )
    _CLIP_CACHE[key] = clip
    return clip


def _load_vae_for_remote_encode(vae_name: str | None):
    name = str(vae_name or "").strip()
    if not name or name == NONE_CHOICE:
        return None
    cached = _VAE_CACHE.get(name)
    if cached is not None:
        LOGGER.info("[Cutlery Remote CLIP] Reusing cached VAE name=%s", name)
        return cached

    import nodes

    LOGGER.info("[Cutlery Remote CLIP] Loading VAE name=%s", name)
    vae = nodes.VAELoader().load_vae(name)[0]
    _VAE_CACHE[name] = vae
    return vae


def _load_lora_file(lora_name: str) -> tuple[Any, Any]:
    import comfy.utils
    import folder_paths

    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    cached = _LORA_CACHE.get(lora_path)
    if cached is not None:
        return cached
    LOGGER.info("[Cutlery Remote CLIP] Loading CLIP LoRA name=%s path=%s", lora_name, lora_path)
    lora, lora_metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
    _LORA_CACHE[lora_path] = (lora, lora_metadata)
    return lora, lora_metadata


def _apply_clip_loras(clip, entries: list[dict[str, Any]]):
    import comfy.sd

    for entry in entries:
        lora_name = str(entry.get("lora_name") or "").strip()
        strength_clip = float(entry.get("strength_clip", 0.0))
        if not lora_name or strength_clip == 0:
            continue
        lora, lora_metadata = _load_lora_file(lora_name)
        LOGGER.info("[Cutlery Remote CLIP] Applying CLIP LoRA name=%s strength_clip=%s", lora_name, strength_clip)
        _model, clip = comfy.sd.load_lora_for_models(
            None,
            clip,
            lora,
            0.0,
            strength_clip,
            lora_metadata=lora_metadata,
        )
    return clip


def _normalize_remote_loras(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    entries: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        lora_name = str(item.get("lora_name") or item.get("name") or "").strip()
        if not lora_name:
            continue
        entries.append({"lora_name": lora_name, "strength_clip": float(item.get("strength_clip", 1.0))})
    return entries


def encode_remote_clip_text(body: dict[str, Any]) -> dict[str, Any]:
    global _ACTIVE_CLIP_KEY

    prompt = str(body.get("prompt") or "")
    text_encoder = str(body.get("text_encoder") or "").strip()
    if not text_encoder:
        raise ValueError("text_encoder is required.")
    clip_type = str(body.get("clip_type") or "stable_diffusion").strip()
    device = str(body.get("device") or "default").strip()
    loras = _normalize_remote_loras(body.get("loras"))

    active_key = _ensure_active_clip_key(text_encoder, clip_type, device)
    base_clip = _load_clip_for_remote_encode(text_encoder, clip_type, device)
    _ACTIVE_CLIP_KEY = active_key
    clip = base_clip.clone() if hasattr(base_clip, "clone") else base_clip
    clip = _apply_clip_loras(clip, loras)
    LOGGER.info(
        "[Cutlery Remote CLIP] Encoding prompt text_encoder=%s clip_type=%s lora_count=%s prompt_chars=%s",
        text_encoder,
        clip_type,
        len(loras),
        len(prompt),
    )
    tokens = clip.tokenize(prompt)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    return {"ok": True, "conditioning": encode_value_bundle(conditioning)}


def encode_remote_dual_clip_text(body: dict[str, Any]) -> dict[str, Any]:
    global _ACTIVE_CLIP_KEY

    prompt = str(body.get("prompt") or "")
    clip_name1 = str(body.get("clip_name1") or body.get("text_encoder1") or "").strip()
    clip_name2 = str(body.get("clip_name2") or body.get("text_encoder2") or "").strip()
    if not clip_name1:
        raise ValueError("clip_name1 is required.")
    if not clip_name2:
        raise ValueError("clip_name2 is required.")
    clip_type = str(body.get("clip_type") or body.get("type") or "stable_diffusion").strip()
    device = str(body.get("device") or "default").strip()
    loras = _normalize_remote_loras(body.get("loras"))

    active_key = _ensure_active_dual_clip_key(clip_name1, clip_name2, clip_type, device)
    base_clip = _load_dual_clip_for_remote_encode(clip_name1, clip_name2, clip_type, device)
    _ACTIVE_CLIP_KEY = active_key
    clip = base_clip.clone() if hasattr(base_clip, "clone") else base_clip
    clip = _apply_clip_loras(clip, loras)
    LOGGER.info(
        "[Cutlery Remote CLIP] Encoding prompt dual_clip1=%s dual_clip2=%s clip_type=%s lora_count=%s prompt_chars=%s",
        clip_name1,
        clip_name2,
        clip_type,
        len(loras),
        len(prompt),
    )
    tokens = clip.tokenize(prompt)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    return {"ok": True, "conditioning": encode_value_bundle(conditioning)}


QWEN_IMAGE_EDIT_PLUS_LLAMA_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the original input where appropriate."
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)


def _materialized_qwen_image_relative_name(source_name: str, sha256: str) -> str:
    basename = Path(str(source_name or "").replace("\\", "/")).name or "image.png"
    if not basename.lower().endswith(".png"):
        basename = f"{basename}.png"
    safe_basename = re.sub(r"[^A-Za-z0-9._-]+", "_", basename).strip("._") or "image.png"
    return f"{REMOTE_CLIP_MATERIALIZED_QWEN_IMAGE_DIR}/{sha256[:12]}-{safe_basename}"


async def _materialize_qwen_image_upload(
    source_name: str,
    stream: Any,
    expected_sha256: str = "",
    limit_bytes: int | None = None,
) -> dict[str, Any]:
    effective_limit = _remote_clip_file_upload_limit_bytes() if limit_bytes is None else int(limit_bytes)
    input_root = _primary_input_root().resolve()
    materialized_root = (input_root / REMOTE_CLIP_MATERIALIZED_QWEN_IMAGE_DIR).resolve()
    materialized_root.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    total_size = 0
    tmp_handle = tempfile.NamedTemporaryFile("wb", delete=False, dir=materialized_root, prefix=".upload-", suffix=".tmp")
    tmp_path = Path(tmp_handle.name)
    try:
        with tmp_handle as handle:
            async for chunk in stream.iter_chunked(REMOTE_CLIP_UPLOAD_CHUNK_SIZE):
                data = bytes(chunk or b"")
                if not data:
                    continue
                if total_size + len(data) > effective_limit:
                    raise RemoteClipUploadTooLarge(
                        f"Uploaded Qwen image exceeds the {effective_limit}-byte limit."
                    )
                handle.write(data)
                digest.update(data)
                total_size += len(data)

        sha256 = digest.hexdigest()
        expected = str(expected_sha256 or "").strip().lower()
        if expected and expected != sha256:
            raise ValueError("Uploaded Qwen image SHA-256 did not match the expected hash.")

        relative_name = _materialized_qwen_image_relative_name(source_name, sha256)
        target = (input_root / Path(relative_name)).resolve()
        if os.path.commonpath([str(materialized_root), str(target)]) != str(materialized_root):
            raise ValueError("Materialized Qwen image path escaped the remote materialized image directory.")

        if target.exists() and _sha256_file(target) == sha256:
            tmp_path.unlink(missing_ok=True)
            return {
                "ok": True,
                "name": relative_name,
                "subfolder": REMOTE_CLIP_MATERIALIZED_QWEN_IMAGE_DIR,
                "type": "input",
                "sha256": sha256,
                "size": target.stat().st_size,
                "materialized": True,
            }

        tmp_path.replace(target)
        _LORA_HASH_CACHE.pop(str(target.resolve()), None)
        LOGGER.info(
            "[Cutlery Remote CLIP] Materialized remote Qwen image name=%s bytes=%s sha256=%s",
            relative_name,
            _format_bytes(total_size),
            sha256,
        )
        return {
            "ok": True,
            "name": relative_name,
            "subfolder": REMOTE_CLIP_MATERIALIZED_QWEN_IMAGE_DIR,
            "type": "input",
            "sha256": sha256,
            "size": total_size,
            "materialized": True,
        }
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _materialize_qwen_png_bytes_to_remote(
    filename: str,
    payload: bytes,
    *,
    progress_node_id: str | None = None,
) -> dict[str, Any]:
    sha256 = _sha256_bytes(payload)
    tmp_handle = tempfile.NamedTemporaryFile("wb", delete=False, suffix=".png")
    tmp_path = Path(tmp_handle.name)
    try:
        with tmp_handle as handle:
            handle.write(payload)
        progress = _RemoteClipUploadProgress(progress_node_id, len(payload))
        response = _post_file(
            "/cutlery/remote/clip/images/materialize",
            tmp_path,
            {
                "X-Cutlery-Image-Name": urllib.parse.quote(filename, safe="/.-_"),
                "X-Cutlery-Image-SHA256": sha256,
            },
            source_name=filename,
            progress=progress,
            progress_label="Qwen image",
        )
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "Remote Qwen image materialization failed."))
        progress.finish()
        return response
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _image_tensor_to_png_bytes(frame: Any) -> bytes:
    from PIL import Image
    import torch

    array = frame.detach().cpu().clamp(0.0, 1.0)[..., :3].mul(255.0).round().to(dtype=torch.uint8).numpy()
    with io.BytesIO() as buffer:
        Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
        return buffer.getvalue()


def _encode_image_file_bundle(image: Any, *, name: str = "image") -> dict[str, Any]:
    import torch

    if not isinstance(image, torch.Tensor):
        raise TypeError("Remote Qwen image transport expects a torch IMAGE tensor.")
    tensor = image.detach()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or int(tensor.shape[-1]) < 3:
        raise ValueError("Remote Qwen image transport expects IMAGE tensors shaped [batch, height, width, channels].")

    frames = []
    for index, frame in enumerate(tensor, start=1):
        payload = _image_tensor_to_png_bytes(frame)
        frames.append(
            {
                "filename": f"{name}_{index:04d}.png",
                "mime_type": "image/png",
                "byte_count": len(payload),
                "sha256": _sha256_bytes(payload),
                "data": base64.b64encode(payload).decode("ascii"),
            }
        )
    return {
        "schema": REMOTE_CLIP_IMAGE_BUNDLE_SCHEMA,
        "format": "png",
        "mime_type": "image/png",
        "width": int(tensor.shape[2]),
        "height": int(tensor.shape[1]),
        "channels": 3,
        "frames": frames,
    }


def _encode_image_file_ref_bundle(
    image: Any,
    *,
    name: str = "image",
    progress_node_id: str | None = None,
) -> dict[str, Any]:
    import torch

    if not isinstance(image, torch.Tensor):
        raise TypeError("Remote Qwen image transport expects a torch IMAGE tensor.")
    tensor = image.detach()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4 or int(tensor.shape[-1]) < 3:
        raise ValueError("Remote Qwen image transport expects IMAGE tensors shaped [batch, height, width, channels].")

    frames = []
    for index, frame in enumerate(tensor, start=1):
        filename = f"{name}_{index:04d}.png"
        payload = _image_tensor_to_png_bytes(frame)
        materialized = _materialize_qwen_png_bytes_to_remote(filename, payload, progress_node_id=progress_node_id)
        frame_payload = {
            "filename": filename,
            "name": str(materialized.get("name") or ""),
            "subfolder": str(materialized.get("subfolder") or REMOTE_CLIP_MATERIALIZED_QWEN_IMAGE_DIR),
            "type": str(materialized.get("type") or "input"),
            "mime_type": "image/png",
            "byte_count": int(materialized.get("size") or len(payload)),
            "sha256": str(materialized.get("sha256") or _sha256_bytes(payload)).lower(),
        }
        frames.append(frame_payload)
    return {
        "schema": REMOTE_CLIP_IMAGE_FILE_REF_BUNDLE_SCHEMA,
        "format": "png",
        "mime_type": "image/png",
        "width": int(tensor.shape[2]),
        "height": int(tensor.shape[1]),
        "channels": 3,
        "frames": frames,
    }


def _decode_image_file_bundle(bundle: dict[str, Any]):
    import numpy as np
    from PIL import Image
    import torch

    frames_payload = bundle.get("frames")
    if not isinstance(frames_payload, list) or not frames_payload:
        raise ValueError("Remote Qwen image bundle must include at least one PNG frame.")

    frames = []
    for frame in frames_payload:
        if not isinstance(frame, dict):
            raise ValueError("Remote Qwen image frame must be an object.")
        payload = base64.b64decode(str(frame.get("data") or "").encode("ascii"), validate=True)
        expected_sha256 = str(frame.get("sha256") or "").strip().lower()
        if expected_sha256 and _sha256_bytes(payload) != expected_sha256:
            raise ValueError("Remote Qwen image frame SHA-256 did not match the expected hash.")
        with Image.open(io.BytesIO(payload)) as image:
            rgb = image.convert("RGB")
            frames.append(torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0))
    return torch.stack(frames, dim=0)


def _decode_image_file_ref_bundle(bundle: dict[str, Any]):
    import folder_paths
    import numpy as np
    from PIL import Image
    import torch

    frames_payload = bundle.get("frames")
    if not isinstance(frames_payload, list) or not frames_payload:
        raise ValueError("Remote Qwen image file-ref bundle must include at least one PNG frame.")

    frames = []
    for frame in frames_payload:
        if not isinstance(frame, dict):
            raise ValueError("Remote Qwen image file-ref frame must be an object.")
        if str(frame.get("type") or "input") != "input":
            raise ValueError("Remote Qwen image file-ref frames must reference ComfyUI input files.")
        name = str(frame.get("name") or frame.get("filename") or "").replace("\\", "/").strip().strip("/")
        if not name:
            raise ValueError("Remote Qwen image file-ref frame must include a filename.")
        path = Path(folder_paths.get_annotated_filepath(name))
        payload = path.read_bytes()
        expected_sha256 = str(frame.get("sha256") or "").strip().lower()
        if expected_sha256 and _sha256_bytes(payload) != expected_sha256:
            raise ValueError("Remote Qwen image file-ref SHA-256 did not match the expected hash.")
        with Image.open(io.BytesIO(payload)) as image:
            rgb = image.convert("RGB")
            frames.append(torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0))
    return torch.stack(frames, dim=0)


def _decode_optional_image_bundle(bundle: Any):
    if bundle is None:
        return None
    if not isinstance(bundle, dict):
        raise ValueError("Remote Qwen image input must be an image file bundle.")
    if bundle.get("schema") == REMOTE_CLIP_IMAGE_BUNDLE_SCHEMA:
        return _decode_image_file_bundle(bundle)
    if bundle.get("schema") == REMOTE_CLIP_IMAGE_FILE_REF_BUNDLE_SCHEMA:
        return _decode_image_file_ref_bundle(bundle)
    return decode_value_bundle(bundle, max_blob_bytes=_remote_clip_response_limit_bytes())


def _qwen_image_edit_plus_conditioning(clip, prompt: str, *, vae=None, image1=None, image2=None, image3=None):
    import comfy.utils
    import node_helpers

    ref_latents = []
    images_vl = []
    image_prompt = ""

    for index, image in enumerate([image1, image2, image3]):
        if image is None:
            continue
        samples = image.movedim(-1, 1)
        total = int(384 * 384)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)

        scaled = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
        images_vl.append(scaled.movedim(1, -1))
        if vae is not None:
            total = int(1024 * 1024)
            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by / 8.0) * 8
            height = round(samples.shape[2] * scale_by / 8.0) * 8

            scaled = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            ref_latents.append(vae.encode(scaled.movedim(1, -1)[:, :, :, :3]))

        image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(index + 1)

    tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=QWEN_IMAGE_EDIT_PLUS_LLAMA_TEMPLATE)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    if ref_latents:
        conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
    return conditioning


def encode_remote_qwen_image_edit_plus_text(body: dict[str, Any]) -> dict[str, Any]:
    global _ACTIVE_CLIP_KEY

    prompt = str(body.get("prompt") or "")
    text_encoder = str(body.get("text_encoder") or "").strip()
    if not text_encoder:
        raise ValueError("text_encoder is required.")
    clip_type = str(body.get("clip_type") or "qwen_image").strip() or "qwen_image"
    device = str(body.get("device") or "default").strip()
    vae_name = str(body.get("vae_name") or "").strip()
    images_payload = body.get("images") if isinstance(body.get("images"), dict) else {}

    active_key = _ensure_active_clip_key(text_encoder, clip_type, device)
    base_clip = _load_clip_for_remote_encode(text_encoder, clip_type, device)
    _ACTIVE_CLIP_KEY = active_key
    clip = base_clip.clone() if hasattr(base_clip, "clone") else base_clip
    vae = _load_vae_for_remote_encode(vae_name)
    images = {
        name: _decode_optional_image_bundle(images_payload.get(name) or body.get(name))
        for name in ("image1", "image2", "image3")
    }
    image_count = sum(1 for image in images.values() if image is not None)
    LOGGER.info(
        "[Cutlery Remote CLIP] Encoding Qwen image edit prompt text_encoder=%s clip_type=%s vae=%s image_count=%s prompt_chars=%s",
        text_encoder,
        clip_type,
        vae_name or NONE_CHOICE,
        image_count,
        len(prompt),
    )
    conditioning = _qwen_image_edit_plus_conditioning(clip, prompt, vae=vae, **images)
    return {"ok": True, "conditioning": encode_value_bundle(conditioning)}


def _remote_clip_job_payload(payload_json: str) -> dict[str, Any]:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as error:
        raise ValueError("Remote CLIP job payload must be valid JSON.") from error
    if not isinstance(payload, dict):
        raise ValueError("Remote CLIP job payload must be a JSON object.")
    return payload


def _run_remote_clip_job_encoder(payload_json: str, encoder) -> dict[str, Any]:
    result = encoder(_remote_clip_job_payload(payload_json))
    bundle = result.get("conditioning") if isinstance(result, dict) else None
    if not isinstance(bundle, dict):
        raise RuntimeError("Remote CLIP encoder did not return a conditioning bundle.")
    return {"ui": {REMOTE_CLIP_JOB_UI_KEY: [bundle]}}


class _CutleryRemoteClipJob:
    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "payload_json": ("STRING", {"multiline": True}),
            },
        }


class CutleryRemoteClipTextEncodeJob(_CutleryRemoteClipJob):
    DESCRIPTION = "Execute one Remote CLIP text-encoding request queued by a trusted peer."

    def execute(self, payload_json: str):
        return _run_remote_clip_job_encoder(payload_json, encode_remote_clip_text)


class CutleryRemoteDualClipTextEncodeJob(_CutleryRemoteClipJob):
    DESCRIPTION = "Execute one Remote dual-CLIP text-encoding request queued by a trusted peer."

    def execute(self, payload_json: str):
        return _run_remote_clip_job_encoder(payload_json, encode_remote_dual_clip_text)


class CutleryRemoteQwenImageEditPlusEncodeJob(_CutleryRemoteClipJob):
    DESCRIPTION = "Execute one Remote Qwen image-edit text-encoding request queued by a trusted peer."

    def execute(self, payload_json: str):
        return _run_remote_clip_job_encoder(payload_json, encode_remote_qwen_image_edit_plus_text)


_REMOTE_CLIP_JOB_CLASS_TYPES = {
    "single": "CutleryRemoteClipTextEncodeJob",
    "dual": "CutleryRemoteDualClipTextEncodeJob",
    "qwen_image_edit_plus": "CutleryRemoteQwenImageEditPlusEncodeJob",
}


def _remote_clip_job_error(status: Any) -> str:
    if not isinstance(status, dict):
        return "Remote CLIP job failed."
    messages = status.get("messages")
    if isinstance(messages, list) and messages:
        return str(messages[-1])
    return "Remote CLIP job failed."


def _cancel_remote_clip_job(prompt_queue: Any, prompt_id: str) -> None:
    def is_prompt(item: Any) -> bool:
        return isinstance(item, (list, tuple)) and len(item) > 1 and str(item[1]) == prompt_id

    prompt_queue.delete_queue_item(is_prompt)
    prompt_queue.interrupt_if_running(prompt_id)


async def _submit_remote_clip_job(kind: str, body: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if PromptServer is None:
        return {"ok": False, "error": "ComfyUI PromptServer is not available."}, 503
    class_type = _REMOTE_CLIP_JOB_CLASS_TYPES.get(kind)
    if class_type is None:
        raise ValueError(f"Unsupported Remote CLIP job kind: {kind}")

    try:
        import execution
    except Exception as error:
        return {"ok": False, "error": f"Could not import ComfyUI execution module: {error}"}, 500

    server = PromptServer.instance
    prompt_queue = getattr(server, "prompt_queue", None)
    if prompt_queue is None:
        return {"ok": False, "error": "ComfyUI prompt queue is not available."}, 503

    prompt_id = str(uuid.uuid4())
    prompt = {
        REMOTE_CLIP_JOB_NODE_ID: {
            "class_type": class_type,
            "inputs": {"payload_json": json.dumps(body, separators=(",", ":"))},
            "_meta": {"title": "Remote CLIP Text Encode"},
        },
    }
    if hasattr(server, "node_replace_manager"):
        server.node_replace_manager.apply_replacements(prompt)
    valid = await execution.validate_prompt(prompt_id, prompt, None)
    if not valid[0]:
        return {"ok": False, "error": valid[1], "node_errors": valid[3]}, 400

    extra_data = {
        "create_time": int(time.time() * 1000),
        "cutlery_remote_clip": {"kind": kind},
    }
    sensitive = {}
    for key in getattr(execution, "SENSITIVE_EXTRA_DATA_KEYS", []):
        if key in extra_data:
            sensitive[key] = extra_data.pop(key)
    number = float(getattr(server, "number", 0))
    server.number = number + 1
    prompt_queue.put((number, prompt_id, prompt, extra_data, valid[2], sensitive))

    deadline = asyncio.get_running_loop().time() + _remote_clip_encode_timeout()
    try:
        while True:
            history = prompt_queue.get_history(prompt_id=prompt_id)
            entry = history.get(prompt_id) if isinstance(history, dict) else None
            if isinstance(entry, dict):
                status = entry.get("status")
                if not isinstance(status, dict) or status.get("status_str") != "success":
                    return {"ok": False, "prompt_id": prompt_id, "error": _remote_clip_job_error(status)}, 500
                outputs = entry.get("outputs")
                node_output = outputs.get(REMOTE_CLIP_JOB_NODE_ID) if isinstance(outputs, dict) else None
                bundles = node_output.get(REMOTE_CLIP_JOB_UI_KEY) if isinstance(node_output, dict) else None
                bundle = bundles[-1] if isinstance(bundles, list) and bundles else None
                if not isinstance(bundle, dict):
                    return {
                        "ok": False,
                        "prompt_id": prompt_id,
                        "error": "Remote CLIP job completed without a conditioning bundle.",
                    }, 500
                return {"ok": True, "prompt_id": prompt_id, "conditioning": bundle}, 200

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                _cancel_remote_clip_job(prompt_queue, prompt_id)
                return {"ok": False, "prompt_id": prompt_id, "error": "Remote CLIP job timed out."}, 504
            await asyncio.sleep(min(0.1, remaining))
    except asyncio.CancelledError:
        _cancel_remote_clip_job(prompt_queue, prompt_id)
        raise


def register_remote_clip_routes() -> None:
    if PromptServer is None or web is None:
        return
    routes = PromptServer.instance.routes
    if getattr(routes, "_cutlery_remote_clip_routes_registered", False):
        return

    @routes.get("/cutlery/remote/clip/inventory")
    async def cutlery_remote_clip_inventory(request):
        disabled = _remote_clip_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        query = getattr(request, "query", {}) or {}
        raw_include_hashes = str(query.get("include_hashes", query.get("hashes", "1"))).strip().lower()
        include_hashes = raw_include_hashes not in {"0", "false", "no", "off"}
        payload = await asyncio.to_thread(local_remote_clip_inventory, include_hashes=include_hashes)
        return _json_response(payload)

    @routes.get("/cutlery/remote/clip/choices")
    async def cutlery_remote_clip_choices(request):
        try:
            payload = await asyncio.to_thread(remote_clip_widget_choices, WIDGET_INVENTORY_TIMEOUT_SECONDS)
            return _json_response(payload)
        except Exception as error:
            LOGGER.warning("[Cutlery Remote CLIP] Remote clip choice refresh failed", exc_info=True)
            return _json_response({"ok": False, "error": str(error), "text_encoders": [], "clip_types": list(CLIP_TYPES), "vaes": []}, status=502)

    @routes.post("/cutlery/remote/clip/text-encode")
    async def cutlery_remote_clip_text_encode(request):
        disabled = _remote_clip_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        body = await _request_json(request)
        try:
            payload, status = await _submit_remote_clip_job("single", body)
            return _json_response(payload, status=status)
        except ValueError as error:
            return _json_response({"ok": False, "error": str(error)}, status=400)
        except Exception as error:
            LOGGER.warning("[Cutlery Remote CLIP] Text encode failed", exc_info=True)
            return _json_response({"ok": False, "error": str(error)}, status=500)

    @routes.post("/cutlery/remote/clip/dual-text-encode")
    async def cutlery_remote_dual_clip_text_encode(request):
        disabled = _remote_clip_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        body = await _request_json(request)
        try:
            payload, status = await _submit_remote_clip_job("dual", body)
            return _json_response(payload, status=status)
        except ValueError as error:
            return _json_response({"ok": False, "error": str(error)}, status=400)
        except Exception as error:
            LOGGER.warning("[Cutlery Remote CLIP] Dual text encode failed", exc_info=True)
            return _json_response({"ok": False, "error": str(error)}, status=500)

    @routes.post("/cutlery/remote/clip/qwen-image-edit-plus")
    async def cutlery_remote_qwen_image_edit_plus_text_encode(request):
        disabled = _remote_clip_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        try:
            _raise_request_body_limit(request, _remote_clip_file_upload_limit_bytes())
            body = await _request_json(request)
            payload, status = await _submit_remote_clip_job("qwen_image_edit_plus", body)
            return _json_response(payload, status=status)
        except ValueError as error:
            return _json_response({"ok": False, "error": str(error)}, status=400)
        except Exception as error:
            if _is_request_entity_too_large(error):
                return _json_response(
                    {
                        "ok": False,
                        "error": (
                            f"Remote Qwen image payload is too large. Increase {REMOTE_CLIP_FILE_UPLOAD_LIMIT_MB_ENV} "
                            "on the remote ComfyUI instance if this workflow should send larger image inputs."
                        ),
                    },
                    status=413,
                )
            LOGGER.warning("[Cutlery Remote CLIP] Qwen image edit text encode failed", exc_info=True)
            return _json_response({"ok": False, "error": str(error)}, status=500)

    @routes.post("/cutlery/remote/clip/clips/materialize")
    async def cutlery_remote_clip_materialize(request):
        disabled = _remote_clip_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        headers = _request_headers(request)
        source_name = urllib.parse.unquote(_header_value(headers, "X-Cutlery-Clip-Name"))
        expected_sha256 = _header_value(headers, "X-Cutlery-Clip-SHA256")
        stream = getattr(request, "content", None)
        if stream is None or not hasattr(stream, "iter_chunked"):
            return _json_response({"ok": False, "error": "Request body stream is required."}, status=400)
        try:
            _raise_request_body_limit(request, _remote_clip_file_upload_limit_bytes())
            return _json_response(
                await _materialize_clip_upload(
                    source_name,
                    stream,
                    expected_sha256,
                    _remote_clip_file_upload_limit_bytes(),
                )
            )
        except RemoteClipUploadTooLarge as error:
            return _json_response({"ok": False, "error": str(error)}, status=413)
        except ValueError as error:
            return _json_response({"ok": False, "error": str(error)}, status=400)
        except Exception as error:
            if _is_request_entity_too_large(error):
                return _json_response(
                    {
                        "ok": False,
                        "error": (
                            f"Uploaded CLIP/text encoder is too large. Increase {REMOTE_CLIP_FILE_UPLOAD_LIMIT_MB_ENV} "
                            "on the remote ComfyUI instance if this file should be materialized."
                        ),
                    },
                    status=413,
                )
            LOGGER.warning("[Cutlery Remote CLIP] CLIP/text encoder materialization failed", exc_info=True)
            return _json_response({"ok": False, "error": str(error)}, status=500)

    @routes.post("/cutlery/remote/clip/loras/materialize")
    async def cutlery_remote_clip_lora_materialize(request):
        disabled = _remote_clip_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        headers = _request_headers(request)
        source_name = urllib.parse.unquote(_header_value(headers, "X-Cutlery-Lora-Name"))
        expected_sha256 = _header_value(headers, "X-Cutlery-Lora-SHA256")
        stream = getattr(request, "content", None)
        if stream is None or not hasattr(stream, "iter_chunked"):
            return _json_response({"ok": False, "error": "Request body stream is required."}, status=400)
        try:
            upload_limit = _remote_clip_lora_upload_limit_bytes()
            _raise_request_body_limit(request, upload_limit)
            return _json_response(
                await _materialize_lora_upload(
                    source_name,
                    stream,
                    expected_sha256,
                    upload_limit,
                )
            )
        except RemoteClipUploadTooLarge as error:
            return _json_response({"ok": False, "error": str(error)}, status=413)
        except ValueError as error:
            return _json_response({"ok": False, "error": str(error)}, status=400)
        except Exception as error:
            if _is_request_entity_too_large(error):
                return _json_response(
                    {
                        "ok": False,
                        "error": (
                            f"Uploaded LoRA is too large. Increase {REMOTE_CLIP_LORA_UPLOAD_LIMIT_MB_ENV} "
                            "on the remote ComfyUI instance if this file should be materialized."
                        ),
                    },
                    status=413,
                )
            LOGGER.warning("[Cutlery Remote CLIP] LoRA materialization failed", exc_info=True)
            return _json_response({"ok": False, "error": str(error)}, status=500)

    @routes.post("/cutlery/remote/clip/images/materialize")
    async def cutlery_remote_clip_qwen_image_materialize(request):
        disabled = _remote_clip_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        headers = _request_headers(request)
        source_name = urllib.parse.unquote(_header_value(headers, "X-Cutlery-Image-Name"))
        expected_sha256 = _header_value(headers, "X-Cutlery-Image-SHA256")
        stream = getattr(request, "content", None)
        if stream is None or not hasattr(stream, "iter_chunked"):
            return _json_response({"ok": False, "error": "Request body stream is required."}, status=400)
        try:
            _raise_request_body_limit(request, _remote_clip_file_upload_limit_bytes())
            return _json_response(
                await _materialize_qwen_image_upload(
                    source_name,
                    stream,
                    expected_sha256,
                    _remote_clip_file_upload_limit_bytes(),
                )
            )
        except RemoteClipUploadTooLarge as error:
            return _json_response({"ok": False, "error": str(error)}, status=413)
        except ValueError as error:
            return _json_response({"ok": False, "error": str(error)}, status=400)
        except Exception as error:
            if _is_request_entity_too_large(error):
                return _json_response(
                    {
                        "ok": False,
                        "error": (
                            f"Uploaded Qwen input image is too large. Increase {REMOTE_CLIP_FILE_UPLOAD_LIMIT_MB_ENV} "
                            "on the remote ComfyUI instance if this file should be materialized."
                        ),
                    },
                    status=413,
                )
            LOGGER.warning("[Cutlery Remote CLIP] Qwen image materialization failed", exc_info=True)
            return _json_response({"ok": False, "error": str(error)}, status=500)

    @routes.post("/cutlery/remote/clip/clips/clear")
    async def cutlery_remote_clip_clear(request):
        headers = _request_headers(request)
        if _header_value(headers, "Authorization"):
            disabled = _remote_clip_server_disabled_response()
            if disabled is not None:
                return disabled
            ok, payload, status = _authorized(request)
            if not ok:
                return _json_response(payload or {}, status=status)
            try:
                return _json_response(_clear_materialized_clips())
            except Exception as error:
                LOGGER.warning("[Cutlery Remote CLIP] Clear materialized CLIP/text encoders failed", exc_info=True)
                return _json_response({"ok": False, "error": str(error)}, status=500)
        try:
            return _json_response(clear_remote_materialized_clips())
        except Exception as error:
            LOGGER.warning("[Cutlery Remote CLIP] Clear remote materialized CLIP/text encoders failed", exc_info=True)
            return _json_response({"ok": False, "error": str(error)}, status=500)

    @routes.post("/cutlery/remote/clip/loras/clear")
    async def cutlery_remote_clip_lora_clear(request):
        headers = _request_headers(request)
        if _header_value(headers, "Authorization"):
            disabled = _remote_clip_server_disabled_response()
            if disabled is not None:
                return disabled
            ok, payload, status = _authorized(request)
            if not ok:
                return _json_response(payload or {}, status=status)
            try:
                return _json_response(_clear_materialized_loras())
            except Exception as error:
                LOGGER.warning("[Cutlery Remote CLIP] Clear materialized LoRAs failed", exc_info=True)
                return _json_response({"ok": False, "error": str(error)}, status=500)
        try:
            return _json_response(clear_remote_materialized_loras())
        except Exception as error:
            LOGGER.warning("[Cutlery Remote CLIP] Clear remote materialized LoRAs failed", exc_info=True)
            return _json_response({"ok": False, "error": str(error)}, status=500)

    @routes.post("/cutlery/remote/clip/images/clear")
    async def cutlery_remote_clip_qwen_images_clear(request):
        headers = _request_headers(request)
        if _header_value(headers, "Authorization"):
            disabled = _remote_clip_server_disabled_response()
            if disabled is not None:
                return disabled
            ok, payload, status = _authorized(request)
            if not ok:
                return _json_response(payload or {}, status=status)
            try:
                return _json_response(_clear_materialized_qwen_images())
            except Exception as error:
                LOGGER.warning("[Cutlery Remote CLIP] Clear materialized Qwen images failed", exc_info=True)
                return _json_response({"ok": False, "error": str(error)}, status=500)
        try:
            return _json_response(clear_remote_materialized_qwen_images())
        except Exception as error:
            LOGGER.warning("[Cutlery Remote CLIP] Clear remote materialized Qwen images failed", exc_info=True)
            return _json_response({"ok": False, "error": str(error)}, status=500)

    @routes.post("/cutlery/remote/clip/unload")
    async def cutlery_remote_clip_unload(request):
        disabled = _remote_clip_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        try:
            return _json_response(unload_remote_clip_cache())
        except Exception as error:
            LOGGER.warning("[Cutlery Remote CLIP] Unload failed", exc_info=True)
            return _json_response({"ok": False, "error": str(error)}, status=500)

    setattr(routes, "_cutlery_remote_clip_routes_registered", True)


def _text_encoder_widget_choices(inventory: dict[str, Any]) -> list[str]:
    choices = [str(item) for item in inventory.get("text_encoders", []) if str(item).strip()]
    return choices or [CONFIGURE_REMOTE_CHOICE]


def _vae_widget_choices(inventory: dict[str, Any]) -> list[str]:
    raw_choices = [str(item) for item in inventory.get("vaes", []) if str(item).strip()]
    if raw_choices and raw_choices[0] in {CONFIGURE_REMOTE_CHOICE, LOADING_REMOTE_CHOICES}:
        return raw_choices
    choices = [NONE_CHOICE]
    choices.extend(item for item in raw_choices if item != NONE_CHOICE)
    return choices


def _encode_optional_image_value(
    image: Any,
    *,
    name: str = "image",
    progress_node_id: str | None = None,
) -> dict[str, Any] | None:
    if image is None:
        return None
    return _encode_image_file_ref_bundle(image, name=name, progress_node_id=progress_node_id)


def _is_remote_clip_client_mode() -> bool:
    return _remote_clip_mode() != REMOTE_CLIP_MODE_REMOTE


def _remote_model_placeholder_combo(default_value: str, tooltip: str):
    return ([default_value], {"default": default_value, "tooltip": tooltip})


class CutleryRemoteClipTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        inventory = _inventory_for_widgets()
        text_encoders = _text_encoder_widget_choices(inventory)
        default_text_encoder = str(text_encoders[0])
        clip_types = inventory.get("clip_types") or list(CLIP_TYPES)
        required: dict[str, Any] = {
            "prompt": ("STRING", {"multiline": True, "tooltip": "Prompt text to encode on the remote ComfyUI instance."}),
            "text_encoder": (
                text_encoders,
                {
                    "default": default_text_encoder,
                    "defaultInput": True,
                    "tooltip": "Text encoder file listed by the remote ComfyUI instance. A connected STRING or Primitive value overrides this picker.",
                },
            ),
            "clip_type": (clip_types, {"default": "stable_diffusion", "tooltip": "ComfyUI CLIP loader type used on the remote instance."}),
        }
        return {
            "required": required,
            "optional": {
                "lora_chain": (CUTLERY_LORA_CHAIN, {"tooltip": "Deferred LoRA chain to apply to the remote CLIP before encoding."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_TOOLTIPS = ("Conditioning produced by the remote text encoder.",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Encode prompt text on another Cutlery-enabled ComfyUI instance and return native CONDITIONING."
    SEARCH_ALIASES = ["remote clip text encode", "remote conditioning", "remote text encoder"]

    @classmethod
    def VALIDATE_INPUTS(cls, text_encoder: str):
        if not str(text_encoder or "").strip():
            return "A remote CLIP text encoder must be selected."
        return True

    def encode(
        self,
        prompt: str,
        text_encoder: str,
        clip_type: str = "stable_diffusion",
        lora_chain: Any | None = None,
        unique_id: str | None = None,
    ):
        if text_encoder in {CONFIGURE_REMOTE_CHOICE, LOADING_REMOTE_CHOICES}:
            raise RuntimeError(f"{_remote_clip_target_hint()} Restart ComfyUI after changing .env.")
        (remote_text_encoder,) = _prepare_remote_clip_names([str(text_encoder or "")], progress_node_id=unique_id)
        payload = {
            "prompt": str(prompt or ""),
            "text_encoder": remote_text_encoder,
            "clip_type": str(clip_type or "stable_diffusion"),
            "loras": _prepare_remote_lora_entries(_normalize_lora_chain_entries(lora_chain), progress_node_id=unique_id),
        }
        cache_key = _remote_conditioning_cache_key("single", payload)
        cached_bundle = _get_cached_remote_conditioning(cache_key)
        if cached_bundle is not None:
            LOGGER.info(
                "[Cutlery Remote CLIP] Reusing cached remote encode text_encoder=%s clip_type=%s lora_count=%s prompt_chars=%s",
                payload["text_encoder"],
                payload["clip_type"],
                len(payload["loras"]),
                len(payload["prompt"]),
            )
            return (decode_value_bundle(cached_bundle, max_blob_bytes=_remote_clip_response_limit_bytes()),)
        LOGGER.info(
            "[Cutlery Remote CLIP] Dispatching remote encode text_encoder=%s clip_type=%s lora_count=%s prompt_chars=%s",
            payload["text_encoder"],
            payload["clip_type"],
            len(payload["loras"]),
            len(payload["prompt"]),
        )
        bundle = post_remote_clip_encode(payload)
        _remember_remote_conditioning(cache_key, bundle)
        return (decode_value_bundle(bundle, max_blob_bytes=_remote_clip_response_limit_bytes()),)


class CutleryRemoteDualClipTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        inventory = _inventory_for_widgets()
        text_encoders = _text_encoder_widget_choices(inventory)
        default_text_encoder = str(text_encoders[0])
        clip_types = inventory.get("clip_types") or list(CLIP_TYPES)
        clip_name1_widget = (
            _remote_model_placeholder_combo(default_text_encoder, "First CLIP/text encoder file for the remote dual CLIP loader.")
            if _is_remote_clip_client_mode()
            else (text_encoders, {"default": default_text_encoder, "defaultInput": True, "tooltip": "First CLIP/text encoder file for the remote dual CLIP loader. A connected STRING or Primitive value overrides this picker."})
        )
        clip_name2_widget = (
            _remote_model_placeholder_combo(default_text_encoder, "Second CLIP/text encoder or projection file for the remote dual CLIP loader.")
            if _is_remote_clip_client_mode()
            else (text_encoders, {"default": default_text_encoder, "defaultInput": True, "tooltip": "Second CLIP/text encoder or projection file for the remote dual CLIP loader. A connected STRING or Primitive value overrides this picker."})
        )
        required: dict[str, Any] = {
            "prompt": ("STRING", {"multiline": True, "tooltip": "Prompt text to encode on the remote ComfyUI instance."}),
            "clip_name1": clip_name1_widget,
            "clip_name2": clip_name2_widget,
            "clip_type": (clip_types, {"default": "ltxv", "tooltip": "ComfyUI dual CLIP loader type used on the remote instance."}),
        }
        return {
            "required": required,
            "optional": {
                "lora_chain": (CUTLERY_LORA_CHAIN, {"tooltip": "Deferred LoRA chain to apply to the remote CLIP before encoding."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_TOOLTIPS = ("Conditioning produced by the remote dual text encoder.",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Encode prompt text on another Cutlery-enabled ComfyUI instance using two CLIP/text encoder files."
    SEARCH_ALIASES = ["remote dual clip text encode", "remote dual conditioning", "remote ltxv text encoder"]

    def encode(
        self,
        prompt: str,
        clip_name1: str,
        clip_name2: str,
        clip_type: str = "ltxv",
        lora_chain: Any | None = None,
        unique_id: str | None = None,
    ):
        if clip_name1 in {CONFIGURE_REMOTE_CHOICE, LOADING_REMOTE_CHOICES} or clip_name2 in {CONFIGURE_REMOTE_CHOICE, LOADING_REMOTE_CHOICES}:
            raise RuntimeError(f"{_remote_clip_target_hint()} Restart ComfyUI after changing .env.")
        remote_clip_name1, remote_clip_name2 = _prepare_remote_clip_names(
            [str(clip_name1 or ""), str(clip_name2 or "")],
            progress_node_id=unique_id,
        )
        payload = {
            "prompt": str(prompt or ""),
            "clip_name1": remote_clip_name1,
            "clip_name2": remote_clip_name2,
            "clip_type": str(clip_type or "ltxv"),
            "loras": _prepare_remote_lora_entries(_normalize_lora_chain_entries(lora_chain), progress_node_id=unique_id),
        }
        cache_key = _remote_conditioning_cache_key("dual", payload)
        cached_bundle = _get_cached_remote_conditioning(cache_key)
        if cached_bundle is not None:
            LOGGER.info(
                "[Cutlery Remote CLIP] Reusing cached remote dual encode clip1=%s clip2=%s clip_type=%s lora_count=%s prompt_chars=%s",
                payload["clip_name1"],
                payload["clip_name2"],
                payload["clip_type"],
                len(payload["loras"]),
                len(payload["prompt"]),
            )
            return (decode_value_bundle(cached_bundle, max_blob_bytes=_remote_clip_response_limit_bytes()),)
        LOGGER.info(
            "[Cutlery Remote CLIP] Dispatching remote dual encode clip1=%s clip2=%s clip_type=%s lora_count=%s prompt_chars=%s",
            payload["clip_name1"],
            payload["clip_name2"],
            payload["clip_type"],
            len(payload["loras"]),
            len(payload["prompt"]),
        )
        bundle = post_remote_dual_clip_encode(payload)
        _remember_remote_conditioning(cache_key, bundle)
        return (decode_value_bundle(bundle, max_blob_bytes=_remote_clip_response_limit_bytes()),)


class CutleryRemoteTextEncodeQwenImageEditPlus:
    @classmethod
    def INPUT_TYPES(cls):
        inventory = _inventory_for_widgets()
        text_encoders = _text_encoder_widget_choices(inventory)
        vaes = _vae_widget_choices(inventory)
        text_encoder_widget = (
            _remote_model_placeholder_combo(text_encoders[0], "Qwen image text encoder file listed by the remote ComfyUI instance.")
            if _is_remote_clip_client_mode()
            else (text_encoders, {"default": text_encoders[0], "defaultInput": True, "tooltip": "Qwen image text encoder file listed by the remote ComfyUI instance. A connected STRING or Primitive value overrides this picker."})
        )
        vae_widget = (
            _remote_model_placeholder_combo(vaes[0], "VAE file listed by the remote ComfyUI instance for Qwen reference latents.")
            if _is_remote_clip_client_mode()
            else (vaes, {"default": vaes[0], "tooltip": "VAE file listed by the remote ComfyUI instance for Qwen reference latents."})
        )
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "tooltip": "Prompt text to encode with the remote Qwen image-edit text encoder."}),
                "text_encoder": text_encoder_widget,
                "vae_name": vae_widget,
            },
            "optional": {
                "image1": ("IMAGE", {"tooltip": "First optional Qwen reference image sent to the remote ComfyUI instance."}),
                "image2": ("IMAGE", {"tooltip": "Second optional Qwen reference image sent to the remote ComfyUI instance."}),
                "image3": ("IMAGE", {"tooltip": "Third optional Qwen reference image sent to the remote ComfyUI instance."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_TOOLTIPS = ("Conditioning produced by the remote Qwen image edit text encoder.",)
    FUNCTION = "encode"
    CATEGORY = CATEGORY
    DESCRIPTION = "Run ComfyUI's TextEncodeQwenImageEditPlus on another Cutlery-enabled ComfyUI instance."
    SEARCH_ALIASES = ["remote textencodeqwenimageeditplus", "remote qwen image edit text encode", "remote qwen image edit plus"]

    def encode(
        self,
        prompt: str,
        text_encoder: str,
        vae_name: str = NONE_CHOICE,
        image1: Any | None = None,
        image2: Any | None = None,
        image3: Any | None = None,
        unique_id: str | None = None,
    ):
        if text_encoder in {CONFIGURE_REMOTE_CHOICE, LOADING_REMOTE_CHOICES}:
            raise RuntimeError(f"{_remote_clip_target_hint()} Restart ComfyUI after changing .env.")
        if vae_name in {CONFIGURE_REMOTE_CHOICE, LOADING_REMOTE_CHOICES}:
            raise RuntimeError(f"{_remote_clip_target_hint()} Restart ComfyUI after changing .env.")
        remote_text_encoder = _normalize_clip_name(text_encoder)
        if not remote_text_encoder:
            raise RuntimeError("Remote Qwen image edit text encoder is required.")
        images = {
            name: bundle
            for name, bundle in {
                "image1": _encode_optional_image_value(image1, name="image1", progress_node_id=unique_id),
                "image2": _encode_optional_image_value(image2, name="image2", progress_node_id=unique_id),
                "image3": _encode_optional_image_value(image3, name="image3", progress_node_id=unique_id),
            }.items()
            if bundle is not None
        }
        payload = {
            "prompt": str(prompt or ""),
            "text_encoder": remote_text_encoder,
            "clip_type": "qwen_image",
            "vae_name": "" if vae_name == NONE_CHOICE else str(vae_name or ""),
            "images": images,
        }
        cache_key = _remote_conditioning_cache_key("qwen_image_edit_plus", payload)
        cached_bundle = _get_cached_remote_conditioning(cache_key)
        if cached_bundle is not None:
            LOGGER.info(
                "[Cutlery Remote CLIP] Reusing cached remote Qwen image edit encode text_encoder=%s vae=%s image_count=%s prompt_chars=%s",
                payload["text_encoder"],
                payload["vae_name"] or NONE_CHOICE,
                len(images),
                len(payload["prompt"]),
            )
            return (decode_value_bundle(cached_bundle, max_blob_bytes=_remote_clip_response_limit_bytes()),)
        LOGGER.info(
            "[Cutlery Remote CLIP] Dispatching remote Qwen image edit encode text_encoder=%s vae=%s image_count=%s prompt_chars=%s",
            payload["text_encoder"],
            payload["vae_name"] or NONE_CHOICE,
            len(images),
            len(payload["prompt"]),
        )
        bundle = post_remote_qwen_image_edit_plus_encode(payload)
        _remember_remote_conditioning(cache_key, bundle)
        return (decode_value_bundle(bundle, max_blob_bytes=_remote_clip_response_limit_bytes()),)


register_remote_clip_routes()


NODE_CLASS_MAPPINGS = {
    "CutleryRemoteClipTextEncode": CutleryRemoteClipTextEncode,
    "CutleryRemoteDualClipTextEncode": CutleryRemoteDualClipTextEncode,
    "CutleryRemoteTextEncodeQwenImageEditPlus": CutleryRemoteTextEncodeQwenImageEditPlus,
    "CutleryRemoteClipTextEncodeJob": CutleryRemoteClipTextEncodeJob,
    "CutleryRemoteDualClipTextEncodeJob": CutleryRemoteDualClipTextEncodeJob,
    "CutleryRemoteQwenImageEditPlusEncodeJob": CutleryRemoteQwenImageEditPlusEncodeJob,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CutleryRemoteClipTextEncode": "Remote CLIP Text Encode",
    "CutleryRemoteDualClipTextEncode": "Remote Dual CLIP Text Encode",
    "CutleryRemoteTextEncodeQwenImageEditPlus": "Remote TextEncodeQwenImageEditPlus",
    "CutleryRemoteClipTextEncodeJob": "Remote CLIP Text Encode",
    "CutleryRemoteDualClipTextEncodeJob": "Remote Dual CLIP Text Encode",
    "CutleryRemoteQwenImageEditPlusEncodeJob": "Remote Qwen Image Edit Text Encode",
}


def _register_remote_clip_external_cache() -> None:
    try:
        from .cutlery_vram import register_external_model_cache
    except ImportError:
        try:
            from cutlery_vram import register_external_model_cache
        except ImportError:
            return

    def _status() -> dict[str, Any]:
        return {
            "registered": True,
            "active_clip_key": list(_ACTIVE_CLIP_KEY) if _ACTIVE_CLIP_KEY is not None else None,
            "clip_cache_count": len(_CLIP_CACHE),
            "lora_cache_count": len(_LORA_CACHE),
            "vae_cache_count": len(_VAE_CACHE),
            "conditioning_cache_count": len(_REMOTE_CLIP_CONDITIONING_CACHE),
        }

    register_external_model_cache("remote_clip_text_encoder", unload_remote_clip_cache, _status)


_register_remote_clip_external_cache()
