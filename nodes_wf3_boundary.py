from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import mimetypes
import os
import re
import shutil
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

try:
    from .cutlery_interrupt import throw_if_interrupted
except ImportError:  # pragma: no cover - supports direct module imports in tests.
    from cutlery_interrupt import throw_if_interrupted

try:
    from .cutlery_config import WORKFLOW_RUN_ENV, get_feature_config
except ImportError:  # pragma: no cover - supports direct module imports in tests.
    from cutlery_config import WORKFLOW_RUN_ENV, get_feature_config

try:
    from .cutlery_remote.boundary_types import (
        BOUNDARY_SOCKET_TYPES,
        SUPPORTED_BOUNDARY_PORT_TYPES,
        normalize_boundary_port_type,
    )
except ImportError:  # pragma: no cover - supports direct module imports in tests.
    from cutlery_remote.boundary_types import (
        BOUNDARY_SOCKET_TYPES,
        SUPPORTED_BOUNDARY_PORT_TYPES,
        normalize_boundary_port_type,
    )

try:
    from aiohttp import web
    from server import PromptServer
except Exception:
    web = None
    PromptServer = None

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    import nodes as _COMFY_NODES
except Exception:
    _COMFY_NODES = None


CATEGORY = "Cutlery"
LOGGER = logging.getLogger("cutlery.workflow.boundary")
MAX_WF3_PORTS = 64
VALUE_NAMES = tuple(f"value_{index + 1}" for index in range(MAX_WF3_PORTS))
SUPPORTED_PORT_TYPES = SUPPORTED_BOUNDARY_PORT_TYPES
MAX_VIDEO_INPUT_BYTES = 1000 * 1024 * 1024
VIDEO_DOWNLOAD_TIMEOUT_SECONDS = 60
MATERIALIZED_MARKER_NAME = ".cutlery-materialized"
LEGACY_MATERIALIZED_MARKER_NAME = ".cutlery-wf3-materialized"
PROMPT_CANCELLATION_TTL_SECONDS = 2 * 60 * 60
MAX_PROMPT_CANCELLATION_TOMBSTONES = 4096
WORKFLOW_RUN_ENABLED_ENV = WORKFLOW_RUN_ENV
LEGACY_WORKFLOW_RUN_ENABLED_ENV = "CUTLERY_WF3_RUN_ENABLED"
PORT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
JSON_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)
_PROMPT_CANCELLATION_LOCK = threading.Lock()
_PROMPT_CANCELLATION_TOMBSTONES: dict[str, float] = {}
PORT_SOCKET_TYPES = BOUNDARY_SOCKET_TYPES


def _workflow_run_enabled() -> bool:
    return get_feature_config().workflow_run


def _json_response(payload: dict[str, Any], status: int = 200):
    if web is None:
        return payload
    return web.json_response(payload, status=status)


def _clean_prompt_id(prompt_id: object) -> str:
    clean_prompt_id = str(prompt_id or "").strip()
    if not clean_prompt_id:
        raise ValueError("prompt_id is required.")
    return clean_prompt_id


def _positive_finite_seconds(
    value: object,
    *,
    field_name: str,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive finite number.")
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive finite number.") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{field_name} must be a positive finite number.")
    if maximum is not None and seconds > maximum:
        raise ValueError(f"{field_name} must be less than or equal to {maximum:g}.")
    return seconds


def _prune_prompt_cancellation_tombstones(now: float) -> None:
    cutoff = now - PROMPT_CANCELLATION_TTL_SECONDS
    expired = [
        prompt_id
        for prompt_id, recorded_at in _PROMPT_CANCELLATION_TOMBSTONES.items()
        if recorded_at < cutoff
    ]
    for prompt_id in expired:
        _PROMPT_CANCELLATION_TOMBSTONES.pop(prompt_id, None)


def record_prompt_cancellation(prompt_id: object) -> str:
    """Record cancellation before queue state exists, bounded by count and age."""

    clean_prompt_id = _clean_prompt_id(prompt_id)
    now = time.monotonic()
    with _PROMPT_CANCELLATION_LOCK:
        _prune_prompt_cancellation_tombstones(now)
        if (
            clean_prompt_id not in _PROMPT_CANCELLATION_TOMBSTONES
            and len(_PROMPT_CANCELLATION_TOMBSTONES) >= MAX_PROMPT_CANCELLATION_TOMBSTONES
        ):
            oldest_prompt_id = min(
                _PROMPT_CANCELLATION_TOMBSTONES,
                key=_PROMPT_CANCELLATION_TOMBSTONES.get,
            )
            _PROMPT_CANCELLATION_TOMBSTONES.pop(oldest_prompt_id, None)
        _PROMPT_CANCELLATION_TOMBSTONES[clean_prompt_id] = now
    return clean_prompt_id


def consume_prompt_cancellation(prompt_id: object) -> bool:
    """Consume one pre-queue cancellation tombstone if it is still live."""

    clean_prompt_id = _clean_prompt_id(prompt_id)
    now = time.monotonic()
    with _PROMPT_CANCELLATION_LOCK:
        _prune_prompt_cancellation_tombstones(now)
        return _PROMPT_CANCELLATION_TOMBSTONES.pop(clean_prompt_id, None) is not None


def prompt_cancellation_recorded(prompt_id: object) -> bool:
    """Return whether a live pre-queue cancellation tombstone exists."""

    clean_prompt_id = _clean_prompt_id(prompt_id)
    now = time.monotonic()
    with _PROMPT_CANCELLATION_LOCK:
        _prune_prompt_cancellation_tombstones(now)
        return clean_prompt_id in _PROMPT_CANCELLATION_TOMBSTONES


async def _request_json(request: Any) -> dict[str, Any]:
    json_fn = getattr(request, "json", None)
    if not callable(json_fn):
        return {}
    payload = json_fn()
    if inspect.isawaitable(payload):
        payload = await payload
    return payload if isinstance(payload, dict) else {}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _strict_json_output(
    value: Any,
    *,
    path: str,
    _active_containers: set[int] | None = None,
) -> Any:
    active_containers = _active_containers if _active_containers is not None else set()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains non-finite float {value!r}.")
        return value
    if isinstance(value, list):
        container_id = id(value)
        if container_id in active_containers:
            raise ValueError(f"{path} contains a circular list reference.")
        active_containers.add(container_id)
        try:
            return [
                _strict_json_output(
                    item,
                    path=f"{path}[{index}]",
                    _active_containers=active_containers,
                )
                for index, item in enumerate(value)
            ]
        finally:
            active_containers.remove(container_id)
    if isinstance(value, dict):
        container_id = id(value)
        if container_id in active_containers:
            raise ValueError(f"{path} contains a circular dictionary reference.")
        active_containers.add(container_id)
        normalized: dict[str, Any] = {}
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"{path} contains dictionary key {key!r} of type "
                        f"{type(key).__name__}; JSON object keys must be strings."
                    )
                normalized[key] = _strict_json_output(
                    item,
                    path=f"{path}[{key!r}]",
                    _active_containers=active_containers,
                )
            return normalized
        finally:
            active_containers.remove(container_id)
    raise TypeError(
        f"{path} contains unsupported value type {type(value).__name__}; "
        "JSON outputs only support null, strings, booleans, finite numbers, "
        "lists, and string-key dictionaries."
    )


def _normalize_json_schema(value: Any) -> dict[str, Any]:
    if _blank(value):
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("JSON port schema must be a JSON object.")
    wrapper_schema = value.get("schema")
    if isinstance(wrapper_schema, dict) and (
        "name" in value or "strict" in value or ("type" not in value and "properties" not in value)
    ):
        return wrapper_schema
    return value


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _port_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value) if value.strip() else []
    if isinstance(value, dict):
        if isinstance(value.get("ports"), list):
            value = value["ports"]
        else:
            value = [{"name": key, "type": item} for key, item in value.items()]
    if not isinstance(value, list):
        raise ValueError("ports_json must be a JSON array, an object with a ports array, or a name/type object.")
    return value


def parse_port_specs(value: Any) -> list[dict[str, Any]]:
    records = _port_records(value)
    if len(records) > MAX_WF3_PORTS:
        raise ValueError(
            f"ports_json declares {len(records)} ports; the maximum is {MAX_WF3_PORTS}."
        )
    ports: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Port {index} must be an object.")
        name = str(record.get("name") or "").strip()
        if not PORT_NAME_RE.match(name):
            raise ValueError(f"Port {index} has invalid name {name!r}.")
        if name in seen:
            raise ValueError(f"Port name {name!r} is duplicated.")
        kind = normalize_boundary_port_type(record.get("type") or "string")
        if kind not in SUPPORTED_PORT_TYPES:
            raise ValueError(f"Port {name!r} has unsupported type {kind!r}.")
        spec = {
            "name": name,
            "type": kind,
            "socket_type": PORT_SOCKET_TYPES[kind],
            "required": bool(record.get("required", False)),
        }
        if kind == "json" and "schema" in record:
            spec["schema"] = _normalize_json_schema(record["schema"])
        if "default" in record:
            spec["default"] = record["default"]
        ports.append(spec)
        seen.add(name)
    return ports


def _coerce_value(value: Any, kind: str) -> Any:
    if _blank(value):
        return None
    if kind == "string":
        return str(value)
    if kind == "int":
        return int(float(value))
    if kind == "float":
        return float(value)
    if kind == "bool":
        return _coerce_bool(value)
    if kind == "json":
        return _coerce_json(value)
    if kind == "image":
        return _load_image_value(value)
    if kind in {"mask", "latent", "conditioning"}:
        return value
    if kind == "video":
        return _load_video_value(value)
    return value


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "t", "yes", "y", "on", "1"}:
            return True
        if text in {"false", "f", "no", "n", "off", "0"}:
            return False
        raise ValueError(f"Cannot convert string {value!r} to bool.")
    return bool(value)


def _coerce_json(value: Any) -> Any:
    if isinstance(value, str):
        text = _json_text_payload(value)
        try:
            return _json_safe(json.loads(text))
        except json.JSONDecodeError:
            return _json_safe(value)
    return _json_safe(value)


def _json_text_payload(value: str) -> str:
    text = value.strip()
    match = JSON_CODE_FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
    return text


def _stringify_json_input(value: Any) -> Any:
    if _blank(value):
        return value
    if isinstance(value, str):
        text = _json_text_payload(value)
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            return json.dumps(_json_safe(value), separators=(",", ":"))
    return json.dumps(_json_safe(value), separators=(",", ":"))


def _coerce_output_value(value: Any, kind: str, key: str) -> Any:
    if kind == "image":
        return _save_image_output(value, key)
    if kind == "audio":
        return _save_audio_output(value, key)
    if kind == "video":
        return _save_video_output(value, key)
    if kind in {"mask", "latent", "conditioning"}:
        return value
    if kind == "string":
        return None if value is None else str(value)
    if kind == "int" and value is not None:
        return int(float(value))
    if kind == "float" and value is not None:
        return float(value)
    if kind == "bool" and value is not None:
        return _coerce_bool(value)
    return _strict_json_output(value, path=f"Workflow output {key!r}")


def _load_image_value(value: Any) -> Any:
    if _blank(value):
        return None
    if not isinstance(value, str):
        return value
    comfy_nodes = _COMFY_NODES
    if comfy_nodes is None:
        try:
            import nodes as comfy_nodes
        except Exception as exc:
            raise RuntimeError("Cutlery Workflow Input could not import ComfyUI LoadImage.") from exc
    image, _mask = comfy_nodes.LoadImage().load_image(value.strip())
    return image


def _load_video_value(value: Any) -> Any:
    if _blank(value):
        return None
    if not isinstance(value, str):
        return value
    try:
        from comfy_api.latest import InputImpl
    except Exception as exc:
        raise RuntimeError("Cutlery Workflow Input needs ComfyUI VideoFromFile for VIDEO inputs.") from exc
    return InputImpl.VideoFromFile(value.strip())


def _save_image_output(value: Any, key: str) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return {"path": value}
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Cutlery Workflow Output needs PIL and numpy to save IMAGE outputs.") from exc
    if folder_paths is None:
        raise RuntimeError("Cutlery Workflow Output could not resolve the ComfyUI output directory.")

    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, _prefix = folder_paths.get_save_image_path(
        f"cutlery/{key}",
        output_dir,
        value[0].shape[1] if hasattr(value, "shape") else None,
        value[0].shape[0] if hasattr(value, "shape") else None,
    )
    Path(full_output_folder).mkdir(parents=True, exist_ok=True)
    results = []
    for batch_number, image in enumerate(value):
        array = 255.0 * image.detach().cpu().numpy() if hasattr(image, "detach") else 255.0 * image
        img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
        file_name = f"{filename_with_batch_num}_{counter:05}_.png"
        path = Path(full_output_folder) / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        results.append(
            {
                "filename": file_name,
                "subfolder": subfolder,
                "type": "output",
                "path": str(path),
            }
        )
        counter += 1
    return results[0] if len(results) == 1 else results


def _media_content_type(filename: str, kind: str) -> str:
    suffix = Path(filename).suffix.lower()
    if kind == "audio":
        return {
            ".flac": "audio/flac",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
            ".opus": "audio/ogg",
            ".m4a": "audio/mp4",
            ".wav": "audio/wav",
        }.get(suffix, "application/octet-stream")
    return {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
    }.get(suffix, "application/octet-stream")


def _folder_type_directory(folder_type: str) -> str:
    if folder_paths is None:
        raise RuntimeError("Cutlery Workflow Output could not resolve ComfyUI directories.")
    normalized = str(folder_type or "output").lower()
    if normalized == "input":
        return folder_paths.get_input_directory()
    if normalized == "temp" and hasattr(folder_paths, "get_temp_directory"):
        return folder_paths.get_temp_directory()
    return folder_paths.get_output_directory()


def _output_relative_reference(path: Path, kind: str, folder_type: str = "output") -> dict[str, Any]:
    if folder_paths is not None:
        try:
            output_root = Path(_folder_type_directory(folder_type)).resolve()
            resolved = path.resolve()
            relative = resolved.relative_to(output_root)
            return {
                "filename": relative.name,
                "subfolder": relative.parent.as_posix() if str(relative.parent) != "." else "",
                "type": folder_type or "output",
                "path": str(resolved),
                "contentType": _media_content_type(relative.name, kind),
            }
        except Exception:
            pass
    return {
        "filename": path.name,
        "subfolder": "",
        "type": folder_type or "output",
        "path": str(path),
        "contentType": _media_content_type(path.name, kind),
    }


def _saved_result_to_reference(value: dict[str, Any], kind: str) -> dict[str, Any]:
    filename = str(value.get("filename") or "").strip()
    subfolder = str(value.get("subfolder") or "").strip()
    folder_type = str(value.get("type") or "output").strip() or "output"
    raw_path = value.get("path")
    if isinstance(raw_path, str) and raw_path.strip():
        path = Path(raw_path)
    else:
        if not filename:
            raise RuntimeError(f"Cutlery Workflow Output {kind.upper()} record did not include filename or path.")
        path = Path(_folder_type_directory(folder_type))
        if subfolder:
            path = path / subfolder
        path = path / filename
    record = _output_relative_reference(path, kind, folder_type)
    record["filename"] = filename or record["filename"]
    record["subfolder"] = subfolder
    record["type"] = folder_type
    if isinstance(value.get("contentType"), str):
        record["contentType"] = value["contentType"]
    elif isinstance(value.get("mimeType"), str):
        record["contentType"] = value["mimeType"]
    return record


def _normalize_existing_media_reference(value: Any, kind: str) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        records = [_normalize_existing_media_reference(item, kind) for item in value]
        records = [item for item in records if item is not None]
        return records[0] if len(records) == 1 else records
    if isinstance(value, str):
        return _output_relative_reference(Path(value), kind)
    if isinstance(value, dict):
        if value.get("filename") is None and value.get("path") is None:
            return None
        return _saved_result_to_reference(value, kind)
    return None


def _save_audio_output(value: Any, key: str) -> Any:
    existing = _normalize_existing_media_reference(value, "audio")
    if existing is not None:
        return existing
    try:
        from comfy_api.latest import io, ui
    except Exception as exc:
        raise RuntimeError("Cutlery Workflow Output needs comfy_api.latest.ui.AudioSaveHelper to save AUDIO outputs.") from exc
    try:
        saved = ui.AudioSaveHelper.save_audio(
            value,
            filename_prefix=f"cutlery/{key}",
            folder_type=io.FolderType.output,
            cls=None,
            format="flac",
            quality="128k",
        )
    except Exception as exc:
        raise RuntimeError(f"Cutlery Workflow Output could not save AUDIO output {key!r}: {exc}") from exc
    return _normalize_existing_media_reference(saved, "audio")


def _save_video_output(value: Any, key: str) -> Any:
    existing = _normalize_existing_media_reference(value, "video")
    if existing is not None:
        return existing
    if folder_paths is None:
        raise RuntimeError("Cutlery Workflow Output could not resolve the ComfyUI output directory.")
    try:
        from comfy_api.latest import Types
        video_format = Types.VideoContainer.MP4
        video_codec = Types.VideoCodec.H264
    except Exception:
        video_format = "mp4"
        video_codec = "h264"

    width, height = None, None
    if hasattr(value, "get_dimensions"):
        try:
            width, height = value.get_dimensions()
        except Exception:
            width, height = None, None
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, _prefix = folder_paths.get_save_image_path(
        f"cutlery/{key}",
        output_dir,
        width,
        height,
    )
    filename_path = Path(str(filename))
    file_name = f"{filename_path.name}_{counter:05}_.mp4"
    target = Path(full_output_folder) / filename_path.parent / file_name
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        value.save_to(str(target), format=video_format, codec=video_codec)
    except Exception as exc:
        raise RuntimeError(f"Cutlery Workflow Output could not save VIDEO output {key!r}: {exc}") from exc
    return _output_relative_reference(target, "video", "output")


def _wf3_inputs(wf3: Any) -> dict[str, Any]:
    if not isinstance(wf3, dict):
        return {}
    inputs = wf3.get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def _safe_filename_part(value: Any, default: str) -> str:
    text = SAFE_FILENAME_RE.sub("_", str(value or "").strip()).strip("._")
    return text or default


def _request_input_dir(input_dir: Path, request_id: str) -> Path:
    return input_dir / "cutlery" / _safe_filename_part(request_id, "request")


def _mark_materialized_request_dir(request_dir: Path) -> None:
    request_dir.mkdir(parents=True, exist_ok=True)
    (request_dir / MATERIALIZED_MARKER_NAME).write_text("1", encoding="utf-8")


def _cleanup_materialized_request_dir(input_dir: Path, request_id: str) -> None:
    safe_request_id = _safe_filename_part(request_id, "request")
    layouts = (("cutlery", MATERIALIZED_MARKER_NAME), ("cutlery_wf3", LEGACY_MATERIALIZED_MARKER_NAME))
    for subfolder, marker_name in layouts:
        cleanup_root = (input_dir / subfolder).resolve()
        request_dir = (cleanup_root / safe_request_id).resolve()
        if not (request_dir / marker_name).is_file():
            continue
        try:
            shutil.rmtree(request_dir)
        except FileNotFoundError:
            continue
        except OSError:
            LOGGER.warning("Could not remove materialized Cutlery request directory %s", request_dir, exc_info=True)


class _PreQueueMaterializationGuard:
    """Own marked request inputs until a successful prompt-queue insertion."""

    def __init__(self, input_dir: Path, request_id: str):
        self.input_dir = input_dir
        self.request_id = request_id
        self._armed = True

    def disarm(self) -> None:
        self._armed = False

    def cleanup(self) -> None:
        if self._armed:
            _cleanup_materialized_request_dir(self.input_dir, self.request_id)


def _unique_target_path(directory: Path, filename: str) -> Path:
    target = directory / _safe_filename_part(filename, "input")
    if not target.exists():
        return target
    stem = target.stem or "input"
    suffix = target.suffix
    return directory / f"{stem}-{uuid.uuid4().hex[:8]}{suffix}"


def _is_http_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _content_type_suffix(content_type: str) -> str:
    normalized = str(content_type or "").split(";", 1)[0].strip().lower()
    if not normalized:
        return ""
    guessed = mimetypes.guess_extension(normalized) or ""
    if guessed == ".jpe":
        return ".jpg"
    return guessed


def _video_download_filename(url: str, content_type: str) -> str:
    parsed = urllib.parse.urlparse(url)
    raw_name = Path(urllib.parse.unquote(parsed.path or "")).name
    suffix = Path(raw_name).suffix if raw_name else ""
    if not suffix:
        suffix = _content_type_suffix(content_type) or ".mp4"
    stem = Path(raw_name).stem if raw_name else "video"
    return f"{_safe_filename_part(stem, 'video')}{suffix}"


def _copy_video_path(source_value: str, request_dir: Path) -> str:
    source = Path(source_value).expanduser()
    if not source.is_file():
        raise ValueError(f"Video input does not point to a readable file: {source_value}")
    request_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_target_path(request_dir, source.name)
    shutil.copy2(source, target)
    _mark_materialized_request_dir(request_dir)
    return str(target.resolve())


def _download_video_url(url: str, request_dir: Path, open_url=None) -> str:
    parsed = urllib.parse.urlparse(url)
    request = urllib.request.Request(parsed.geturl(), headers={"User-Agent": "Cutlery-Workflow/1.0"})
    opener = open_url or urllib.request.urlopen
    request_dir.mkdir(parents=True, exist_ok=True)
    target: Path | None = None
    total = 0
    try:
        throw_if_interrupted()
        with opener(request, timeout=VIDEO_DOWNLOAD_TIMEOUT_SECONDS) as response:
            content_type = ""
            headers = getattr(response, "headers", None)
            if headers is not None:
                content_type = headers.get("Content-Type", "")
            target = _unique_target_path(request_dir, _video_download_filename(parsed.geturl(), content_type))
            with target.open("wb") as handle:
                while True:
                    throw_if_interrupted()
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_VIDEO_INPUT_BYTES:
                        raise ValueError("Video input download exceeded the 1000 MB limit.")
                    handle.write(chunk)
    except Exception:
        if target is not None:
            try:
                target.unlink()
            except OSError:
                pass
        raise
    if target is None or total <= 0:
        raise ValueError("Video input URL did not produce a readable video file.")
    _mark_materialized_request_dir(request_dir)
    return str(target.resolve())


def _materialize_video_value(value: Any, request_dir: Path, open_url=None) -> Any:
    if _blank(value):
        return None
    if isinstance(value, dict):
        value = value.get("path") or value.get("url") or value.get("file")
    if not isinstance(value, str):
        return value
    text = value.strip()
    if _is_http_url(text):
        LOGGER.info("[Cutlery Workflow] Materializing video URL host=%s", urllib.parse.urlparse(text).netloc)
        return _download_video_url(text, request_dir, open_url=open_url)
    return _copy_video_path(text, request_dir)


def materialize_run_values(
    values: dict[str, Any],
    *,
    ports: list[dict[str, Any]],
    input_dir: Path,
    request_id: str,
    open_url=None,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    request_dir = _request_input_dir(input_dir, request_id)
    for spec in ports:
        name = spec["name"]
        if name in values:
            value = values[name]
        elif "default" in spec:
            value = spec["default"]
        elif spec.get("required"):
            raise ValueError(f"Missing required workflow input {name!r}.")
        else:
            value = None

        if spec["type"] == "video":
            normalized[name] = _materialize_video_value(value, request_dir, open_url=open_url)
            continue

        if spec["type"] == "json":
            normalized[name] = _stringify_json_input(value)
            continue

        if spec["type"] != "image" or _blank(value):
            normalized[name] = value
            continue
        if not isinstance(value, (str, os.PathLike)):
            normalized[name] = value
            continue

        source = Path(value).expanduser()
        if not source.is_file():
            raise ValueError(f"Image input {name!r} does not point to a readable file: {value}")
        request_dir.mkdir(parents=True, exist_ok=True)
        target = request_dir / source.name
        if target.exists():
            target = request_dir / f"{source.stem}-{uuid.uuid4().hex[:8]}{source.suffix}"
        shutil.copy2(source, target)
        _mark_materialized_request_dir(request_dir)
        normalized[name] = target.relative_to(input_dir).as_posix()
    return normalized


def _is_api_prompt(workflow: Any) -> bool:
    return isinstance(workflow, dict) and all(
        isinstance(node, dict) and "class_type" in node for node in workflow.values()
    )


def _editor_links_by_id(workflow: dict[str, Any]) -> dict[Any, tuple[str, int]]:
    links: dict[Any, tuple[str, int]] = {}
    for link in workflow.get("links") or []:
        if isinstance(link, dict):
            link_id = link.get("id")
            origin_id = link.get("origin_id")
            origin_slot = link.get("origin_slot")
        elif isinstance(link, (list, tuple)) and len(link) >= 3:
            link_id, origin_id, origin_slot = link[0], link[1], link[2]
        else:
            continue
        if link_id is None or origin_id is None or origin_slot is None:
            continue
        links[link_id] = (str(origin_id), int(origin_slot))

    nodes = {
        str(node.get("id")): node
        for node in workflow.get("nodes") or []
        if isinstance(node, dict) and node.get("id") is not None
    }

    def resolve_origin(origin: tuple[str, int], seen: set[str]) -> tuple[str, int]:
        origin_id, _origin_slot = origin
        node = nodes.get(origin_id)
        if not isinstance(node, dict) or str(node.get("type") or "") != "Reroute" or origin_id in seen:
            return origin
        input_link = next(
            (
                record.get("link")
                for record in node.get("inputs") or []
                if isinstance(record, dict) and record.get("link") in links
            ),
            None,
        )
        if input_link is None:
            return origin
        return resolve_origin(links[input_link], {*seen, origin_id})

    for link_id, origin in list(links.items()):
        links[link_id] = resolve_origin(origin, set())
    return links


def _node_class(class_type: str) -> Any:
    if class_type in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS[class_type]
    try:
        import nodes as comfy_nodes
    except Exception:
        return None
    return getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(class_type)


def _widget_inputs(
    class_type: str,
    node: dict[str, Any],
    linked_names: set[str],
) -> list[tuple[str, bool, bool]]:
    class_def = _node_class(class_type)
    if class_def is None or not hasattr(class_def, "INPUT_TYPES"):
        return []
    try:
        input_types = class_def.INPUT_TYPES()
    except Exception:
        return []
    configured: list[tuple[str, bool, bool]] = []
    options_by_name: dict[str, dict[str, Any]] = {}
    for section in ("required", "optional"):
        records = input_types.get(section) if isinstance(input_types, dict) else None
        if not isinstance(records, dict):
            continue
        for name, spec in records.items():
            options = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict) else {}
            options_by_name[name] = options
            if options.get("forceInput"):
                continue
            configured.append((name, bool(options.get("control_after_generate")), name in linked_names))

    explicit: list[tuple[str, bool, bool]] = []
    for record in node.get("inputs") or []:
        if not isinstance(record, dict) or not isinstance(record.get("widget"), dict):
            continue
        name = str(record["widget"].get("name") or record.get("name") or "").strip()
        if not name:
            continue
        explicit.append(
            (
                name,
                bool(options_by_name.get(name, {}).get("control_after_generate")),
                record.get("link") is not None,
            )
        )
    return explicit or configured


def workflow_to_api_prompt(workflow: dict[str, Any]) -> dict[str, Any]:
    nodes = workflow.get("nodes") if isinstance(workflow, dict) else None
    if not isinstance(nodes, list):
        raise ValueError("Editor workflow JSON must contain a nodes array.")
    links = _editor_links_by_id(workflow)
    prompt: dict[str, Any] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "")
        class_type = str(node.get("type") or node.get("class_type") or "").strip()
        if not node_id or not class_type:
            continue
        if class_type in {"Note", "Reroute"}:
            continue
        api_inputs: dict[str, Any] = {}
        linked_names: set[str] = set()
        for input_index, input_record in enumerate(node.get("inputs") or []):
            if not isinstance(input_record, dict):
                continue
            name = str(input_record.get("name") or "").strip()
            link_id = input_record.get("link")
            if not name or link_id is None or link_id not in links:
                continue
            api_name = (
                VALUE_NAMES[input_index]
                if class_type in {"CutleryWorkflowInput", "CutleryWorkflowOutput"} and input_index < len(VALUE_NAMES)
                else name
            )
            api_inputs[api_name] = [links[link_id][0], links[link_id][1]]
            linked_names.add(api_name)

        widgets = node.get("widgets_values")
        if isinstance(widgets, dict):
            for name, value in widgets.items():
                if name not in api_inputs:
                    api_inputs[str(name)] = value
        elif isinstance(widgets, list):
            widget_inputs = _widget_inputs(class_type, node, linked_names)
            value_index = 0
            for input_index, (name, has_control, is_linked) in enumerate(widget_inputs):
                if value_index >= len(widgets):
                    break
                if not is_linked:
                    api_inputs[name] = widgets[value_index]
                value_index += 1
                remaining_inputs = len(widget_inputs) - input_index - 1
                if has_control and len(widgets) - value_index > remaining_inputs:
                    value_index += 1

        api_node = {
            "class_type": class_type,
            "inputs": api_inputs,
        }
        title = node.get("title") or node.get("properties", {}).get("Node name for S&R")
        if title:
            api_node["_meta"] = {"title": str(title)}
        prompt[node_id] = api_node
    return prompt


def _prompt_link(value: Any) -> tuple[str, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return str(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _normalize_api_boundary_input_names(workflow: dict[str, Any]) -> dict[str, Any]:
    normalized_workflow: dict[str, Any] | None = None
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or node.get("class_type") != "CutleryWorkflowOutput":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        normalized_inputs = inputs
        for index, spec in enumerate(parse_port_specs(inputs.get("ports_json", "[]")), start=1):
            internal_name = f"value_{index}"
            public_name = spec["name"]
            if internal_name in inputs or public_name not in inputs:
                continue
            if normalized_inputs is inputs:
                normalized_inputs = dict(inputs)
            normalized_inputs[internal_name] = normalized_inputs.pop(public_name)
        if normalized_inputs is inputs:
            continue
        if normalized_workflow is None:
            normalized_workflow = dict(workflow)
        normalized_workflow[node_id] = {**node, "inputs": normalized_inputs}
    return normalized_workflow or workflow


def _materialize_workflow_output_image_saves(workflow: dict[str, Any]) -> dict[str, Any]:
    saved_sources: set[tuple[str, int]] = set()
    for node in workflow.values():
        if not isinstance(node, dict) or node.get("class_type") != "SaveImage":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        source = _prompt_link(inputs.get("images"))
        if source is not None:
            saved_sources.add(source)
    missing: list[tuple[tuple[str, int], str]] = []
    for node in workflow.values():
        if not isinstance(node, dict) or node.get("class_type") != "CutleryWorkflowOutput":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for index, spec in enumerate(parse_port_specs(inputs.get("ports_json", "[]")), start=1):
            if spec["type"] != "image":
                continue
            source = _prompt_link(inputs.get(f"value_{index}"))
            if source is None or source in saved_sources:
                continue
            missing.append((source, spec["name"]))
            saved_sources.add(source)

    if not missing:
        return workflow

    materialized = dict(workflow)
    numeric_ids = [int(node_id) for node_id in materialized if str(node_id).isdigit()]
    next_id = max(numeric_ids, default=0) + 1
    for source, port_name in missing:
        while str(next_id) in materialized:
            next_id += 1
        materialized[str(next_id)] = {
            "class_type": "SaveImage",
            "inputs": {
                "images": [source[0], source[1]],
                "filename_prefix": f"cutlery/{_safe_filename_part(port_name, 'image')}",
            },
            "_meta": {
                "title": f"Save Workflow Output: {port_name}",
                "cutlery_materialized": "workflow_output_image",
            },
        }
        next_id += 1
    return materialized


def _materialized_workflow_output_save_ids(workflow: dict[str, Any]) -> list[str]:
    return [
        str(node_id)
        for node_id, node in workflow.items()
        if isinstance(node, dict)
        and isinstance(node.get("_meta"), dict)
        and node["_meta"].get("cutlery_materialized") == "workflow_output_image"
    ]


def normalize_workflow_json(workflow: Any) -> dict[str, Any]:
    if _is_api_prompt(workflow):
        prompt = _normalize_api_boundary_input_names(workflow)
    elif isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list):
        prompt = workflow_to_api_prompt(workflow)
    else:
        raise ValueError(
            "workflow must be ComfyUI API prompt JSON or editor workflow JSON with nodes and links."
        )
    return _materialize_workflow_output_image_saves(prompt)


def _workflow_input_ports(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    ports: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in workflow.values():
        if node.get("class_type") != "CutleryWorkflowInput":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for spec in parse_port_specs(inputs.get("ports_json", "[]")):
            if spec["name"] in seen:
                raise ValueError(f"Workflow input port {spec['name']!r} is duplicated across input nodes.")
            ports.append(spec)
            seen.add(spec["name"])
    return ports


def collect_wf3_outputs(history: dict[str, Any], prompt_id: str) -> dict[str, dict[str, Any]]:
    entry = history.get(prompt_id) if isinstance(history, dict) else None
    node_outputs = entry.get("outputs") if isinstance(entry, dict) else None
    if not isinstance(node_outputs, dict):
        return {}
    outputs: dict[str, dict[str, Any]] = {}
    for node_output in node_outputs.values():
        events = node_output.get("wf3") if isinstance(node_output, dict) else None
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            key = str(event.get("key") or "").strip()
            if not key:
                continue
            outputs[key] = {
                "type": str(event.get("type") or "string"),
                "value": event.get("value"),
            }
    return outputs


_MISSING_WORKFLOW_INPUT = object()


def _workflow_input_values(
    ports_json: str,
    wf3: dict[str, Any] | None,
    passthrough: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[Any]]:
    ports = parse_port_specs(ports_json)
    inputs = _wf3_inputs(wf3)
    values = []
    for index, spec in enumerate(ports, start=1):
        name = spec["name"]
        if name in inputs:
            value = inputs[name]
        elif f"value_{index}" in passthrough and not _blank(passthrough.get(f"value_{index}")):
            value = passthrough[f"value_{index}"]
        elif "default" in spec:
            value = spec["default"]
        elif spec.get("required"):
            value = _MISSING_WORKFLOW_INPUT
        else:
            value = None
        values.append(value)
    return ports, values


class CutleryWorkflowInput:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"value_{index + 1}": ("*", {"forceInput": True}) for index in range(MAX_WF3_PORTS)}
        return {
            "required": {
                "ports_json": (
                    "STRING",
                    {
                        "default": '[{"name":"prompt","type":"string"}]',
                        "multiline": True,
                    },
                ),
            },
            "optional": optional,
            "hidden": {
                "wf3": "WF3",
            },
        }

    RETURN_TYPES = tuple("*" for _ in range(MAX_WF3_PORTS))
    RETURN_NAMES = tuple(f"value_{index + 1}" for index in range(MAX_WF3_PORTS))
    FUNCTION = "read"
    CATEGORY = CATEGORY
    DESCRIPTION = "Request boundary node for typed Cutlery workflow inputs."

    def read(self, ports_json: str, wf3: dict[str, Any] | None = None, **kwargs):
        ports, values = _workflow_input_values(ports_json, wf3, kwargs)
        for spec, value in zip(ports, values):
            if value is _MISSING_WORKFLOW_INPUT:
                raise ValueError(f"Missing required workflow input {spec['name']!r}.")
        values = [_coerce_value(value, spec["type"]) for spec, value in zip(ports, values)]
        values.extend([None] * (MAX_WF3_PORTS - len(values)))
        return tuple(values)

    @classmethod
    def IS_CHANGED(cls, ports_json: str, wf3: dict[str, Any] | None = None, **kwargs):
        ports, values = _workflow_input_values(ports_json, wf3, kwargs)
        named_values = {
            spec["name"]: None if value is _MISSING_WORKFLOW_INPUT else value
            for spec, value in zip(ports, values)
        }
        return json.dumps({"ports": ports_json, "values": _json_safe(named_values)}, sort_keys=True)


class CutleryWorkflowOutput:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {f"value_{index + 1}": ("*", {"forceInput": True}) for index in range(MAX_WF3_PORTS)}
        return {
            "required": {
                "ports_json": (
                    "STRING",
                    {
                        "default": '[{"name":"result","type":"string"}]',
                        "multiline": True,
                    },
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = ()
    FUNCTION = "emit"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Response boundary node for typed Cutlery workflow outputs."

    def emit(self, ports_json: str, **kwargs):
        events = []
        for index, spec in enumerate(parse_port_specs(ports_json), start=1):
            value = kwargs.get(f"value_{index}")
            events.append(
                {
                    "key": spec["name"],
                    "type": spec["type"],
                    "value": _coerce_output_value(value, spec["type"], spec["name"]),
                }
            )
        return {"ui": {"wf3": events}}

    @classmethod
    def IS_CHANGED(cls, ports_json: str, **_kwargs):
        return float("nan")


def cancel_prompt(prompt_id: object, prompt_queue: Any | None = None) -> dict[str, Any]:
    """Cancel one queued or running ComfyUI prompt without affecting its neighbours."""

    clean_prompt_id = _clean_prompt_id(prompt_id)
    queue = prompt_queue
    if queue is None:
        if PromptServer is None:
            raise RuntimeError("ComfyUI PromptServer is not available.")
        queue = PromptServer.instance.prompt_queue

    def is_prompt(item: Any) -> bool:
        return isinstance(item, (list, tuple)) and len(item) > 1 and str(item[1]) == clean_prompt_id

    removed = bool(queue.delete_queue_item(is_prompt))
    interrupted = bool(queue.interrupt_if_running(clean_prompt_id))
    should_cleanup_inputs = removed or (not interrupted and prompt_cancellation_recorded(clean_prompt_id))
    if should_cleanup_inputs and folder_paths is not None:
        try:
            _cleanup_materialized_request_dir(Path(folder_paths.get_input_directory()), clean_prompt_id)
        except Exception:
            LOGGER.warning(
                "Could not clean materialized Cutlery inputs for cancelled queued prompt %s",
                clean_prompt_id,
                exc_info=True,
            )
    return {
        "ok": True,
        "prompt_id": clean_prompt_id,
        "removed_from_queue": removed,
        "interrupted_running": interrupted,
        "cancelled": removed or interrupted,
    }


async def _run_workflow(body: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if PromptServer is None or folder_paths is None:
        return {"ok": False, "error": "ComfyUI PromptServer is not available."}, 503
    try:
        submitted_workflow = body.get("workflow", body.get("prompt"))
        remote_target_remaps: dict[str, str] = {}
        if isinstance(submitted_workflow, dict) and isinstance(submitted_workflow.get("nodes"), list):
            workflow = workflow_to_api_prompt(submitted_workflow)
            try:
                from .nodes_remote import _compile_remote_groups_request
            except ImportError:
                from nodes_remote import _compile_remote_groups_request

            compilation = await _compile_remote_groups_request(
                {
                    "workflow": submitted_workflow,
                    "prompt": workflow,
                    "partial_execution_targets": body.get("partial_execution_targets"),
                }
            )
            workflow = compilation["prompt"]
            remote_target_remaps = compilation["remaps"]
            remote_targets = compilation["targets"]
            if remote_targets:
                LOGGER.info(
                    "[Cutlery Workflow] Compiled %d remote group(s) for WF3 HTTP execution targets=%s",
                    len(remote_targets),
                    ", ".join(remote_targets),
                )
            workflow = normalize_workflow_json(workflow)
        else:
            workflow = normalize_workflow_json(submitted_workflow)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, 400

    prompt_id = str(body.get("prompt_id") or uuid.uuid4())
    if consume_prompt_cancellation(prompt_id):
        return {
            "ok": False,
            "prompt_id": prompt_id,
            "cancelled": True,
            "error": "Workflow execution was cancelled before it was queued.",
        }, 409
    input_dir = Path(folder_paths.get_input_directory())
    materialization_guard = _PreQueueMaterializationGuard(input_dir, prompt_id)
    try:
        try:
            ports = _workflow_input_ports(workflow)
            values = body.get("values") if isinstance(body.get("values"), dict) else {}
            normalized_values = materialize_run_values(
                values,
                ports=ports,
                input_dir=input_dir,
                request_id=prompt_id,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc)}, 400

        try:
            import execution
        except Exception as exc:
            return {"ok": False, "error": f"Could not import ComfyUI execution module: {exc}"}, 500

        server = PromptServer.instance
        if hasattr(server, "node_replace_manager"):
            server.node_replace_manager.apply_replacements(workflow)

        partial_execution_targets = body.get("partial_execution_targets")
        if isinstance(partial_execution_targets, list):
            partial_execution_targets = [
                remote_target_remaps.get(str(target), target)
                for target in partial_execution_targets
            ]
            target_ids = {str(target) for target in partial_execution_targets}
            partial_execution_targets = list(partial_execution_targets)
            for node_id in _materialized_workflow_output_save_ids(workflow):
                if node_id not in target_ids:
                    partial_execution_targets.append(node_id)
        valid = await execution.validate_prompt(prompt_id, workflow, partial_execution_targets)
        if not valid[0]:
            return {"ok": False, "error": valid[1], "node_errors": valid[3]}, 400

        try:
            timeout_seconds = _positive_finite_seconds(
                body.get("timeout_seconds", 300),
                field_name="timeout_seconds",
                maximum=86400,
            )
            poll_seconds = _positive_finite_seconds(
                body.get("poll_seconds", 0.25),
                field_name="poll_seconds",
            )
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}, 400

        extra_data = body.get("extra_data") if isinstance(body.get("extra_data"), dict) else {}
        wf3 = extra_data.get("wf3") if isinstance(extra_data.get("wf3"), dict) else {}
        wf3["inputs"] = normalized_values
        wf3["ports"] = ports
        request_dir = _request_input_dir(input_dir, prompt_id).resolve()
        if (request_dir / MATERIALIZED_MARKER_NAME).is_file():
            wf3["_materialized_request_dir"] = str(request_dir)
        extra_data["wf3"] = wf3
        if "client_id" in body:
            extra_data["client_id"] = body["client_id"]
        extra_data["create_time"] = int(time.time() * 1000)

        number = float(body["number"]) if "number" in body else float(getattr(server, "number", 0))
        if "number" not in body:
            setattr(server, "number", number + 1)
        if body.get("front"):
            number = -number

        sensitive = {}
        for key in getattr(execution, "SENSITIVE_EXTRA_DATA_KEYS", []):
            if key in extra_data:
                sensitive[key] = extra_data.pop(key)

        if consume_prompt_cancellation(prompt_id):
            return {
                "ok": False,
                "prompt_id": prompt_id,
                "cancelled": True,
                "error": "Workflow execution was cancelled before it was queued.",
            }, 409

        server.prompt_queue.put((number, prompt_id, workflow, extra_data, valid[2], sensitive))
        materialization_guard.disarm()
    finally:
        materialization_guard.cleanup()

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        history = server.prompt_queue.get_history(prompt_id=prompt_id)
        if history:
            entry = history.get(prompt_id, {})
            status = entry.get("status") if isinstance(entry, dict) else None
            outputs = collect_wf3_outputs(history, prompt_id)
            ok = not isinstance(status, dict) or status.get("status_str") != "error"
            return {
                "ok": ok,
                "prompt_id": prompt_id,
                "outputs": outputs,
                "status": status,
                "node_errors": valid[3],
            }, 200 if ok else 500
        if consume_prompt_cancellation(prompt_id):
            cancellation = cancel_prompt(prompt_id, server.prompt_queue)
            if not cancellation["interrupted_running"]:
                _cleanup_materialized_request_dir(input_dir, prompt_id)
            return {
                "ok": False,
                "prompt_id": prompt_id,
                "cancelled": True,
                "error": "Workflow execution was cancelled.",
                "cancellation": cancellation,
            }, 409
        await asyncio.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))

    cancellation = cancel_prompt(prompt_id, server.prompt_queue)
    return {
        "ok": False,
        "prompt_id": prompt_id,
        "error": "Workflow execution timed out.",
        "cancellation": cancellation,
    }, 504


def register_wf3_boundary_routes() -> None:
    if PromptServer is None or web is None:
        return
    routes = PromptServer.instance.routes
    if getattr(routes, "_cutlery_wf3_boundary_routes_registered", False):
        return

    @routes.post("/cutlery/run")
    @routes.post("/cutlery/wf3/run")
    async def cutlery_run(request):
        if not _workflow_run_enabled():
            return _json_response(
                {
                    "ok": False,
                    "code": "workflow_run_disabled",
                    "error": (
                        "Network workflow execution is disabled. Set "
                        f"{WORKFLOW_RUN_ENABLED_ENV}=1 to enable it explicitly."
                    ),
                },
                status=403,
            )
        body = await _request_json(request)
        payload, status = await _run_workflow(body)
        return _json_response(payload, status=status)

    setattr(routes, "_cutlery_wf3_boundary_routes_registered", True)


register_wf3_boundary_routes()


NODE_CLASS_MAPPINGS = {
    "CutleryWorkflowInput": CutleryWorkflowInput,
    "CutleryWorkflowOutput": CutleryWorkflowOutput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CutleryWorkflowInput": "Workflow Input",
    "CutleryWorkflowOutput": "Workflow Output",
}


__all__ = [
    "CutleryWorkflowInput",
    "CutleryWorkflowOutput",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "collect_wf3_outputs",
    "materialize_run_values",
    "parse_port_specs",
    "register_wf3_boundary_routes",
    "workflow_to_api_prompt",
    "normalize_workflow_json",
]
