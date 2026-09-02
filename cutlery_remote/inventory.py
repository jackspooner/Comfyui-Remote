from __future__ import annotations

import hashlib
import os
from pathlib import Path
import posixpath
from typing import Any


MODEL_TRANSFER_STAGING_PREFIX = ".cutlery-upload-"
MODEL_TRANSFER_STAGING_SUFFIX = ".part"

CANONICAL_MODEL_TYPES = (
    "checkpoints",
    "diffusion_models",
    "text_encoders",
    "vae",
    "loras",
    "clip_vision",
    "controlnet",
    "style_models",
    "upscale_models",
    "latent_upscale_models",
    "gligen",
    "clip_gguf",
    "unet_gguf",
    "vae_approx",
    "geometry_estimation",
    "audio_encoders",
    "wav2vec2",
    "nlf",
    "mmaudio",
    "background_removal",
    "frame_interpolation",
    "detection",
    "model_patches",
    "photomaker",
    "optical_flow",
    "ipadapter",
)

MODEL_TYPE_ALIASES = {
    "checkpoint": "checkpoints",
    "checkpoints": "checkpoints",
    "ckpt": "checkpoints",
    "diffusion": "diffusion_models",
    "diffusion_model": "diffusion_models",
    "diffusion_models": "diffusion_models",
    "unet": "diffusion_models",
    "clip": "text_encoders",
    "text_encoder": "text_encoders",
    "text_encoders": "text_encoders",
    "vae": "vae",
    "lora": "loras",
    "loras": "loras",
    "clip_vision": "clip_vision",
    "controlnet": "controlnet",
    "controlnets": "controlnet",
    "style_model": "style_models",
    "style_models": "style_models",
    "upscale_model": "upscale_models",
    "upscale_models": "upscale_models",
    "latent_upscale_model": "latent_upscale_models",
    "latent_upscale_models": "latent_upscale_models",
    "gligen": "gligen",
    "clip_gguf": "clip_gguf",
    "clipgguf": "clip_gguf",
    "unet_gguf": "unet_gguf",
    "unetgguf": "unet_gguf",
    "vae_approx": "vae_approx",
    "approx_vae": "vae_approx",
    "geometry": "geometry_estimation",
    "geometry_estimation": "geometry_estimation",
    "audio_encoder": "audio_encoders",
    "audio_encoders": "audio_encoders",
    "wav2vec": "wav2vec2",
    "wav2vec2": "wav2vec2",
    "nlf": "nlf",
    "mm_audio": "mmaudio",
    "mmaudio": "mmaudio",
    "background_removal": "background_removal",
    "frame_interpolation": "frame_interpolation",
    "frame_interpolation_model": "frame_interpolation",
    "face_detection": "detection",
    "detection": "detection",
    "model_patch": "model_patches",
    "model_patches": "model_patches",
    "photomaker": "photomaker",
    "optical_flow": "optical_flow",
    "ip_adapter": "ipadapter",
    "ipadapter": "ipadapter",
}


def normalize_model_type(model_type: object) -> str:
    key = str(model_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = MODEL_TYPE_ALIASES.get(key)
    if normalized is None:
        raise ValueError(f"Unsupported Cutlery remote model type {model_type!r}.")
    return normalized


def _folder_paths_module():
    import folder_paths  # type: ignore

    return folder_paths


def is_model_transfer_staging_name(value: object) -> bool:
    text = str(value or "").strip().replace("\\", "/")
    filename = posixpath.basename(text)
    return filename.startswith(MODEL_TRANSFER_STAGING_PREFIX) and filename.endswith(
        MODEL_TRANSFER_STAGING_SUFFIX
    )


def _safe_names(values: Any) -> list[str]:
    return sorted(
        {
            str(value or "").strip()
            for value in values or []
            if str(value or "").strip() and not is_model_transfer_staging_name(value)
        }
    )


def list_model_names(model_type: object) -> list[str]:
    folder_key = normalize_model_type(model_type)
    folder_paths = _folder_paths_module()
    try:
        names = folder_paths.get_filename_list(folder_key)
    except KeyError:
        return []
    return _safe_names(names)


def _model_filename(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    return posixpath.basename(text)


def _registry_name_key(value: object) -> str:
    portable_name = str(value or "").strip().replace("\\", os.sep).replace("/", os.sep)
    return os.path.normcase(portable_name)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_for_name(model_type: str, name: str, *, include_hashes: bool) -> dict[str, Any]:
    record: dict[str, Any] = {"model_type": model_type, "name": name}
    if include_hashes:
        folder_paths = _folder_paths_module()
        full_path = Path(folder_paths.get_full_path_or_raise(model_type, name))
        record["size"] = full_path.stat().st_size
        record["mtime"] = full_path.stat().st_mtime
        record["hash"] = _sha256_file(full_path)
    return record


def local_model_inventory(*, model_type: object | None = None, include_hashes: bool = False) -> dict[str, Any]:
    model_types = [normalize_model_type(model_type)] if model_type else list(CANONICAL_MODEL_TYPES)
    payload: dict[str, Any] = {
        "ok": True,
        "model_types": list(CANONICAL_MODEL_TYPES),
        "records": {},
    }
    records_by_type: dict[str, list[dict[str, Any]]] = {}
    for folder_key in model_types:
        names = list_model_names(folder_key)
        payload[folder_key] = names
        records_by_type[folder_key] = [
            _record_for_name(folder_key, name, include_hashes=include_hashes)
            for name in names
        ]
    payload["records"] = records_by_type
    return payload


def resolve_model_name(model_type: object, model_name: object) -> dict[str, Any]:
    folder_key = normalize_model_type(model_type)
    name = str(model_name or "").strip()
    if not name:
        return {"ok": False, "model_type": folder_key, "model_name": "", "error": "model_name is required."}
    available = list_model_names(folder_key)
    matches = [candidate for candidate in available if candidate == name]
    if not matches:
        requested_key = _registry_name_key(name)
        matches = [
            candidate
            for candidate in available
            if _registry_name_key(candidate) == requested_key
        ]
    if not matches:
        return {
            "ok": False,
            "model_type": folder_key,
            "model_name": name,
            "error": f"{name!r} is not available in remote model type {folder_key!r}.",
        }
    if len(matches) > 1:
        return {
            "ok": False,
            "model_type": folder_key,
            "model_name": name,
            "matches": matches,
            "error": (
                f"{name!r} matches multiple remote registry names in model type "
                f"{folder_key!r}: {', '.join(matches)}."
            ),
        }
    return {"ok": True, "model_type": folder_key, "model_name": matches[0]}


def find_local_model_by_filename(model_type: object, model_name: object) -> dict[str, Any]:
    folder_key = normalize_model_type(model_type)
    requested = str(model_name or "").strip()
    filename = _model_filename(requested)
    if not filename:
        return {
            "ok": False,
            "model_type": folder_key,
            "model_name": requested,
            "error": "model_name must include a filename.",
        }

    available = list_model_names(folder_key)
    exact = [name for name in available if name == requested]
    if not exact:
        requested_key = _registry_name_key(requested)
        exact = [
            name
            for name in available
            if _registry_name_key(name) == requested_key
        ]
    basename_matches = [name for name in available if _model_filename(name) == filename]
    matches = exact or basename_matches
    if not matches:
        return {
            "ok": False,
            "model_type": folder_key,
            "model_name": requested,
            "filename": filename,
            "error": f"No local {folder_key!r} model file named {filename!r} was found.",
        }
    if len(matches) > 1:
        return {
            "ok": False,
            "model_type": folder_key,
            "model_name": requested,
            "filename": filename,
            "matches": matches,
            "error": f"Multiple local {folder_key!r} model files named {filename!r} were found: {', '.join(matches)}.",
        }

    local_name = matches[0]
    folder_paths = _folder_paths_module()
    full_path = Path(folder_paths.get_full_path_or_raise(folder_key, local_name))
    return {
        "ok": True,
        "model_type": folder_key,
        "model_name": local_name,
        "filename": filename,
        "path": str(full_path),
        "size": full_path.stat().st_size,
        "mtime": full_path.stat().st_mtime,
    }
