from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import re
from typing import Any


ModelTypeValue = str | tuple[str, ...]

LOGGER = logging.getLogger("cutlery.remote.model_inputs")

MANIFEST_FILENAME = "cutlery_model_inputs.json"
MANIFEST_VERSION = 1
MANIFEST_KEYS = frozenset({"version", "inputs", "peer_managed"})
RULE_KEYS = frozenset({"by_extension", "default"})
_CATEGORY_RE = re.compile(r"^[a-z0-9_]+$")

# <comfy_root>/custom_nodes/Cutlery-Remote/cutlery_remote/model_inputs.py
CUSTOM_NODES_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelTypeSpec:
    primary: ModelTypeValue
    alternates: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelInputReference:
    node_id: str
    class_type: str
    input_name: str
    model_type: ModelTypeValue
    model_name: str
    alternate_model_types: tuple[str, ...] = field(default=(), compare=False)

    @property
    def model_types(self) -> tuple[str, ...]:
        if isinstance(self.model_type, tuple):
            return self.model_type
        return (self.model_type, *self.alternate_model_types)


ModelInputRule = ModelTypeValue | ModelTypeSpec | Callable[[str], ModelTypeValue | ModelTypeSpec | None]

SKIPPED_MODEL_SENTINELS = {
    "none",
    ".none",
    "automatic",
    "auto",
    ".auto",
    ".use_ckpt_clip",
    ".use_ckpt_vae",
    "taesd",
    "taesdxl",
    "taesd3",
    "taef1",
}
KJ_IMAGE_TAES = {"taesd", "taesdxl", "taesd3", "taef1"}
KJ_VIDEO_TAES = {"taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5"}


def _is_skipped_model_value(value: str) -> bool:
    return value.strip().lower() in SKIPPED_MODEL_SENTINELS


def _clip_gguf_or_text_encoder(value: str) -> str:
    return "clip_gguf" if value.lower().endswith(".gguf") else "text_encoders"


def _diffusion_or_connector(value: str) -> str:
    return "text_encoders" if "connector" in value.lower() else "diffusion_models"


def _gguf_extra_model(value: str) -> str | None:
    lower = value.lower()
    if _is_skipped_model_value(lower):
        return None
    if lower.endswith(".gguf"):
        return "unet_gguf"
    if "connector" in lower:
        return "text_encoders"
    return None


def _vae_kj_model_type(value: str) -> str | None:
    lower = value.lower()
    if lower == "pixel_space" or lower in KJ_IMAGE_TAES:
        return None
    stem = PurePosixPath(lower.replace("\\", "/")).name.rsplit(".", 1)[0]
    return "vae_approx" if stem in KJ_VIDEO_TAES else "vae"


def _primary_with_alternates(primary: str, *alternates: str) -> ModelTypeSpec:
    return ModelTypeSpec(primary=primary, alternates=alternates)


def _lora_slot_inputs(count: int = 16) -> dict[str, str]:
    return {f"lora_{index}": "loras" for index in range(count)}


LOADER_MODEL_INPUTS: dict[str, dict[str, ModelInputRule]] = {
    "CheckpointLoader": {"ckpt_name": "checkpoints"},
    "CheckpointLoaderSimple": {"ckpt_name": "checkpoints"},
    "unCLIPCheckpointLoader": {"ckpt_name": "checkpoints"},
    "ImageOnlyCheckpointLoader": {"ckpt_name": "checkpoints"},
    "UNETLoader": {"unet_name": "diffusion_models"},
    "VAELoader": {"vae_name": "vae"},
    "LoraLoader": {"lora_name": "loras"},
    "LoraLoaderModelOnly": {"lora_name": "loras"},
    "LoraLoaderBypass": {"lora_name": "loras"},
    "LoraLoaderBypassModelOnly": {"lora_name": "loras"},
    "CreateHookLora": {"lora_name": "loras"},
    "CreateHookLoraModelOnly": {"lora_name": "loras"},
    "CreateHookModelAsLora": {"ckpt_name": "checkpoints"},
    "CreateHookModelAsLoraModelOnly": {"ckpt_name": "checkpoints"},
    "ControlNetLoader": {"control_net_name": "controlnet"},
    "DiffControlNetLoader": {"control_net_name": "controlnet"},
    "CLIPLoader": {"clip_name": "text_encoders"},
    "CLIPLoaderGGUF": {"clip_name": _clip_gguf_or_text_encoder},
    "DualCLIPLoader": {"clip_name1": "text_encoders", "clip_name2": "text_encoders"},
    "DualCLIPLoaderGGUF": {"clip_name1": _clip_gguf_or_text_encoder, "clip_name2": _clip_gguf_or_text_encoder},
    "TripleCLIPLoader": {"clip_name1": "text_encoders", "clip_name2": "text_encoders", "clip_name3": "text_encoders"},
    "TripleCLIPLoaderGGUF": {
        "clip_name1": _clip_gguf_or_text_encoder,
        "clip_name2": _clip_gguf_or_text_encoder,
        "clip_name3": _clip_gguf_or_text_encoder,
    },
    "QuadrupleCLIPLoader": {
        "clip_name1": "text_encoders",
        "clip_name2": "text_encoders",
        "clip_name3": "text_encoders",
        "clip_name4": "text_encoders",
    },
    "QuadrupleCLIPLoaderGGUF": {
        "clip_name1": _clip_gguf_or_text_encoder,
        "clip_name2": _clip_gguf_or_text_encoder,
        "clip_name3": _clip_gguf_or_text_encoder,
        "clip_name4": _clip_gguf_or_text_encoder,
    },
    "UnetLoaderGGUF": {"unet_name": "unet_gguf"},
    "UnetLoaderGGUFAdvanced": {"unet_name": "unet_gguf"},
    "CLIPVisionLoader": {"clip_name": "clip_vision"},
    "StyleModelLoader": {"style_model_name": "style_models"},
    "UpscaleModelLoader": {"model_name": "upscale_models"},
    "GLIGENLoader": {"gligen_name": "gligen"},
    "CheckpointLoaderKJ": {"ckpt_name": "checkpoints"},
    "DiffusionModelLoaderKJ": {"model_name": "diffusion_models"},
    "DiffusionModelSelector": {"model_name": _diffusion_or_connector},
    "GGUFLoaderKJ": {"model_name": "unet_gguf", "extra_model_name": _gguf_extra_model},
    "VAELoaderKJ": {"vae_name": _vae_kj_model_type},
    "LoraReduceRankKJ": {"lora_name": "loras"},
    "LTX2LoraLoaderAdvanced": {"lora_name": "loras"},
    "DiTBlockLoraLoader": {"lora_name": "loras"},
    "LTXVAudioVAELoader": {"ckpt_name": "checkpoints"},
    "LTXAVTextEncoderLoader": {"text_encoder": "text_encoders", "ckpt_name": "checkpoints"},
    "GemmaAPITextEncode": {"ckpt_name": "checkpoints"},
    "LTXVGemmaCLIPModelLoader": {"gemma_path": "text_encoders", "ltxv_path": "checkpoints"},
    "LowVRAMCheckpointLoader": {"ckpt_name": "checkpoints"},
    "LowVRAMAudioVAELoader": {"ckpt_name": "checkpoints"},
    "LowVRAMLatentUpscaleModelLoader": {"model_name": "latent_upscale_models"},
    "LTXICLoRALoaderModelOnly": {"lora_name": "loras"},
    "LTXVQ8LoraModelLoader": {"lora_name": "loras"},
    "LTXVEditAnythingSplitLora": {"lora_name": "loras"},
    "LTXVEditAnythingModuleLoader": {"module_name": "loras"},
    "LTXVEditAnythingLoraLoader": {"lora_name": "loras"},
    "WanVideoModelLoader": {"model": "diffusion_models"},
    "WanVideoVAELoader": {"model_name": "vae"},
    "WanVideoTinyVAELoader": {"model_name": "vae_approx"},
    "LoadWanVideoT5TextEncoder": {"model_name": "text_encoders"},
    "LoadWanVideoClipTextEncoder": {"model_name": _primary_with_alternates("clip_vision", "text_encoders")},
    "WanVideoExtraModelSelect": {"extra_model": "diffusion_models"},
    "WanVideoVACEModelSelect": {"vace_model": "diffusion_models"},
    "WanVideoLoraSelect": {"lora": "loras"},
    "WanVideoLoraSelectMulti": _lora_slot_inputs(),
    "WanVideoControlnetLoader": {"model": "controlnet"},
    "QwenLoader": {"model": "text_encoders"},
    "WanVideoPromptExtenderSelect": {"model": "text_encoders"},
    "FantasyPortraitModelLoader": {"model": "diffusion_models"},
    "FantasyTalkingModelLoader": {"model": "diffusion_models"},
    "WanVideoFlashVSRDecoderLoader": {"model_name": "vae"},
    "WhisperModelLoader": {"model": "audio_encoders"},
    "LoadLynxResampler": {"model_name": "diffusion_models"},
    "LoadNLFModel": {"nlf_model": "nlf"},
    "LoadVQVAE": {"model_name": "vae"},
    "MultiTalkModelLoader": {"model": "diffusion_models"},
    "Wav2VecModelLoader": {"model": "wav2vec2"},
    "OviMMAudioVAELoader": {
        "vae": _primary_with_alternates("vae", "mmaudio"),
        "vocoder": _primary_with_alternates("vae", "mmaudio"),
    },
    "WanVideoUni3C_ControlnetLoader": {"model": "controlnet"},
    "FluxLoader": {
        "model_name": ("checkpoints", "diffusion_models"),
        "clip_name1": "text_encoders",
        "clip_name2_opt": "text_encoders",
        "vae_name": "vae",
        "clip_vision_name": "clip_vision",
        "style_model_name": "style_models",
    },
    "SD35Loader": {
        "model_name": ("checkpoints", "diffusion_models"),
        "clip_name1": "text_encoders",
        "clip_name2_opt": "text_encoders",
        "clip_name3_opt": "text_encoders",
        "vae_name": "vae",
    },
    "ClownModelLoader": {
        "model_name": ("checkpoints", "diffusion_models"),
        "clip_name1_opt": "text_encoders",
        "clip_name2_opt": "text_encoders",
        "clip_name3_opt": "text_encoders",
        "clip_name4_opt": "text_encoders",
        "vae_name": "vae",
    },
    "LayerPatcher": {"embedder": "diffusion_models", "gates": "diffusion_models", "last_layer": "diffusion_models"},
    "UltraSharkSampler Tiled": {"clip_name": "clip_vision"},
    "AudioEncoderLoader": {"audio_encoder_name": "audio_encoders"},
    "LoadBackgroundRemovalModel": {"bg_removal_name": "background_removal"},
    "LoadDA3Model": {"model_name": "geometry_estimation"},
    "LoadMoGeModel": {"model_name": "geometry_estimation"},
    "FrameInterpolationModelLoader": {"model_name": "frame_interpolation"},
    "LoadMediaPipeFaceLandmarker": {"model_name": "detection"},
    "ModelPatchLoader": {"name": "model_patches"},
    "PhotoMakerLoader": {"photomaker_model_name": "photomaker"},
    "OpticalFlowLoader": {"model_name": "optical_flow"},
    "IPAdapterModelLoader": {"ipadapter_file": "ipadapter"},
}


@dataclass(frozen=True)
class ModelInputManifest:
    """Model inputs declared by packages that ship their own loaders."""

    inputs: dict[str, dict[str, ModelInputRule]]
    peer_managed: frozenset[str]
    sources: tuple[str, ...]


def _valid_category(value: object) -> str | None:
    text = str(value or "").strip()
    return text if _CATEGORY_RE.match(text) else None


def _manifest_rule(spec: object) -> ModelInputRule | None:
    category = _valid_category(spec)
    if category is not None:
        return category
    if isinstance(spec, list):
        categories = [_valid_category(entry) for entry in spec]
        if not categories or any(entry is None for entry in categories):
            return None
        return ModelTypeSpec(primary=categories[0], alternates=tuple(categories[1:]))
    if not isinstance(spec, dict):
        return None
    if set(spec) - RULE_KEYS:
        return None
    default = _valid_category(spec.get("default"))
    by_extension = spec.get("by_extension")
    if default is None or not isinstance(by_extension, dict) or not by_extension:
        return None
    suffixes: dict[str, str] = {}
    for suffix, target in by_extension.items():
        suffix_text = str(suffix).strip().lower()
        target_category = _valid_category(target)
        if not suffix_text.startswith(".") or target_category is None:
            return None
        suffixes[suffix_text] = target_category

    def rule(value: str) -> str:
        lower = value.lower()
        for suffix, target_category in suffixes.items():
            if lower.endswith(suffix):
                return target_category
        return default

    return rule


def _parse_manifest(path: Path) -> tuple[dict[str, dict[str, ModelInputRule]], frozenset[str]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        LOGGER.warning("[Cutlery Remote] Ignoring model input manifest %s: %s", path, error)
        return None
    if not isinstance(payload, dict):
        LOGGER.warning("[Cutlery Remote] Ignoring model input manifest %s: expected a JSON object.", path)
        return None
    if payload.get("version") != MANIFEST_VERSION:
        LOGGER.warning(
            "[Cutlery Remote] Ignoring model input manifest %s: unsupported version %r.",
            path,
            payload.get("version"),
        )
        return None
    unknown = sorted(set(payload) - MANIFEST_KEYS)
    if unknown:
        LOGGER.warning(
            "[Cutlery Remote] Ignoring model input manifest %s: unsupported keys %s.",
            path,
            ", ".join(unknown),
        )
        return None
    raw_inputs = payload.get("inputs", {})
    raw_peer_managed = payload.get("peer_managed", [])
    if not isinstance(raw_inputs, dict) or not isinstance(raw_peer_managed, list):
        LOGGER.warning("[Cutlery Remote] Ignoring model input manifest %s: malformed sections.", path)
        return None

    inputs: dict[str, dict[str, ModelInputRule]] = {}
    for class_type, raw_map in raw_inputs.items():
        class_name = str(class_type or "").strip()
        if not class_name or not isinstance(raw_map, dict):
            LOGGER.warning("[Cutlery Remote] Ignoring %s entry in %s: malformed.", class_type, path)
            continue
        resolved: dict[str, ModelInputRule] = {}
        for input_name, rule_spec in raw_map.items():
            name = str(input_name or "").strip()
            rule = _manifest_rule(rule_spec)
            if not name or rule is None:
                LOGGER.warning(
                    "[Cutlery Remote] Ignoring %s.%s in %s: unsupported model input rule.",
                    class_name,
                    input_name,
                    path,
                )
                continue
            resolved[name] = rule
        if resolved:
            inputs[class_name] = resolved

    peer_managed = frozenset(
        str(name).strip() for name in raw_peer_managed if str(name or "").strip()
    )
    return inputs, peer_managed


def _load_manifests(root: Path | None = None) -> ModelInputManifest:
    search_root = Path(root) if root is not None else CUSTOM_NODES_ROOT
    inputs: dict[str, dict[str, ModelInputRule]] = {}
    peer_managed: set[str] = set()
    sources: list[str] = []
    try:
        candidates = sorted(search_root.glob(f"*/{MANIFEST_FILENAME}"))
    except OSError:
        candidates = []
    for path in candidates:
        parsed = _parse_manifest(path)
        if parsed is None:
            continue
        package_inputs, package_peer_managed = parsed
        for class_type, mapping in package_inputs.items():
            inputs.setdefault(class_type, {}).update(mapping)
        peer_managed.update(package_peer_managed)
        sources.append(path.parent.name)
    if sources:
        LOGGER.info(
            "[Cutlery Remote] Loaded model input manifests from %s (%s classes, %s peer-managed).",
            ", ".join(sources),
            len(inputs),
            len(peer_managed),
        )
    return ModelInputManifest(
        inputs=inputs,
        peer_managed=frozenset(peer_managed),
        sources=tuple(sources),
    )


_manifest_cache: ModelInputManifest | None = None
_merged_cache: dict[str, dict[str, ModelInputRule]] | None = None


def reset_manifest_cache() -> None:
    """Drop the cached manifests so the next call re-reads the filesystem."""

    global _manifest_cache, _merged_cache
    _manifest_cache = None
    _merged_cache = None


def model_input_manifest() -> ModelInputManifest:
    global _manifest_cache
    if _manifest_cache is None:
        _manifest_cache = _load_manifests()
    return _manifest_cache


def loader_model_inputs() -> dict[str, dict[str, ModelInputRule]]:
    """Return the built-in loader table merged with package manifests."""

    global _merged_cache
    if _merged_cache is None:
        merged = {class_type: dict(mapping) for class_type, mapping in LOADER_MODEL_INPUTS.items()}
        for class_type, mapping in model_input_manifest().inputs.items():
            merged.setdefault(class_type, {}).update(mapping)
        _merged_cache = merged
    return _merged_cache


def peer_managed_class_types() -> frozenset[str]:
    return model_input_manifest().peer_managed


def iter_peer_managed_nodes(workflow: Any):
    """Yield node ids whose loader copies models through its own path settings."""

    if not isinstance(workflow, dict):
        return
    peer_managed = peer_managed_class_types()
    if not peer_managed:
        return
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "").strip()
        if class_type in peer_managed:
            yield str(node_id), class_type


def _resolve_model_type(rule: ModelInputRule, value: str) -> ModelTypeSpec | None:
    resolved = rule(value) if callable(rule) else rule
    if resolved is None:
        return None
    if isinstance(resolved, ModelTypeSpec):
        return resolved
    return ModelTypeSpec(primary=resolved)


def iter_loader_model_inputs(workflow: Any):
    if not isinstance(workflow, dict):
        return
    loaders = loader_model_inputs()
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "").strip()
        input_map = loaders.get(class_type)
        if not input_map:
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_name, model_type in input_map.items():
            value = inputs.get(input_name)
            if isinstance(value, str) and value.strip():
                model_name = value.strip()
                if _is_skipped_model_value(model_name):
                    continue
                resolved = _resolve_model_type(model_type, model_name)
                if resolved is None:
                    continue
                yield ModelInputReference(
                    node_id=str(node_id),
                    class_type=class_type,
                    input_name=input_name,
                    model_type=resolved.primary,
                    model_name=model_name,
                    alternate_model_types=resolved.alternates,
                )
