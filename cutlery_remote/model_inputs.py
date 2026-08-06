from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any


ModelTypeValue = str | tuple[str, ...]


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
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "").strip()
        input_map = LOADER_MODEL_INPUTS.get(class_type)
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
