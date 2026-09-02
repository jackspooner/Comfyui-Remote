import importlib.util
import asyncio
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _Routes:
    def get(self, _path):
        return lambda fn: fn

    def post(self, _path):
        return lambda fn: fn


def _load_nodes_remote():
    package_name = "cutlery_nodes_remote_model_node_test_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPO_ROOT)]
    sys.modules[package_name] = package
    sys.modules["server"] = types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=_Routes())))
    sys.modules["aiohttp"] = types.SimpleNamespace(web=types.SimpleNamespace(json_response=lambda payload, status=200, **_: payload))
    module_name = f"{package_name}.nodes_remote"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "nodes_remote.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module.resolve_trusted_remote_target = lambda value: module.TrustedRemoteTarget(
        name="test-target",
        base_url=module._clean_base_url(value),
        canonical=f"cutlery://{str(value).removeprefix('http://').removeprefix('https://')}",
        display_label=str(value),
        copy_host="renderhost",
        copy_root="D:/ComfyUI/models",
    )
    module._get_remote_json = lambda *_args, **_kwargs: {
        "ok": True,
        "protocol_version": 1,
        "serializers": ["primitive", "tensor", "latent", "conditioning", "cutlery_lora_chain"],
        "features": {
            "remote_groups": True,
            "remote_node_definitions_v1": True,
            "prompt_specific_interrupt": True,
            "remote_lora_chain_boundary_v1": True,
        },
    }
    module._real_preflight_remote_workflow = module._preflight_remote_workflow
    module._preflight_remote_workflow = lambda *_args, **_kwargs: {"ok": True, "nodes": {}}
    return module


def _compiled_remote_workflow(input_ports, output_ports):
    return {
        "input": {
            "class_type": "CutleryWorkflowInput",
            "inputs": {"ports_json": json.dumps(input_ports)},
        },
        "work": {"class_type": "NoOp", "inputs": {}},
        "output": {
            "class_type": "CutleryWorkflowOutput",
            "inputs": {"ports_json": json.dumps(output_ports)},
        },
    }


class RemoteModelNodeTests(unittest.TestCase):
    def test_remote_group_executor_uses_compiler_selected_cache_policy(self):
        module = _load_nodes_remote()

        self.assertEqual(
            module.CutleryRemoteGroupExecutor.IS_CHANGED(cache_policy=module.REMOTE_GROUP_CACHE_POLICY_SENDER_V1),
            module.REMOTE_GROUP_CACHE_POLICY_SENDER_V1,
        )
        self.assertNotEqual(
            module.CutleryRemoteGroupExecutor.IS_CHANGED(cache_policy=module.REMOTE_GROUP_CACHE_POLICY_REMOTE),
            module.CutleryRemoteGroupExecutor.IS_CHANGED(cache_policy=module.REMOTE_GROUP_CACHE_POLICY_REMOTE),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported remote group cache_policy"):
            module.CutleryRemoteGroupExecutor().run_remote_group(
                remote_base_url="192.0.2.247:8188",
                remote_workflow_json="{}",
                input_ports_json="[]",
                output_ports_json="[]",
                cache_policy="unknown",
            )

    def test_remote_group_preparation_uses_policy_specific_fingerprint(self):
        module = _load_nodes_remote()

        self.assertEqual(
            module.CutleryRemoteGroupPreparation.IS_CHANGED(cache_policy=module.REMOTE_GROUP_CACHE_POLICY_SENDER_V1),
            module.REMOTE_GROUP_PREPARATION_FINGERPRINT_VERSION,
        )
        self.assertNotEqual(
            module.CutleryRemoteGroupPreparation.IS_CHANGED(cache_policy=module.REMOTE_GROUP_CACHE_POLICY_REMOTE),
            module.CutleryRemoteGroupPreparation.IS_CHANGED(cache_policy=module.REMOTE_GROUP_CACHE_POLICY_REMOTE),
        )

    def test_remote_group_executor_roles_are_registered_and_documented(self):
        module = _load_nodes_remote()

        self.assertTrue(module.CutleryRemoteGroupExecutor.OUTPUT_NODE)
        self.assertFalse(module.CutleryRemoteGroupValueExecutor.OUTPUT_NODE)
        self.assertTrue(issubclass(module.CutleryRemoteGroupValueExecutor, module.CutleryRemoteGroupExecutor))
        self.assertIs(module.NODE_CLASS_MAPPINGS["CutleryRemoteGroupExecutor"], module.CutleryRemoteGroupExecutor)
        self.assertIs(module.NODE_CLASS_MAPPINGS["CutleryRemoteGroupValueExecutor"], module.CutleryRemoteGroupValueExecutor)
        self.assertTrue(module.NODE_DISPLAY_NAME_MAPPINGS["CutleryRemoteGroupValueExecutor"])
        for executor in (module.CutleryRemoteGroupExecutor, module.CutleryRemoteGroupValueExecutor):
            self.assertTrue(executor.DESCRIPTION)
            for section in ("required", "optional"):
                for input_spec in executor.INPUT_TYPES()[section].values():
                    self.assertTrue(input_spec[1].get("tooltip"))

    def test_model_input_registry_covers_batch_4_wan_video_wrapper_nodes(self):
        from cutlery_remote.model_inputs import iter_loader_model_inputs

        workflow = {
            "wan": {"class_type": "WanVideoModelLoader", "inputs": {"model": "wan/model.safetensors"}},
            "vae": {"class_type": "WanVideoVAELoader", "inputs": {"model_name": "wan_vae.safetensors"}},
            "tiny": {"class_type": "WanVideoTinyVAELoader", "inputs": {"model_name": "taew2_1.safetensors"}},
            "t5": {"class_type": "LoadWanVideoT5TextEncoder", "inputs": {"model_name": "umt5_xxl.safetensors"}},
            "clip": {"class_type": "LoadWanVideoClipTextEncoder", "inputs": {"model_name": "clip_vision.safetensors"}},
            "extra": {"class_type": "WanVideoExtraModelSelect", "inputs": {"extra_model": "vace.safetensors"}},
            "vace": {"class_type": "WanVideoVACEModelSelect", "inputs": {"vace_model": "vace_module.safetensors"}},
            "lora": {"class_type": "WanVideoLoraSelect", "inputs": {"lora": "wan_lora.safetensors"}},
            "lora_multi": {"class_type": "WanVideoLoraSelectMulti", "inputs": {"lora_0": "motion.safetensors", "lora_1": "none"}},
            "control": {"class_type": "WanVideoControlnetLoader", "inputs": {"model": "control.safetensors"}},
            "qwen": {"class_type": "QwenLoader", "inputs": {"model": "qwen_3b.safetensors"}},
            "extender": {"class_type": "WanVideoPromptExtenderSelect", "inputs": {"model": "qwen_7b.safetensors"}},
            "portrait": {"class_type": "FantasyPortraitModelLoader", "inputs": {"model": "portrait.safetensors"}},
            "talking": {"class_type": "FantasyTalkingModelLoader", "inputs": {"model": "talking.safetensors"}},
            "flashvsr": {"class_type": "WanVideoFlashVSRDecoderLoader", "inputs": {"model_name": "flash_vae.safetensors"}},
            "whisper": {"class_type": "WhisperModelLoader", "inputs": {"model": "whisper.safetensors"}},
            "lynx": {"class_type": "LoadLynxResampler", "inputs": {"model_name": "lynx.safetensors"}},
            "nlf": {"class_type": "LoadNLFModel", "inputs": {"nlf_model": "nlf_l_multi_0.3.2.torchscript"}},
            "vqvae": {"class_type": "LoadVQVAE", "inputs": {"model_name": "vqvae.safetensors"}},
            "multitalk": {"class_type": "MultiTalkModelLoader", "inputs": {"model": "multitalk.safetensors"}},
            "wav2vec": {"class_type": "Wav2VecModelLoader", "inputs": {"model": "wav2vec.safetensors"}},
            "ovi": {"class_type": "OviMMAudioVAELoader", "inputs": {"vae": "mmaudio_vae.safetensors", "vocoder": "bigvgan.safetensors"}},
            "uni3c": {"class_type": "WanVideoUni3C_ControlnetLoader", "inputs": {"model": "uni3c_control.safetensors"}},
        }

        refs = list(iter_loader_model_inputs(workflow))
        compact = [(ref.node_id, ref.class_type, ref.input_name, ref.model_type, ref.model_name) for ref in refs]

        self.assertIn(("wan", "WanVideoModelLoader", "model", "diffusion_models", "wan/model.safetensors"), compact)
        self.assertIn(("vae", "WanVideoVAELoader", "model_name", "vae", "wan_vae.safetensors"), compact)
        self.assertIn(("tiny", "WanVideoTinyVAELoader", "model_name", "vae_approx", "taew2_1.safetensors"), compact)
        self.assertIn(("t5", "LoadWanVideoT5TextEncoder", "model_name", "text_encoders", "umt5_xxl.safetensors"), compact)
        self.assertIn(("clip", "LoadWanVideoClipTextEncoder", "model_name", "clip_vision", "clip_vision.safetensors"), compact)
        self.assertIn(("extra", "WanVideoExtraModelSelect", "extra_model", "diffusion_models", "vace.safetensors"), compact)
        self.assertIn(("vace", "WanVideoVACEModelSelect", "vace_model", "diffusion_models", "vace_module.safetensors"), compact)
        self.assertIn(("lora", "WanVideoLoraSelect", "lora", "loras", "wan_lora.safetensors"), compact)
        self.assertIn(("lora_multi", "WanVideoLoraSelectMulti", "lora_0", "loras", "motion.safetensors"), compact)
        self.assertNotIn(("lora_multi", "WanVideoLoraSelectMulti", "lora_1", "loras", "none"), compact)
        self.assertIn(("control", "WanVideoControlnetLoader", "model", "controlnet", "control.safetensors"), compact)
        self.assertIn(("qwen", "QwenLoader", "model", "text_encoders", "qwen_3b.safetensors"), compact)
        self.assertIn(("extender", "WanVideoPromptExtenderSelect", "model", "text_encoders", "qwen_7b.safetensors"), compact)
        self.assertIn(("portrait", "FantasyPortraitModelLoader", "model", "diffusion_models", "portrait.safetensors"), compact)
        self.assertIn(("talking", "FantasyTalkingModelLoader", "model", "diffusion_models", "talking.safetensors"), compact)
        self.assertIn(("flashvsr", "WanVideoFlashVSRDecoderLoader", "model_name", "vae", "flash_vae.safetensors"), compact)
        self.assertIn(("whisper", "WhisperModelLoader", "model", "audio_encoders", "whisper.safetensors"), compact)
        self.assertIn(("lynx", "LoadLynxResampler", "model_name", "diffusion_models", "lynx.safetensors"), compact)
        self.assertIn(("nlf", "LoadNLFModel", "nlf_model", "nlf", "nlf_l_multi_0.3.2.torchscript"), compact)
        self.assertIn(("vqvae", "LoadVQVAE", "model_name", "vae", "vqvae.safetensors"), compact)
        self.assertIn(("multitalk", "MultiTalkModelLoader", "model", "diffusion_models", "multitalk.safetensors"), compact)
        self.assertIn(("wav2vec", "Wav2VecModelLoader", "model", "wav2vec2", "wav2vec.safetensors"), compact)
        self.assertIn(("ovi", "OviMMAudioVAELoader", "vae", "vae", "mmaudio_vae.safetensors"), compact)
        self.assertIn(("ovi", "OviMMAudioVAELoader", "vocoder", "vae", "bigvgan.safetensors"), compact)
        self.assertIn(("uni3c", "WanVideoUni3C_ControlnetLoader", "model", "controlnet", "uni3c_control.safetensors"), compact)
        ovi_vae = next(ref for ref in refs if ref.node_id == "ovi" and ref.input_name == "vae")
        clip = next(ref for ref in refs if ref.node_id == "clip" and ref.input_name == "model_name")
        self.assertEqual(ovi_vae.model_types, ("vae", "mmaudio"))
        self.assertEqual(clip.model_types, ("clip_vision", "text_encoders"))

    def test_model_input_registry_covers_batch_3_ltxvideo_and_bfs_nodes(self):
        from cutlery_remote.model_inputs import ModelInputReference, iter_loader_model_inputs

        workflow = {
            "ltx_api": {"class_type": "GemmaAPITextEncode", "inputs": {"ckpt_name": "ltxv/model.safetensors"}},
            "ltx_gemma": {
                "class_type": "LTXVGemmaCLIPModelLoader",
                "inputs": {
                    "gemma_path": "gemma/gemma-3.safetensors",
                    "ltxv_path": "ltxv/ltxv-2b.safetensors",
                },
            },
            "ltx_low_audio": {"class_type": "LowVRAMAudioVAELoader", "inputs": {"ckpt_name": "ltxv/audio_vae.safetensors"}},
            "ltx_upscale": {"class_type": "LowVRAMLatentUpscaleModelLoader", "inputs": {"model_name": "latent/upscale.safetensors"}},
            "ltx_iclora": {"class_type": "LTXICLoRALoaderModelOnly", "inputs": {"lora_name": "ltxv/ic_lora.safetensors"}},
            "ltx_q8": {"class_type": "LTXVQ8LoraModelLoader", "inputs": {"lora_name": "ltxv/q8_lora.safetensors"}},
            "bfs_module": {"class_type": "LTXVEditAnythingModuleLoader", "inputs": {"module_name": "ea/module.safetensors"}},
            "bfs_lora": {"class_type": "LTXVEditAnythingLoraLoader", "inputs": {"lora_name": "ea/edit-anything.safetensors"}},
        }

        refs = list(iter_loader_model_inputs(workflow))

        self.assertEqual(
            refs,
            [
                ModelInputReference("ltx_api", "GemmaAPITextEncode", "ckpt_name", "checkpoints", "ltxv/model.safetensors"),
                ModelInputReference("ltx_gemma", "LTXVGemmaCLIPModelLoader", "gemma_path", "text_encoders", "gemma/gemma-3.safetensors"),
                ModelInputReference("ltx_gemma", "LTXVGemmaCLIPModelLoader", "ltxv_path", "checkpoints", "ltxv/ltxv-2b.safetensors"),
                ModelInputReference("ltx_low_audio", "LowVRAMAudioVAELoader", "ckpt_name", "checkpoints", "ltxv/audio_vae.safetensors"),
                ModelInputReference("ltx_upscale", "LowVRAMLatentUpscaleModelLoader", "model_name", "latent_upscale_models", "latent/upscale.safetensors"),
                ModelInputReference("ltx_iclora", "LTXICLoRALoaderModelOnly", "lora_name", "loras", "ltxv/ic_lora.safetensors"),
                ModelInputReference("ltx_q8", "LTXVQ8LoraModelLoader", "lora_name", "loras", "ltxv/q8_lora.safetensors"),
                ModelInputReference("bfs_module", "LTXVEditAnythingModuleLoader", "module_name", "loras", "ea/module.safetensors"),
                ModelInputReference("bfs_lora", "LTXVEditAnythingLoraLoader", "lora_name", "loras", "ea/edit-anything.safetensors"),
            ],
        )

    def test_model_input_registry_covers_batch_5_res4lyf_nodes_and_skips_sentinels(self):
        from cutlery_remote.model_inputs import ModelInputReference, iter_loader_model_inputs

        workflow = {
            "flux": {
                "class_type": "FluxLoader",
                "inputs": {
                    "model_name": "flux1-dev.safetensors",
                    "clip_name1": ".use_ckpt_clip",
                    "clip_name2_opt": "t5xxl_fp16.safetensors",
                    "vae_name": ".use_ckpt_vae",
                    "clip_vision_name": ".none",
                    "style_model_name": "style.safetensors",
                },
            },
            "clown": {
                "class_type": "ClownModelLoader",
                "inputs": {
                    "model_name": "wan.sft",
                    "clip_name1_opt": ".none",
                    "clip_name2_opt": "umt5_xxl_fp8.safetensors",
                    "clip_name3_opt": ".none",
                    "clip_name4_opt": "clip_l.safetensors",
                    "vae_name": "ae.safetensors",
                },
            },
            "sd35": {
                "class_type": "SD35Loader",
                "inputs": {
                    "model_name": "sd35_medium.safetensors",
                    "clip_name1": "clip_g.safetensors",
                    "clip_name2_opt": ".none",
                    "clip_name3_opt": "t5xxl_fp16.safetensors",
                    "vae_name": "taesd3",
                },
            },
            "patcher": {
                "class_type": "LayerPatcher",
                "inputs": {
                    "embedder": "embedder.safetensors",
                    "gates": "gates.safetensors",
                    "last_layer": "last_layer.safetensors",
                },
            },
            "tiled": {
                "class_type": "UltraSharkSampler Tiled",
                "inputs": {"clip_name": "clip-vit-large-patch14.safetensors"},
            },
            "latent": {
                "class_type": "EmptyLatentImage64",
                "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
            },
        }

        refs = list(iter_loader_model_inputs(workflow))

        self.assertEqual(
            refs,
            [
                ModelInputReference("flux", "FluxLoader", "model_name", ("checkpoints", "diffusion_models"), "flux1-dev.safetensors"),
                ModelInputReference("flux", "FluxLoader", "clip_name2_opt", "text_encoders", "t5xxl_fp16.safetensors"),
                ModelInputReference("flux", "FluxLoader", "style_model_name", "style_models", "style.safetensors"),
                ModelInputReference("clown", "ClownModelLoader", "model_name", ("checkpoints", "diffusion_models"), "wan.sft"),
                ModelInputReference("clown", "ClownModelLoader", "clip_name2_opt", "text_encoders", "umt5_xxl_fp8.safetensors"),
                ModelInputReference("clown", "ClownModelLoader", "clip_name4_opt", "text_encoders", "clip_l.safetensors"),
                ModelInputReference("clown", "ClownModelLoader", "vae_name", "vae", "ae.safetensors"),
                ModelInputReference("sd35", "SD35Loader", "model_name", ("checkpoints", "diffusion_models"), "sd35_medium.safetensors"),
                ModelInputReference("sd35", "SD35Loader", "clip_name1", "text_encoders", "clip_g.safetensors"),
                ModelInputReference("sd35", "SD35Loader", "clip_name3_opt", "text_encoders", "t5xxl_fp16.safetensors"),
                ModelInputReference("patcher", "LayerPatcher", "embedder", "diffusion_models", "embedder.safetensors"),
                ModelInputReference("patcher", "LayerPatcher", "gates", "diffusion_models", "gates.safetensors"),
                ModelInputReference("patcher", "LayerPatcher", "last_layer", "diffusion_models", "last_layer.safetensors"),
                ModelInputReference("tiled", "UltraSharkSampler Tiled", "clip_name", "clip_vision", "clip-vit-large-patch14.safetensors"),
            ],
        )

    def test_model_input_registry_covers_batch_2_gguf_and_kj_nodes(self):
        from cutlery_remote.model_inputs import ModelInputReference, iter_loader_model_inputs

        workflow = {
            "quad": {
                "class_type": "QuadrupleCLIPLoaderGGUF",
                "inputs": {
                    "clip_name1": "clip_l.safetensors",
                    "clip_name2": "qwen_text.gguf",
                    "clip_name3": "t5xxl.gguf",
                    "clip_name4": "clip_g.safetensors",
                },
            },
            "unet": {"class_type": "UnetLoaderGGUFAdvanced", "inputs": {"unet_name": "wan.gguf"}},
            "ckpt_kj": {"class_type": "CheckpointLoaderKJ", "inputs": {"ckpt_name": "base.safetensors"}},
            "selector": {"class_type": "DiffusionModelSelector", "inputs": {"model_name": "ltxv_connector.safetensors"}},
            "gguf_kj": {
                "class_type": "GGUFLoaderKJ",
                "inputs": {"model_name": "ltxv.gguf", "extra_model_name": "vace_connector.safetensors"},
            },
            "vae_kj": {"class_type": "VAELoaderKJ", "inputs": {"vae_name": "ae.safetensors"}},
            "reduce": {"class_type": "LoraReduceRankKJ", "inputs": {"lora_name": "style.safetensors"}},
            "ltx2": {"class_type": "LTX2LoraLoaderAdvanced", "inputs": {"lora_name": "ltx2_style.safetensors"}},
            "dit": {"class_type": "DiTBlockLoraLoader", "inputs": {"lora_name": "blocks.safetensors"}},
        }

        refs = list(iter_loader_model_inputs(workflow))

        self.assertEqual(
            refs,
            [
                ModelInputReference("quad", "QuadrupleCLIPLoaderGGUF", "clip_name1", "text_encoders", "clip_l.safetensors"),
                ModelInputReference("quad", "QuadrupleCLIPLoaderGGUF", "clip_name2", "clip_gguf", "qwen_text.gguf"),
                ModelInputReference("quad", "QuadrupleCLIPLoaderGGUF", "clip_name3", "clip_gguf", "t5xxl.gguf"),
                ModelInputReference("quad", "QuadrupleCLIPLoaderGGUF", "clip_name4", "text_encoders", "clip_g.safetensors"),
                ModelInputReference("unet", "UnetLoaderGGUFAdvanced", "unet_name", "unet_gguf", "wan.gguf"),
                ModelInputReference("ckpt_kj", "CheckpointLoaderKJ", "ckpt_name", "checkpoints", "base.safetensors"),
                ModelInputReference("selector", "DiffusionModelSelector", "model_name", "text_encoders", "ltxv_connector.safetensors"),
                ModelInputReference("gguf_kj", "GGUFLoaderKJ", "model_name", "unet_gguf", "ltxv.gguf"),
                ModelInputReference("gguf_kj", "GGUFLoaderKJ", "extra_model_name", "text_encoders", "vace_connector.safetensors"),
                ModelInputReference("vae_kj", "VAELoaderKJ", "vae_name", "vae", "ae.safetensors"),
                ModelInputReference("reduce", "LoraReduceRankKJ", "lora_name", "loras", "style.safetensors"),
                ModelInputReference("ltx2", "LTX2LoraLoaderAdvanced", "lora_name", "loras", "ltx2_style.safetensors"),
                ModelInputReference("dit", "DiTBlockLoraLoader", "lora_name", "loras", "blocks.safetensors"),
            ],
        )

    def test_model_input_registry_skips_batch_2_virtual_values(self):
        from cutlery_remote.model_inputs import ModelInputReference, iter_loader_model_inputs

        workflow = {
            "gguf_kj": {"class_type": "GGUFLoaderKJ", "inputs": {"model_name": "wan.gguf", "extra_model_name": "none"}},
            "pixel": {"class_type": "VAELoaderKJ", "inputs": {"vae_name": "pixel_space"}},
            "taesd": {"class_type": "VAELoaderKJ", "inputs": {"vae_name": "taesd"}},
        }

        refs = list(iter_loader_model_inputs(workflow))

        self.assertEqual(refs, [ModelInputReference("gguf_kj", "GGUFLoaderKJ", "model_name", "unet_gguf", "wan.gguf")])

    def test_model_input_registry_covers_batch_6_media_resource_loaders(self):
        from cutlery_remote.model_inputs import ModelInputReference, iter_loader_model_inputs

        workflow = {
            "audio": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": "audio/encoder.safetensors"}},
            "bg": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "birefnet.safetensors"}},
            "da3": {"class_type": "LoadDA3Model", "inputs": {"model_name": "depth_anything_3.safetensors", "weight_dtype": "fp16"}},
            "moge": {"class_type": "LoadMoGeModel", "inputs": {"model_name": "moge.safetensors"}},
            "interp": {"class_type": "FrameInterpolationModelLoader", "inputs": {"model_name": "rife.pth"}},
            "face": {"class_type": "LoadMediaPipeFaceLandmarker", "inputs": {"model_name": "face_landmarker.safetensors"}},
            "patch": {"class_type": "ModelPatchLoader", "inputs": {"name": "qwen_patch.safetensors"}},
            "photo": {"class_type": "PhotoMakerLoader", "inputs": {"photomaker_model_name": "photomaker-v1.bin"}},
            "flow": {"class_type": "OpticalFlowLoader", "inputs": {"model_name": "raft_large.pth"}},
            "font": {"class_type": "CreateTextMask", "inputs": {"font": "TTNorms-Black.otf"}},
        }

        refs = list(iter_loader_model_inputs(workflow))

        self.assertEqual(
            refs,
            [
                ModelInputReference("audio", "AudioEncoderLoader", "audio_encoder_name", "audio_encoders", "audio/encoder.safetensors"),
                ModelInputReference("bg", "LoadBackgroundRemovalModel", "bg_removal_name", "background_removal", "birefnet.safetensors"),
                ModelInputReference("da3", "LoadDA3Model", "model_name", "geometry_estimation", "depth_anything_3.safetensors"),
                ModelInputReference("moge", "LoadMoGeModel", "model_name", "geometry_estimation", "moge.safetensors"),
                ModelInputReference("interp", "FrameInterpolationModelLoader", "model_name", "frame_interpolation", "rife.pth"),
                ModelInputReference("face", "LoadMediaPipeFaceLandmarker", "model_name", "detection", "face_landmarker.safetensors"),
                ModelInputReference("patch", "ModelPatchLoader", "name", "model_patches", "qwen_patch.safetensors"),
                ModelInputReference("photo", "PhotoMakerLoader", "photomaker_model_name", "photomaker", "photomaker-v1.bin"),
                ModelInputReference("flow", "OpticalFlowLoader", "model_name", "optical_flow", "raft_large.pth"),
            ],
        )

    def test_remote_model_name_node_keeps_object_info_cheap_and_outputs_wildcard_string(self):
        module = _load_nodes_remote()
        module.local_model_inventory = mock.Mock(side_effect=AssertionError("INPUT_TYPES must not inventory remote or local files"))

        inputs = module.CutleryRemoteModelName.INPUT_TYPES()
        output = module.CutleryRemoteModelName().select_model("checkpoints", "remote.safetensors", "192.0.2.247:8188")

        self.assertEqual(inputs["required"]["model_type"][0], list(module.CANONICAL_MODEL_TYPES))
        self.assertEqual(inputs["required"]["model_name"][0], "STRING")
        self.assertEqual(inputs["required"]["remote_target"][0], "STRING")
        self.assertEqual(module.CutleryRemoteModelName.RETURN_TYPES, ("*",))
        self.assertEqual(output, ("remote.safetensors",))

    def test_remote_group_executor_posts_bundled_inputs_and_decodes_outputs(self):
        module = _load_nodes_remote()
        captured = {}

        def fake_post_json(base_url, path, body, token=None, timeout_seconds=None, on_cancel=None):
            captured.update({"base_url": base_url, "path": path, "body": body, "token": token, "timeout_seconds": timeout_seconds})
            return {"ok": True, "outputs": {"caption": module.encode_value_bundle("remote done")}}

        module._post_remote_json = fake_post_json
        input_ports = [{"name": "prompt", "type": "STRING"}]
        output_ports = [{"name": "caption", "type": "STRING"}]
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            result = module.CutleryRemoteGroupExecutor().run_remote_group(
                remote_base_url="192.0.2.247:8188",
                remote_workflow_json=json.dumps(_compiled_remote_workflow(input_ports, output_ports)),
                input_ports_json=json.dumps(input_ports),
                output_ports_json=json.dumps(output_ports),
                timeout_seconds=7,
                value_1="hello",
            )

        self.assertEqual(captured["base_url"], "http://192.0.2.247:8188")
        self.assertEqual(captured["path"], "/cutlery/remote/group/run")
        self.assertEqual(captured["body"]["values"]["prompt"]["schema"], module.VALUE_BUNDLE_SCHEMA)
        self.assertEqual(captured["token"], "abc123")
        self.assertEqual(captured["timeout_seconds"], 22)
        self.assertTrue(captured["body"]["prompt_id"])
        self.assertEqual(result[0], "remote done")

    def test_remote_group_executor_materializes_inbound_lora_chain_before_dispatch(self):
        module = _load_nodes_remote()
        calls = []

        def fake_post_json(base_url, path, body, token=None, timeout_seconds=None, on_cancel=None):
            calls.append(
                {
                    "base_url": base_url,
                    "path": path,
                    "body": body,
                    "token": token,
                    "timeout_seconds": timeout_seconds,
                }
            )
            if path == "/cutlery/remote/models/resolve":
                model_name = body["model_name"]
                if model_name == "already/remote.safetensors":
                    return {
                        "ok": True,
                        "model_type": "loras",
                        "model_name": r"already\remote.safetensors",
                    }
                return {
                    "ok": False,
                    "model_type": "loras",
                    "model_name": model_name,
                    "error": "missing",
                }
            if path == "/cutlery/remote/group/run":
                return {"ok": True, "outputs": {}}
            raise AssertionError(path)

        module._post_remote_json = fake_post_json
        module.materialize_remote_lora_file = mock.Mock(
            return_value={
                "ok": True,
                "name": "styles/missing.safetensors",
                "sha256": "a6712629445a",
                "size": 5,
            }
        )
        input_ports = [{"name": "lora_chain", "type": "cutlery_lora_chain"}]
        chain = {
            "loras": [
                {
                    "lora_name": "already/remote.safetensors",
                    "strength_model": 0.5,
                    "strength_clip": 0.25,
                },
                {
                    "lora_name": "missing.safetensors",
                    "strength_model": 0.8,
                    "strength_clip": 0.0,
                },
                {
                    "lora_name": "missing.safetensors",
                    "strength_model": 0.0,
                    "strength_clip": 1.1,
                },
                {
                    "lora_name": "zero-strength.safetensors",
                    "strength_model": 0.0,
                    "strength_clip": 0.0,
                },
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            local_path = Path(temp_dir) / "missing.safetensors"
            local_path.write_bytes(b"lora!")
            module.find_local_model_by_filename = mock.Mock(
                return_value={
                    "ok": True,
                    "model_type": "loras",
                    "model_name": "styles/missing.safetensors",
                    "filename": "missing.safetensors",
                    "path": str(local_path),
                }
            )
            with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
                module.CutleryRemoteGroupExecutor().run_remote_group(
                    remote_base_url="127.0.0.1:8189",
                    remote_workflow_json=json.dumps(_compiled_remote_workflow(input_ports, [])),
                    input_ports_json=json.dumps(input_ports),
                    output_ports_json="[]",
                    timeout_seconds=7,
                    value_1=chain,
                )

        module.find_local_model_by_filename.assert_called_once_with(
            "loras",
            "missing.safetensors",
        )
        module.materialize_remote_lora_file.assert_called_once_with(
            "http://127.0.0.1:8189",
            local_path,
            "styles/missing.safetensors",
            auth_headers={"Authorization": "Bearer abc123"},
            timeout_seconds=7,
            check_cancelled=module.throw_if_interrupted,
            max_response_bytes=module.REMOTE_RESPONSE_LIMIT_BYTES,
        )
        resolve_names = [
            call["body"]["model_name"]
            for call in calls
            if call["path"] == "/cutlery/remote/models/resolve"
        ]
        self.assertEqual(
            resolve_names,
            [
                "already/remote.safetensors",
                "missing.safetensors",
            ],
        )
        group_run = next(call for call in calls if call["path"] == "/cutlery/remote/group/run")
        prepared = module.decode_value_bundle(group_run["body"]["values"]["lora_chain"])
        self.assertEqual(
            [entry["lora_name"] for entry in prepared["loras"]],
            [
                r"already\remote.safetensors",
                "styles/missing.safetensors",
                "styles/missing.safetensors",
                "zero-strength.safetensors",
            ],
        )
        self.assertEqual(
            [
                (entry["strength_model"], entry["strength_clip"])
                for entry in prepared["loras"]
            ],
            [(0.5, 0.25), (0.8, 0.0), (0.0, 1.1), (0.0, 0.0)],
        )
        self.assertEqual(chain["loras"][1]["lora_name"], "missing.safetensors")

    def test_remote_group_executor_requires_lora_chain_boundary_feature_before_materializing(self):
        module = _load_nodes_remote()
        module._get_remote_json = mock.Mock(
            return_value={
                "ok": True,
                "protocol_version": 1,
                "serializers": [
                    "primitive",
                    "tensor",
                    "latent",
                    "conditioning",
                    "cutlery_lora_chain",
                ],
                "features": {
                    "remote_groups": True,
                    "remote_node_definitions_v1": True,
                    "prompt_specific_interrupt": True,
                },
            }
        )
        module.find_local_model_by_filename = mock.Mock()
        module.copy_model_file_to_remote = mock.Mock()
        module._post_remote_json = mock.Mock(
            side_effect=AssertionError("remote calls must not run after a failed smoke check")
        )
        input_ports = [{"name": "lora_chain", "type": "cutlery_lora_chain"}]

        with self.assertRaisesRegex(RuntimeError, "remote_lora_chain_boundary_v1"):
            module.CutleryRemoteGroupExecutor().run_remote_group(
                remote_base_url="127.0.0.1:8189",
                remote_workflow_json=json.dumps(_compiled_remote_workflow(input_ports, [])),
                input_ports_json=json.dumps(input_ports),
                output_ports_json="[]",
                value_1={
                    "loras": [
                        {
                            "lora_name": "missing.safetensors",
                            "strength_model": 1.0,
                            "strength_clip": 1.0,
                        }
                    ]
                },
            )

        module.find_local_model_by_filename.assert_not_called()
        module.copy_model_file_to_remote.assert_not_called()
        module._post_remote_json.assert_not_called()

    def test_remote_group_executor_requires_tensor_tree_feature_for_conditioning_adapter(self):
        module = _load_nodes_remote()
        module._post_remote_json = mock.Mock(
            side_effect=AssertionError("remote calls must not run after a failed smoke check")
        )
        workflow = _compiled_remote_workflow([], [])
        workflow["conditioning_adapter"] = {
            "class_type": "WF3ConditioningToBlob",
            "inputs": {},
        }

        with self.assertRaisesRegex(RuntimeError, "cutlery_tensor_tree_v2"):
            module.CutleryRemoteGroupExecutor().run_remote_group(
                remote_base_url="127.0.0.1:8189",
                remote_workflow_json=json.dumps(workflow),
                input_ports_json="[]",
                output_ports_json="[]",
            )

        module._post_remote_json.assert_not_called()

    def test_remote_group_executor_fails_before_dispatch_when_chain_lora_is_missing_locally_and_remotely(self):
        module = _load_nodes_remote()

        def fake_post_json(base_url, path, body, token=None, timeout_seconds=None, on_cancel=None):
            if path == "/cutlery/remote/models/resolve":
                return {
                    "ok": False,
                    "model_type": "loras",
                    "model_name": body["model_name"],
                    "error": "missing",
                }
            raise AssertionError(f"unexpected remote call {path}")

        module._post_remote_json = fake_post_json
        module.find_local_model_by_filename = mock.Mock(
            return_value={
                "ok": False,
                "model_type": "loras",
                "model_name": "missing.safetensors",
                "error": "No local LoRA matched.",
            }
        )
        module.copy_model_file_to_remote = mock.Mock()
        input_ports = [{"name": "lora_chain", "type": "cutlery_lora_chain"}]

        with self.assertRaisesRegex(RuntimeError, "could not find a local 'loras' file"):
            module.CutleryRemoteGroupExecutor().run_remote_group(
                remote_base_url="127.0.0.1:8189",
                remote_workflow_json=json.dumps(_compiled_remote_workflow(input_ports, [])),
                input_ports_json=json.dumps(input_ports),
                output_ports_json="[]",
                value_1={
                    "loras": [
                        {
                            "lora_name": "missing.safetensors",
                            "strength_model": 1.0,
                            "strength_clip": 1.0,
                        }
                    ]
                },
            )

        module.find_local_model_by_filename.assert_called_once_with(
            "loras",
            "missing.safetensors",
        )
        module.copy_model_file_to_remote.assert_not_called()

    def test_remote_group_executor_rejects_malformed_port_contracts_before_remote_calls(self):
        module = _load_nodes_remote()
        module._get_remote_json = mock.Mock(side_effect=AssertionError("smoke must not run"))
        module._post_remote_json = mock.Mock(side_effect=AssertionError("dispatch must not run"))
        too_many = [
            {"name": f"port_{index}", "type": "string"}
            for index in range(module.MAX_REMOTE_GROUP_PORTS + 1)
        ]
        cases = (
            ("{", "valid JSON"),
            ("{}", "JSON array"),
            ('["prompt"]', "must be an object"),
            ('[{"name":"bad name","type":"string"}]', "valid port identifier"),
            ('[{"name":"prompt"}]', "type must be"),
            ('[{"name":"prompt","type":"MODEL"}]', "not a supported"),
            (
                '[{"name":"prompt","type":"string"},{"name":"prompt","type":"string"}]',
                "duplicates port name",
            ),
            (json.dumps(too_many), "maximum is 64"),
        )

        for input_ports_json, expected in cases:
            with self.subTest(input_ports_json=input_ports_json):
                with self.assertRaisesRegex(ValueError, expected):
                    module.CutleryRemoteGroupExecutor().run_remote_group(
                        remote_base_url="127.0.0.1:8189",
                        remote_workflow_json="{}",
                        input_ports_json=input_ports_json,
                        output_ports_json="[]",
                    )

        with self.assertRaisesRegex(ValueError, "output_ports_json must be valid JSON"):
            module.CutleryRemoteGroupExecutor().run_remote_group(
                remote_base_url="127.0.0.1:8189",
                remote_workflow_json="{}",
                input_ports_json="[]",
                output_ports_json="{",
            )

        module._get_remote_json.assert_not_called()
        module._post_remote_json.assert_not_called()

    def test_remote_group_executor_rejects_directionally_unsupported_ports_before_smoke(self):
        module = _load_nodes_remote()
        module._get_remote_json = mock.Mock(side_effect=AssertionError("smoke must not run"))
        module._post_remote_json = mock.Mock(side_effect=AssertionError("dispatch must not run"))
        cases = (
            ([{"name": "clip", "type": "VIDEO"}], [], "Local-to-remote VIDEO"),
            ([], [{"name": "mask", "type": "MASK"}], "Remote-to-local MASK"),
            ([], [{"name": "latent", "type": "LATENT"}], "Remote-to-local LATENT"),
            (
                [],
                [{"name": "conditioning", "type": "CONDITIONING"}],
                "Remote-to-local CONDITIONING",
            ),
            (
                [],
                [{"name": "lora_chain", "type": "cutlery_lora_chain"}],
                "Remote-to-local CUTLERY_LORA_CHAIN",
            ),
        )

        for input_ports, output_ports, expected in cases:
            with self.subTest(input_ports=input_ports, output_ports=output_ports):
                with self.assertRaisesRegex(ValueError, expected):
                    module.CutleryRemoteGroupExecutor().run_remote_group(
                        remote_base_url="127.0.0.1:8189",
                        remote_workflow_json=json.dumps(
                            _compiled_remote_workflow(input_ports, output_ports)
                        ),
                        input_ports_json=json.dumps(input_ports),
                        output_ports_json=json.dumps(output_ports),
                    )

        module._get_remote_json.assert_not_called()
        module._post_remote_json.assert_not_called()

    def test_remote_group_executor_rejects_old_prompt_missing_embedded_boundaries(self):
        module = _load_nodes_remote()
        module._get_remote_json = mock.Mock(side_effect=AssertionError("smoke must not run"))
        module._post_remote_json = mock.Mock(side_effect=AssertionError("dispatch must not run"))

        with self.assertRaisesRegex(ValueError, "CutleryWorkflowInput nodes and exactly one CutleryWorkflowOutput"):
            module.CutleryRemoteGroupExecutor().run_remote_group(
                remote_base_url="127.0.0.1:8189",
                remote_workflow_json=json.dumps(
                    {"1": {"class_type": "NoOp", "inputs": {}}}
                ),
                input_ports_json='[{"name":"prompt","type":"string"}]',
                output_ports_json="[]",
                value_1="hello",
            )

        module._get_remote_json.assert_not_called()
        module._post_remote_json.assert_not_called()

    def test_remote_group_boundary_contract_accepts_one_input_node_per_port(self):
        module = _load_nodes_remote()
        input_ports = [
            {"name": "input_1", "type": "image"},
            {"name": "input_2", "type": "cutlery_lora_chain"},
        ]
        workflow = {
            "input_image": {
                "class_type": "CutleryWorkflowInput",
                "inputs": {"ports_json": json.dumps([input_ports[0]])},
            },
            "input_lora": {
                "class_type": "CutleryWorkflowInput",
                "inputs": {"ports_json": json.dumps([input_ports[1]])},
            },
            "output": {
                "class_type": "CutleryWorkflowOutput",
                "inputs": {"ports_json": "[]"},
            },
        }

        module._validate_remote_workflow_boundary_contract(workflow, input_ports, [])

    def test_remote_group_executor_rejects_spoofed_embedded_boundary_types_before_smoke(self):
        module = _load_nodes_remote()
        module._get_remote_json = mock.Mock(side_effect=AssertionError("smoke must not run"))
        module._post_remote_json = mock.Mock(side_effect=AssertionError("dispatch must not run"))
        cases = (
            (
                [{"name": "clip", "type": "json"}],
                [],
                [{"name": "clip", "type": "video"}],
                [],
                "WorkflowInput ports do not exactly match",
            ),
            (
                [],
                [{"name": "result", "type": "json"}],
                [],
                [{"name": "result", "type": "latent"}],
                "WorkflowOutput ports do not exactly match",
            ),
        )

        for wrapper_inputs, wrapper_outputs, workflow_inputs, workflow_outputs, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    module.CutleryRemoteGroupExecutor().run_remote_group(
                        remote_base_url="127.0.0.1:8189",
                        remote_workflow_json=json.dumps(
                            _compiled_remote_workflow(workflow_inputs, workflow_outputs)
                        ),
                        input_ports_json=json.dumps(wrapper_inputs),
                        output_ports_json=json.dumps(wrapper_outputs),
                    )

        module._get_remote_json.assert_not_called()
        module._post_remote_json.assert_not_called()

    def test_remote_group_executor_interrupts_exact_remote_prompt_when_local_run_is_cancelled(self):
        module = _load_nodes_remote()
        captured = {}

        def fake_post_json(base_url, path, body, token=None, timeout_seconds=None, on_cancel=None):
            captured["prompt_id"] = body["prompt_id"]
            on_cancel()
            raise asyncio.CancelledError()

        module._post_remote_json = fake_post_json
        module._interrupt_remote_prompt_best_effort = mock.Mock()

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            with self.assertLogs(module.LOGGER, level="INFO") as logs:
                with self.assertRaises(asyncio.CancelledError):
                    module.CutleryRemoteGroupExecutor().run_remote_group(
                        remote_base_url="127.0.0.1:8189",
                        remote_workflow_json=json.dumps({"1": {"class_type": "NoOp", "inputs": {}}}),
                        input_ports_json="[]",
                        output_ports_json="[]",
                        timeout_seconds=7,
                    )

        module._interrupt_remote_prompt_best_effort.assert_called_once_with(
            "http://127.0.0.1:8189",
            captured["prompt_id"],
            token="abc123",
        )
        self.assertIn("Remote group cancelled target=http://127.0.0.1:8189", "\n".join(logs.output))

    def test_remote_group_executor_logs_failed_job_duration(self):
        module = _load_nodes_remote()
        module._post_remote_json = lambda *_args, **_kwargs: {"ok": False, "error": "peer failed"}

        with self.assertLogs(module.LOGGER, level="WARNING") as logs:
            with self.assertRaisesRegex(RuntimeError, "peer failed"):
                module.CutleryRemoteGroupExecutor().run_remote_group(
                    remote_base_url="127.0.0.1:8189",
                    remote_workflow_json=json.dumps({"1": {"class_type": "NoOp", "inputs": {}}}),
                    input_ports_json="[]",
                    output_ports_json="[]",
                )

        self.assertIn("Remote group failed target=http://127.0.0.1:8189", "\n".join(logs.output))

    def test_remote_group_executor_logs_target_and_capabilities_smoke_before_dispatch(self):
        module = _load_nodes_remote()
        calls = []

        def fake_get_json(base_url, path, token=None, timeout_seconds=None):
            calls.append({"method": "GET", "base_url": base_url, "path": path, "token": token, "timeout_seconds": timeout_seconds})
            return {
                "ok": True,
                "protocol_version": 1,
                "serializers": ["primitive", "tensor", "latent", "conditioning", "cutlery_lora_chain"],
                "features": {
                    "remote_groups": True,
                    "remote_node_definitions_v1": True,
                    "prompt_specific_interrupt": True,
                },
            }

        def fake_post_json(base_url, path, body, token=None, timeout_seconds=None, on_cancel=None):
            calls.append({"method": "POST", "base_url": base_url, "path": path, "token": token, "timeout_seconds": timeout_seconds})
            return {"ok": True, "outputs": {}}

        module._get_remote_json = fake_get_json
        module._post_remote_json = fake_post_json

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            with self.assertLogs(module.LOGGER, level="INFO") as logs:
                module.CutleryRemoteGroupExecutor().run_remote_group(
                    remote_base_url="127.0.0.1:8189",
                    remote_workflow_json=json.dumps({"2": {"class_type": "RemoteWork", "inputs": {}}}),
                    input_ports_json="[]",
                    output_ports_json="[]",
                    timeout_seconds=7,
                )

        self.assertEqual(calls[0]["method"], "GET")
        self.assertEqual(calls[0]["base_url"], "http://127.0.0.1:8189")
        self.assertEqual(calls[0]["path"], "/cutlery/remote/capabilities")
        self.assertEqual(calls[0]["token"], "abc123")
        self.assertLessEqual(calls[0]["timeout_seconds"], 7)
        self.assertEqual(calls[1]["method"], "POST")
        self.assertEqual(calls[1]["path"], "/cutlery/remote/group/run")
        log_output = "\n".join(logs.output)
        self.assertIn("Remote group detected target=http://127.0.0.1:8189", log_output)
        self.assertIn("Remote smoke result target=http://127.0.0.1:8189", log_output)
        self.assertIn('"remote_groups":true', log_output)
        self.assertIn("Remote group completed target=http://127.0.0.1:8189", log_output)
        self.assertRegex(log_output, r"duration_seconds=\d+\.\d{2}")

    def test_remote_group_preflight_rejects_machine_specific_combo_value_missing_remotely(self):
        module = _load_nodes_remote()
        signature = {
            "input_is_list": False,
            "inputs": {"required": [{"name": "sampler_name", "kind": "combo", "type": "COMBO"}], "optional": [], "hidden": []},
            "outputs": [],
        }
        module.build_node_definitions_payload = mock.Mock(
            return_value={
                "definitions": {
                    "SamplerNode": {
                        "missing": False,
                        "ok": True,
                        "signature": signature,
                    }
                }
            }
        )
        module._post_remote_json = mock.Mock(
            return_value={
                "ok": True,
                "nodes": {
                    "SamplerNode": {
                        "available": True,
                        "compatible": True,
                        "signature": signature,
                        "inputs": {
                            "required": {
                                "sampler_name": {
                                    "kind": "combo",
                                    "options": ["euler"],
                                    "materializable": False,
                                }
                            },
                            "optional": {},
                            "hidden": {},
                        },
                    }
                },
            }
        )

        with self.assertRaisesRegex(RuntimeError, "does not offer"):
            module._real_preflight_remote_workflow(
                "127.0.0.1:8189",
                {"1": {"class_type": "SamplerNode", "inputs": {"sampler_name": "remote-missing"}}},
                token="abc123",
                timeout_seconds=7,
            )

    def test_remote_group_preflight_does_not_treat_boolean_as_numeric_combo_option(self):
        module = _load_nodes_remote()
        signature = {
            "input_is_list": False,
            "inputs": {
                "required": [{"name": "choice", "kind": "combo", "type": "COMBO"}],
                "optional": [],
                "hidden": [],
            },
            "outputs": [],
        }
        module.build_node_definitions_payload = mock.Mock(
            return_value={
                "definitions": {
                    "TypedComboNode": {
                        "missing": False,
                        "ok": True,
                        "signature": signature,
                    }
                }
            }
        )
        remote_definition = {
            "ok": True,
            "nodes": {
                "TypedComboNode": {
                    "available": True,
                    "compatible": True,
                    "signature": signature,
                    "inputs": {
                        "required": {
                            "choice": {
                                "kind": "combo",
                                "options": [1],
                                "materializable": False,
                            }
                        },
                        "optional": {},
                        "hidden": {},
                    },
                }
            },
        }
        module._post_remote_json = mock.Mock(return_value=remote_definition)

        with self.assertRaisesRegex(RuntimeError, "does not offer"):
            module._real_preflight_remote_workflow(
                "127.0.0.1:8189",
                {"1": {"class_type": "TypedComboNode", "inputs": {"choice": True}}},
                token="abc123",
                timeout_seconds=7,
            )

        remote_definition["nodes"]["TypedComboNode"]["inputs"]["required"]["choice"]["options"] = [1.0]
        result = module._real_preflight_remote_workflow(
            "127.0.0.1:8189",
            {"1": {"class_type": "TypedComboNode", "inputs": {"choice": 1}}},
            token="abc123",
            timeout_seconds=7,
        )
        self.assertTrue(result["ok"])

    def test_remote_group_preflight_allows_missing_materializable_model_choice(self):
        module = _load_nodes_remote()
        signature = {
            "input_is_list": False,
            "inputs": {"required": [{"name": "clip_name", "kind": "combo", "type": "COMBO"}], "optional": [], "hidden": []},
            "outputs": [],
        }
        module.build_node_definitions_payload = mock.Mock(
            return_value={
                "definitions": {
                    "CLIPLoader": {
                        "missing": False,
                        "ok": True,
                        "signature": signature,
                    }
                }
            }
        )
        module._post_remote_json = mock.Mock(
            return_value={
                "ok": True,
                "nodes": {
                    "CLIPLoader": {
                        "available": True,
                        "compatible": True,
                        "signature": signature,
                        "inputs": {
                            "required": {
                                "clip_name": {
                                    "kind": "combo",
                                    "options": ["remote.safetensors"],
                                    "materializable": True,
                                }
                            },
                            "optional": {},
                            "hidden": {},
                        },
                    }
                },
            }
        )

        result = module._real_preflight_remote_workflow(
            "127.0.0.1:8189",
            {"1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "local-only.safetensors"}}},
            token="abc123",
            timeout_seconds=7,
        )

        self.assertTrue(result["ok"])

    def test_remote_group_preflight_rejects_matching_signatures_with_definition_errors(self):
        module = _load_nodes_remote()
        error_signature = {
            "input_is_list": False,
            "inputs": {
                "required": [{"name": "registry", "kind": "error", "type": "ERROR"}],
                "optional": [],
                "hidden": [],
            },
            "outputs": [],
        }
        workflow = {"1": {"class_type": "BrokenRegistryNode", "inputs": {}}}
        cases = (
            (True, False, "remote target"),
            (False, True, "locally"),
        )

        for local_ok, remote_compatible, expected in cases:
            with self.subTest(local_ok=local_ok, remote_compatible=remote_compatible):
                module.build_node_definitions_payload = mock.Mock(
                    return_value={
                        "definitions": {
                            "BrokenRegistryNode": {
                                "missing": False,
                                "ok": local_ok,
                                "signature": error_signature,
                            }
                        }
                    }
                )
                module._post_remote_json = mock.Mock(
                    return_value={
                        "ok": True,
                        "nodes": {
                            "BrokenRegistryNode": {
                                "available": True,
                                "compatible": remote_compatible,
                                "signature": error_signature,
                                "inputs": {
                                    "required": {
                                        "registry": {
                                            "kind": "error",
                                            "error": "registry inspection failed",
                                        }
                                    },
                                    "optional": {},
                                    "hidden": {},
                                },
                            }
                        },
                    }
                )

                with self.assertRaisesRegex(RuntimeError, expected):
                    module._real_preflight_remote_workflow(
                        "127.0.0.1:8189",
                        workflow,
                        token="abc123",
                        timeout_seconds=7,
                    )

    def test_remote_group_preflight_requires_successful_remote_definition_response(self):
        module = _load_nodes_remote()
        module._post_remote_json = mock.Mock(
            return_value={"ok": False, "nodes": {}}
        )
        module.build_node_definitions_payload = mock.Mock(
            side_effect=AssertionError("local definitions must not run after an invalid remote response")
        )

        with self.assertRaisesRegex(RuntimeError, "invalid response"):
            module._real_preflight_remote_workflow(
                "127.0.0.1:8189",
                {"1": {"class_type": "NoOp", "inputs": {}}},
                token="abc123",
                timeout_seconds=7,
            )

        module.build_node_definitions_payload.assert_not_called()

    def test_remote_group_executor_copies_missing_remote_model_by_filename_before_run(self):
        module = _load_nodes_remote()
        calls = []

        def fake_post_json(base_url, path, body, token=None, timeout_seconds=None, on_cancel=None):
            calls.append({"base_url": base_url, "path": path, "body": body, "token": token, "timeout_seconds": timeout_seconds})
            if path == "/cutlery/remote/models/resolve":
                if body["model_name"] == "sd3/clip_l.safetensors":
                    return {"ok": True, "model_type": "text_encoders", "model_name": "sd3/clip_l.safetensors"}
                return {"ok": False, "model_type": "text_encoders", "model_name": body["model_name"], "error": "missing"}
            if path == "/cutlery/remote/group/run":
                return {"ok": True, "outputs": {}}
            raise AssertionError(path)

        module._post_remote_json = fake_post_json
        module.find_local_model_by_filename = mock.Mock(
            return_value={
                "ok": True,
                "model_type": "text_encoders",
                "model_name": "sd3/clip_l.safetensors",
                "filename": "clip_l.safetensors",
                "path": "C:\\Models\\text_encoders\\sd3\\clip_l.safetensors",
            }
        )
        module.copy_model_file_to_remote = mock.Mock(return_value={"ok": True, "remote_model_name": "sd3/clip_l.safetensors"})

        remote_workflow = {
            "2": {
                "class_type": "CutleryRemoteModelName",
                "inputs": {
                    "model_type": "text_encoders",
                    "model_name": "clip_l.safetensors",
                    "remote_target": "192.0.2.247:8188",
                },
            }
        }

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module.CutleryRemoteGroupExecutor().run_remote_group(
                remote_base_url="192.0.2.247:8188",
                remote_workflow_json=json.dumps(remote_workflow),
                input_ports_json="[]",
                output_ports_json="[]",
                timeout_seconds=7,
            )

        module.find_local_model_by_filename.assert_called_once_with("text_encoders", "clip_l.safetensors")
        module.copy_model_file_to_remote.assert_called_once_with(
            "C:\\Models\\text_encoders\\sd3\\clip_l.safetensors",
            "text_encoders",
            "sd3/clip_l.safetensors",
            remote_host="renderhost",
            remote_root="D:/ComfyUI/models",
        )
        group_run = [call for call in calls if call["path"] == "/cutlery/remote/group/run"][0]
        self.assertEqual(group_run["body"]["workflow"]["2"]["inputs"]["model_name"], "sd3/clip_l.safetensors")
        self.assertEqual(group_run["body"]["workflow"]["2"]["inputs"]["model_type"], "text_encoders")

    def test_remote_group_executor_copies_missing_stock_loader_models_by_filename_before_run(self):
        module = _load_nodes_remote()
        calls = []

        def fake_post_json(base_url, path, body, token=None, timeout_seconds=None, on_cancel=None):
            calls.append({"base_url": base_url, "path": path, "body": body, "token": token, "timeout_seconds": timeout_seconds})
            if path == "/cutlery/remote/models/resolve":
                if body["model_name"].startswith("ltxv/"):
                    return {"ok": True, "model_type": "text_encoders", "model_name": body["model_name"]}
                return {"ok": False, "model_type": "text_encoders", "model_name": body["model_name"], "error": "missing"}
            if path == "/cutlery/remote/group/run":
                return {"ok": True, "outputs": {}}
            raise AssertionError(path)

        def fake_find(model_type, model_name):
            return {
                "ok": True,
                "model_type": model_type,
                "model_name": f"ltxv/{model_name}",
                "filename": model_name,
                "path": f"C:\\Models\\text_encoders\\ltxv\\{model_name}",
            }

        module._post_remote_json = fake_post_json
        module.find_local_model_by_filename = mock.Mock(side_effect=fake_find)
        module.copy_model_file_to_remote = mock.Mock(return_value={"ok": True})
        remote_workflow = {
            "9": {
                "class_type": "DualCLIPLoaderGGUF",
                "inputs": {
                    "clip_name1": "gemma-3-12b-it-ablit-norms-beta.safetensors",
                    "clip_name2": "ltxv-2-3-22b-text-encoder.safetensors",
                    "type": "ltxv",
                },
            }
        }

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module.CutleryRemoteGroupExecutor().run_remote_group(
                remote_base_url="127.0.0.1:8189",
                remote_workflow_json=json.dumps(remote_workflow),
                input_ports_json="[]",
                output_ports_json="[]",
                timeout_seconds=7,
            )

        self.assertEqual(
            module.find_local_model_by_filename.mock_calls,
            [
                mock.call("text_encoders", "gemma-3-12b-it-ablit-norms-beta.safetensors"),
                mock.call("text_encoders", "ltxv-2-3-22b-text-encoder.safetensors"),
            ],
        )
        self.assertEqual(module.copy_model_file_to_remote.call_count, 2)
        group_run = [call for call in calls if call["path"] == "/cutlery/remote/group/run"][0]
        loader_inputs = group_run["body"]["workflow"]["9"]["inputs"]
        self.assertEqual(loader_inputs["clip_name1"], "ltxv/gemma-3-12b-it-ablit-norms-beta.safetensors")
        self.assertEqual(loader_inputs["clip_name2"], "ltxv/ltxv-2-3-22b-text-encoder.safetensors")
        self.assertEqual(loader_inputs["type"], "ltxv")

    def test_remote_group_executor_copies_batch_3_loader_models_by_filename_before_run(self):
        module = _load_nodes_remote()
        calls = []

        def fake_post_json(base_url, path, body, token=None, timeout_seconds=None, on_cancel=None):
            calls.append({"base_url": base_url, "path": path, "body": body, "token": token, "timeout_seconds": timeout_seconds})
            if path == "/cutlery/remote/models/resolve":
                if "/" in body["model_name"]:
                    return {"ok": True, "model_type": body["model_type"], "model_name": body["model_name"]}
                return {"ok": False, "model_type": body["model_type"], "model_name": body["model_name"], "error": "missing"}
            if path == "/cutlery/remote/group/run":
                return {"ok": True, "outputs": {}}
            raise AssertionError(path)

        def fake_find(model_type, model_name):
            return {
                "ok": True,
                "model_type": model_type,
                "model_name": f"batch3/{model_name}",
                "filename": model_name,
                "path": f"C:\\Models\\{model_type}\\batch3\\{model_name}",
            }

        module._post_remote_json = fake_post_json
        module.find_local_model_by_filename = mock.Mock(side_effect=fake_find)
        module.copy_model_file_to_remote = mock.Mock(return_value={"ok": True})
        remote_workflow = {
            "10": {
                "class_type": "LTXVGemmaCLIPModelLoader",
                "inputs": {
                    "gemma_path": "gemma-3.safetensors",
                    "ltxv_path": "ltxv-model.safetensors",
                },
            },
            "12": {"class_type": "LTXVEditAnythingModuleLoader", "inputs": {"module_name": "editanything_module.safetensors"}},
            "13": {"class_type": "LowVRAMLatentUpscaleModelLoader", "inputs": {"model_name": "latent-upscale.safetensors"}},
        }

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module.CutleryRemoteGroupExecutor().run_remote_group(
                remote_base_url="127.0.0.1:8189",
                remote_workflow_json=json.dumps(remote_workflow),
                input_ports_json="[]",
                output_ports_json="[]",
                timeout_seconds=7,
            )

        self.assertEqual(
            module.find_local_model_by_filename.mock_calls,
            [
                mock.call("text_encoders", "gemma-3.safetensors"),
                mock.call("checkpoints", "ltxv-model.safetensors"),
                mock.call("loras", "editanything_module.safetensors"),
                mock.call("latent_upscale_models", "latent-upscale.safetensors"),
            ],
        )
        self.assertEqual(module.copy_model_file_to_remote.call_count, 4)
        group_run = [call for call in calls if call["path"] == "/cutlery/remote/group/run"][0]
        self.assertEqual(group_run["body"]["workflow"]["10"]["inputs"]["gemma_path"], "batch3/gemma-3.safetensors")
        self.assertEqual(group_run["body"]["workflow"]["10"]["inputs"]["ltxv_path"], "batch3/ltxv-model.safetensors")
        self.assertEqual(group_run["body"]["workflow"]["12"]["inputs"]["module_name"], "batch3/editanything_module.safetensors")
        self.assertEqual(group_run["body"]["workflow"]["13"]["inputs"]["model_name"], "batch3/latent-upscale.safetensors")

    def test_remote_group_executor_copies_batch_6_loader_models_by_filename_before_run(self):
        module = _load_nodes_remote()
        calls = []

        def fake_post_json(base_url, path, body, token=None, timeout_seconds=None, on_cancel=None):
            calls.append({"base_url": base_url, "path": path, "body": body, "token": token, "timeout_seconds": timeout_seconds})
            if path == "/cutlery/remote/models/resolve":
                if "/" in body["model_name"]:
                    return {"ok": True, "model_type": body["model_type"], "model_name": body["model_name"]}
                return {"ok": False, "model_type": body["model_type"], "model_name": body["model_name"], "error": "missing"}
            if path == "/cutlery/remote/group/run":
                return {"ok": True, "outputs": {}}
            raise AssertionError(path)

        def fake_find(model_type, model_name):
            return {
                "ok": True,
                "model_type": model_type,
                "model_name": f"batch6/{model_name}",
                "filename": model_name,
                "path": f"C:\\Models\\{model_type}\\batch6\\{model_name}",
            }

        module._post_remote_json = fake_post_json
        module.find_local_model_by_filename = mock.Mock(side_effect=fake_find)
        module.copy_model_file_to_remote = mock.Mock(return_value={"ok": True})
        remote_workflow = {
            "20": {"class_type": "AudioEncoderLoader", "inputs": {"audio_encoder_name": "encoder.safetensors"}},
            "21": {"class_type": "LoadDA3Model", "inputs": {"model_name": "depth_anything_3.safetensors", "weight_dtype": "default"}},
            "22": {"class_type": "ModelPatchLoader", "inputs": {"name": "qwen_patch.safetensors"}},
            "23": {"class_type": "OpticalFlowLoader", "inputs": {"model_name": "raft_large.pth"}},
        }

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module.CutleryRemoteGroupExecutor().run_remote_group(
                remote_base_url="127.0.0.1:8189",
                remote_workflow_json=json.dumps(remote_workflow),
                input_ports_json="[]",
                output_ports_json="[]",
                timeout_seconds=7,
            )

        self.assertEqual(
            module.find_local_model_by_filename.mock_calls,
            [
                mock.call("audio_encoders", "encoder.safetensors"),
                mock.call("geometry_estimation", "depth_anything_3.safetensors"),
                mock.call("model_patches", "qwen_patch.safetensors"),
                mock.call("optical_flow", "raft_large.pth"),
            ],
        )
        self.assertEqual(module.copy_model_file_to_remote.call_count, 4)
        group_run = [call for call in calls if call["path"] == "/cutlery/remote/group/run"][0]
        self.assertEqual(group_run["body"]["workflow"]["20"]["inputs"]["audio_encoder_name"], "batch6/encoder.safetensors")
        self.assertEqual(group_run["body"]["workflow"]["21"]["inputs"]["model_name"], "batch6/depth_anything_3.safetensors")
        self.assertEqual(group_run["body"]["workflow"]["22"]["inputs"]["name"], "batch6/qwen_patch.safetensors")
        self.assertEqual(group_run["body"]["workflow"]["23"]["inputs"]["model_name"], "batch6/raft_large.pth")

    def test_remote_group_executor_materializes_wan_loader_fallback_model_types(self):
        module = _load_nodes_remote()
        calls = []

        def fake_post_json(base_url, path, body, token=None, timeout_seconds=None, on_cancel=None):
            calls.append({"base_url": base_url, "path": path, "body": body, "token": token, "timeout_seconds": timeout_seconds})
            if path == "/cutlery/remote/models/resolve":
                if body["model_type"] == "mmaudio" and str(body["model_name"]).startswith("ovi/"):
                    return {"ok": True, "model_type": "mmaudio", "model_name": body["model_name"]}
                return {"ok": False, "model_type": body["model_type"], "model_name": body["model_name"], "error": "missing"}
            if path == "/cutlery/remote/group/run":
                return {"ok": True, "outputs": {}}
            raise AssertionError(path)

        def fake_find(model_type, model_name):
            if model_type == "mmaudio":
                return {
                    "ok": True,
                    "model_type": "mmaudio",
                    "model_name": f"ovi/{model_name}",
                    "filename": model_name,
                    "path": f"C:\\Models\\mmaudio\\ovi\\{model_name}",
                }
            return {"ok": False, "model_type": model_type, "model_name": model_name, "error": "missing"}

        module._post_remote_json = fake_post_json
        module.find_local_model_by_filename = mock.Mock(side_effect=fake_find)
        module.copy_model_file_to_remote = mock.Mock(return_value={"ok": True})
        remote_workflow = {
            "42": {
                "class_type": "OviMMAudioVAELoader",
                "inputs": {
                    "vae": "mmaudio_vae_16k_bf16.safetensors",
                    "vocoder": "mmaudio_vocoder_bigvgan_best_netG_bf16.safetensors",
                    "precision": "bf16",
                },
            }
        }

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module.CutleryRemoteGroupExecutor().run_remote_group(
                remote_base_url="127.0.0.1:8189",
                remote_workflow_json=json.dumps(remote_workflow),
                input_ports_json="[]",
                output_ports_json="[]",
                timeout_seconds=7,
            )

        self.assertEqual(
            module.find_local_model_by_filename.mock_calls,
            [
                mock.call("vae", "mmaudio_vae_16k_bf16.safetensors"),
                mock.call("mmaudio", "mmaudio_vae_16k_bf16.safetensors"),
                mock.call("vae", "mmaudio_vocoder_bigvgan_best_netG_bf16.safetensors"),
                mock.call("mmaudio", "mmaudio_vocoder_bigvgan_best_netG_bf16.safetensors"),
            ],
        )
        group_run = [call for call in calls if call["path"] == "/cutlery/remote/group/run"][0]
        self.assertEqual(group_run["body"]["workflow"]["42"]["inputs"]["vae"], "ovi/mmaudio_vae_16k_bf16.safetensors")
        self.assertEqual(group_run["body"]["workflow"]["42"]["inputs"]["vocoder"], "ovi/mmaudio_vocoder_bigvgan_best_netG_bf16.safetensors")

    def test_remote_group_executor_ignores_non_loader_filename_like_strings(self):
        module = _load_nodes_remote()

        def fake_post_json(base_url, path, body, token=None, timeout_seconds=None, on_cancel=None):
            if path == "/cutlery/remote/group/run":
                return {"ok": True, "outputs": {}}
            raise AssertionError(path)

        module._post_remote_json = fake_post_json
        module.find_local_model_by_filename = mock.Mock()
        module.copy_model_file_to_remote = mock.Mock()
        remote_workflow = {
            "3": {
                "class_type": "PrimitiveString",
                "inputs": {"text": "not-a-model.safetensors"},
            }
        }

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module.CutleryRemoteGroupExecutor().run_remote_group(
                remote_base_url="127.0.0.1:8189",
                remote_workflow_json=json.dumps(remote_workflow),
                input_ports_json="[]",
                output_ports_json="[]",
                timeout_seconds=7,
            )

        module.find_local_model_by_filename.assert_not_called()
        module.copy_model_file_to_remote.assert_not_called()

    def test_remote_group_executor_materializes_ipadapter_loader_file(self):
        module = _load_nodes_remote()
        calls = []

        def fake_post_json(base_url, path, body, token=None, timeout_seconds=None, on_cancel=None):
            calls.append({"base_url": base_url, "path": path, "body": body, "token": token, "timeout_seconds": timeout_seconds})
            if path == "/cutlery/remote/models/resolve":
                if body["model_name"] == "subdir/ip-adapter-plus.safetensors":
                    return {"ok": True, "model_type": "ipadapter", "model_name": body["model_name"]}
                return {"ok": False, "model_type": "ipadapter", "model_name": body["model_name"], "error": "missing"}
            if path == "/cutlery/remote/group/run":
                return {"ok": True, "outputs": {}}
            raise AssertionError(path)

        module._post_remote_json = fake_post_json
        module.find_local_model_by_filename = mock.Mock(
            return_value={
                "ok": True,
                "model_type": "ipadapter",
                "model_name": "subdir/ip-adapter-plus.safetensors",
                "filename": "ip-adapter-plus.safetensors",
                "path": "C:\\Models\\ipadapter\\subdir\\ip-adapter-plus.safetensors",
            }
        )
        module.copy_model_file_to_remote = mock.Mock(return_value={"ok": True})
        remote_workflow = {
            "12": {
                "class_type": "IPAdapterModelLoader",
                "inputs": {"ipadapter_file": "ip-adapter-plus.safetensors"},
            }
        }

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module.CutleryRemoteGroupExecutor().run_remote_group(
                remote_base_url="127.0.0.1:8189",
                remote_workflow_json=json.dumps(remote_workflow),
                input_ports_json="[]",
                output_ports_json="[]",
                timeout_seconds=7,
            )

        module.find_local_model_by_filename.assert_called_once_with("ipadapter", "ip-adapter-plus.safetensors")
        module.copy_model_file_to_remote.assert_called_once_with(
            "C:\\Models\\ipadapter\\subdir\\ip-adapter-plus.safetensors",
            "ipadapter",
            "subdir/ip-adapter-plus.safetensors",
            remote_host="renderhost",
            remote_root="D:/ComfyUI/models",
        )
        group_run = [call for call in calls if call["path"] == "/cutlery/remote/group/run"][0]
        self.assertEqual(group_run["body"]["workflow"]["12"]["inputs"]["ipadapter_file"], "subdir/ip-adapter-plus.safetensors")

    def test_remote_group_executor_does_not_copy_when_remote_resolve_is_unauthorized(self):
        module = _load_nodes_remote()

        def fake_post_json(base_url, path, body, token=None, timeout_seconds=None, on_cancel=None):
            if path == "/cutlery/remote/models/resolve":
                raise module.RemoteHttpError("Unauthorized.", status_code=401)
            raise AssertionError(path)

        module._post_remote_json = fake_post_json
        module.find_local_model_by_filename = mock.Mock()
        module.copy_model_file_to_remote = mock.Mock()
        remote_workflow = {
            "2": {
                "class_type": "CutleryRemoteModelName",
                "inputs": {"model_type": "checkpoints", "model_name": "model.safetensors"},
            }
        }

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            with self.assertRaisesRegex(module.RemoteHttpError, "Unauthorized"):
                module.CutleryRemoteGroupExecutor().run_remote_group(
                    remote_base_url="192.0.2.247:8188",
                    remote_workflow_json=json.dumps(remote_workflow),
                    input_ports_json="[]",
                    output_ports_json="[]",
                    timeout_seconds=7,
                )

        module.find_local_model_by_filename.assert_not_called()
        module.copy_model_file_to_remote.assert_not_called()

    def test_remote_group_executor_materializes_video_media_bundle_locally(self):
        module = _load_nodes_remote()
        with tempfile.TemporaryDirectory() as temp_dir:
            media_root = Path(temp_dir)
            captured_paths = []

            def fake_media_root():
                media_root.mkdir(parents=True, exist_ok=True)
                return media_root

            class _InputImpl:
                @staticmethod
                def VideoFromFile(path):
                    captured_paths.append(Path(path))
                    return f"video:{Path(path).name}"

            module._remote_media_root = fake_media_root
            sys.modules["comfy_api.latest"] = types.SimpleNamespace(InputImpl=_InputImpl)
            value = {
                "__cutlery_remote_media__": True,
                "media_type": "video",
                "filename": "remote.mp4",
                "content_type": "video/mp4",
                "data": b"video bytes",
            }

            result = module._decode_output_value(module.encode_value_bundle(value))

            digest = module.sha256_bytes(b"video bytes")
            self.assertEqual(result, f"video:{digest}.mp4")
            self.assertEqual(captured_paths[0].read_bytes(), b"video bytes")

    def test_remote_media_materialization_is_content_addressed_and_collision_safe(self):
        module = _load_nodes_remote()
        with tempfile.TemporaryDirectory() as temp_dir:
            media_root = Path(temp_dir)
            module._remote_media_root = lambda: media_root.resolve()

            first = module._write_remote_media_file(
                {"filename": "same.mp4", "data": b"first-data"}
            )
            second = module._write_remote_media_file(
                {"filename": "same.mp4", "data": b"other-data"}
            )

            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), b"first-data")
            self.assertEqual(second.read_bytes(), b"other-data")
            self.assertEqual(first.stem, module.sha256_bytes(b"first-data"))
            self.assertEqual(second.stem, module.sha256_bytes(b"other-data"))
            self.assertEqual(list(media_root.glob(".cutlery-remote-media-*.part")), [])

    def test_remote_media_cache_evicts_stale_unowned_files_but_preserves_active_prompt_files(self):
        module = _load_nodes_remote()
        with tempfile.TemporaryDirectory() as temp_dir:
            media_root = Path(temp_dir).resolve()
            module._remote_media_root = lambda: media_root
            stale = media_root / "stale.mp4"
            active = media_root / "active.mp4"
            staging = media_root / ".cutlery-remote-media-stale.part"
            stale.write_bytes(b"stale")
            active.write_bytes(b"active")
            staging.write_bytes(b"partial")
            old_media_time = module.time.time() - module.REMOTE_MEDIA_CACHE_MAX_AGE_SECONDS - 1
            old_staging_time = module.time.time() - module.REMOTE_MEDIA_TEMP_MAX_AGE_SECONDS - 1
            os.utime(stale, (old_media_time, old_media_time))
            os.utime(active, (old_media_time, old_media_time))
            os.utime(staging, (old_staging_time, old_staging_time))
            module._retain_remote_media_for_prompt("active-prompt", active)

            module._evict_remote_media_cache()

            self.assertFalse(stale.exists())
            self.assertFalse(staging.exists())
            self.assertTrue(active.exists())
            module._release_remote_media_prompt("active-prompt")
            self.assertFalse(active.exists())

    def test_remote_video_media_is_owned_by_execution_context_until_prompt_end(self):
        module = _load_nodes_remote()
        self.assertNotIn(
            "PROMPT_ID",
            module.CutleryRemoteGroupExecutor.INPUT_TYPES().get("hidden", {}).values(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            media_root = Path(temp_dir)
            captured_paths = []
            module._remote_media_root = lambda: media_root.resolve()

            class _InputImpl:
                @staticmethod
                def VideoFromFile(path):
                    captured_paths.append(Path(path))
                    return f"video:{Path(path).name}"

            sys.modules["comfy_api.latest"] = types.SimpleNamespace(InputImpl=_InputImpl)
            value = {
                "__cutlery_remote_media__": True,
                "media_type": "video",
                "filename": "remote.mp4",
                "content_type": "video/mp4",
                "data": b"lazy video",
            }
            output_ports = [{"name": "clip", "type": "video"}]
            module._post_remote_json = mock.Mock(
                return_value={
                    "ok": True,
                    "outputs": {"clip": module.encode_value_bundle(value)},
                }
            )

            execution_context = {"value": None}
            comfy_execution = types.ModuleType("comfy_execution")
            comfy_execution_utils = types.ModuleType("comfy_execution.utils")
            comfy_execution_utils.get_executing_context = lambda: execution_context["value"]

            with mock.patch.dict(
                sys.modules,
                {
                    "comfy_execution": comfy_execution,
                    "comfy_execution.utils": comfy_execution_utils,
                },
            ):
                execution_context["value"] = types.SimpleNamespace(
                    prompt_id="local-prompt-1",
                    node_id="remote-wrapper",
                    list_index=0,
                )
                result = module.CutleryRemoteGroupExecutor().run_remote_group(
                    remote_base_url="127.0.0.1:8189",
                    remote_workflow_json=json.dumps(
                        _compiled_remote_workflow([], output_ports)
                    ),
                    input_ports_json="[]",
                    output_ports_json=json.dumps(output_ports),
                )
                execution_context["value"] = types.SimpleNamespace(
                    prompt_id="local-prompt-2",
                    node_id="remote-wrapper",
                    list_index=0,
                )
                second_result = module.CutleryRemoteGroupExecutor().run_remote_group(
                    remote_base_url="127.0.0.1:8189",
                    remote_workflow_json=json.dumps(
                        _compiled_remote_workflow([], output_ports)
                    ),
                    input_ports_json="[]",
                    output_ports_json=json.dumps(output_ports),
                )
                execution_context["value"] = None

            self.assertTrue(result[0].startswith("video:"))
            self.assertEqual(second_result[0], result[0])
            self.assertEqual(module._current_execution_prompt_id(), "")
            self.assertTrue(captured_paths[0].is_file())
            self.assertEqual(captured_paths[0], captured_paths[1])
            module._RemoteMediaLifecycleProvider().on_prompt_end("local-prompt-1")
            self.assertTrue(captured_paths[0].exists())
            module._RemoteMediaLifecycleProvider().on_prompt_end("local-prompt-2")
            self.assertFalse(captured_paths[0].exists())

    def test_remote_audio_media_batches_are_loaded_and_merged_then_files_are_deleted(self):
        module = _load_nodes_remote()
        with tempfile.TemporaryDirectory() as temp_dir:
            media_root = Path(temp_dir)
            loaded_paths = []
            module._remote_media_root = lambda: media_root.resolve()

            def load(path):
                loaded_paths.append(Path(path))
                marker = float(len(loaded_paths))
                import torch

                return torch.full((1, 3), marker), 48000

            comfy_extras = types.ModuleType("comfy_extras")
            nodes_audio = types.ModuleType("comfy_extras.nodes_audio")
            nodes_audio.load = load
            descriptors = [
                {
                    "__cutlery_remote_media__": True,
                    "media_type": "audio",
                    "filename": f"batch-{index}.wav",
                    "content_type": "audio/wav",
                    "data": data,
                }
                for index, data in enumerate((b"audio-one", b"audio-two"))
            ]

            with mock.patch.dict(
                sys.modules,
                {
                    "comfy_extras": comfy_extras,
                    "comfy_extras.nodes_audio": nodes_audio,
                },
            ):
                result = module._decode_output_value(module.encode_value_bundle(descriptors))

            self.assertEqual(result["sample_rate"], 48000)
            self.assertEqual(tuple(result["waveform"].shape), (2, 1, 3))
            self.assertEqual(len(loaded_paths), 2)
            self.assertTrue(all(not path.exists() for path in loaded_paths))


    def test_prepared_workflow_rewrites_only_declared_model_inputs(self):
        module = _load_nodes_remote()
        workflow = {
            "loader": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": "clip.safetensors", "type": "stable_diffusion"},
            }
        }

        rewritten = module._rewrite_prepared_model_inputs(
            workflow,
            {("loader", "clip_name"): "text/subdir/clip.safetensors"},
        )

        self.assertEqual(rewritten["loader"]["inputs"]["clip_name"], "text/subdir/clip.safetensors")
        self.assertEqual(rewritten["loader"]["inputs"]["type"], "stable_diffusion")
        self.assertEqual(workflow["loader"]["inputs"]["clip_name"], "clip.safetensors")

    def test_preparation_rewrites_models_absent_from_preload_subset_only_in_full_workflow(self):
        module = _load_nodes_remote()
        workflow = {
            "56": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": "clip.safetensors", "type": "krea2"},
            }
        }
        prepared = {
            "manifest": {"identity": "prepared-manifest"},
            "rewrites": {("56", "clip_name"): "nested/clip.safetensors"},
        }

        with (
            mock.patch.object(module, "REMOTE_EARLY_MODEL_PRELOAD_ENABLED", False),
            mock.patch.object(module, "_log_remote_group_start_and_smoke", return_value={}),
            mock.patch.object(module, "validate_remote_group_capabilities"),
            mock.patch.object(module, "_prepare_remote_models_blocking", return_value=prepared),
        ):
            result = asyncio.run(
                module.CutleryRemoteGroupPreparation().prepare(
                    remote_base_url="127.0.0.1:8189",
                    remote_workflow_json=json.dumps(workflow),
                    model_refs_json="[]",
                    preparation_manifest_json=json.dumps({"identity": "compiled-manifest"}),
                    preload_workflow_json="{}",
                )
            )

        handle = result[0]
        self.assertEqual(handle["prepared_workflow"]["56"]["inputs"]["clip_name"], "nested/clip.safetensors")
        self.assertFalse(handle["preloaded"])

    def test_executor_returns_pending_task_inside_comfy_event_loop(self):
        module = _load_nodes_remote()
        executor = module.CutleryRemoteGroupExecutor()
        release = asyncio.Event()

        async def remote_execution(*_args, **_kwargs):
            await release.wait()
            return ("done",)

        executor._run_remote_group_async = remote_execution

        async def exercise():
            task = executor.run_remote_group("", "{}", "[]", "[]")
            self.assertIsInstance(task, asyncio.Task)
            self.assertFalse(task.done())
            await asyncio.sleep(0)
            self.assertFalse(task.done())
            release.set()
            self.assertEqual(await task, ("done",))

        asyncio.run(exercise())

    def test_stream_error_message_extracts_peer_execution_failure(self):
        module = _load_nodes_remote()
        payload = {
            "ok": False,
            "prompt_id": "peer-prompt",
            "status": {
                "status_str": "error",
                "messages": [
                    [
                        "execution_error",
                        {
                            "exception_type": "TypeError",
                            "exception_message": "Loader failed clearly.\n",
                        },
                    ]
                ],
            },
        }

        self.assertEqual(
            module._stream_error_message(payload),
            "TypeError: Loader failed clearly.",
        )

    def test_remote_progress_socket_filters_post_compile_helpers(self):
        module = _load_nodes_remote()

        class Socket:
            def __init__(self):
                self.messages = []

            async def send_json(self, message):
                self.messages.append(message)

        websocket = Socket()
        progress_socket = module._RemoteProgressSocket(websocket, "peer-prompt", {"known"})

        asyncio.run(
            progress_socket.send_json(
                {
                    "type": "progress_state",
                    "data": {
                        "prompt_id": "peer-prompt",
                        "nodes": {
                            "known": {"value": 1, "max": 2},
                            "generated-save": {"value": 1, "max": 1},
                        },
                    },
                }
            )
        )

        self.assertEqual(
            websocket.messages,
            [
                {
                    "type": "progress",
                    "data": {
                        "prompt_id": "peer-prompt",
                        "nodes": {"known": {"value": 1, "max": 2}},
                    },
                }
            ],
        )

    def test_stream_message_limit_covers_base64_media_total(self):
        module = _load_nodes_remote()

        self.assertGreater(
            module.MAX_REMOTE_STREAM_MESSAGE_BYTES,
            module.MAX_REMOTE_MEDIA_TOTAL_BYTES * 4 // 3,
        )


if __name__ == "__main__":
    unittest.main()
