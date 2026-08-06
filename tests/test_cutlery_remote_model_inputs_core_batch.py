import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class RemoteModelInputsCoreBatchTests(unittest.TestCase):
    def test_core_and_standard_extra_loader_inputs_are_registered(self):
        from cutlery_remote.model_inputs import iter_loader_model_inputs

        workflow = {
            "1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": "svd.safetensors"}},
            "2": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": "ltxv-audio.safetensors"}},
            "3": {
                "class_type": "LTXAVTextEncoderLoader",
                "inputs": {"text_encoder": "t5xxl_fp16.safetensors", "ckpt_name": "ltxv.safetensors"},
            },
            "4": {"class_type": "GLIGENLoader", "inputs": {"gligen_name": "gligen.safetensors"}},
            "5": {"class_type": "CreateHookLora", "inputs": {"lora_name": "style.safetensors"}},
            "6": {"class_type": "CreateHookModelAsLora", "inputs": {"ckpt_name": "base.safetensors"}},
        }

        refs = [(ref.class_type, ref.input_name, ref.model_type, ref.model_name) for ref in iter_loader_model_inputs(workflow)]

        self.assertEqual(
            refs,
            [
                ("ImageOnlyCheckpointLoader", "ckpt_name", "checkpoints", "svd.safetensors"),
                ("LTXVAudioVAELoader", "ckpt_name", "checkpoints", "ltxv-audio.safetensors"),
                ("LTXAVTextEncoderLoader", "text_encoder", "text_encoders", "t5xxl_fp16.safetensors"),
                ("LTXAVTextEncoderLoader", "ckpt_name", "checkpoints", "ltxv.safetensors"),
                ("GLIGENLoader", "gligen_name", "gligen", "gligen.safetensors"),
                ("CreateHookLora", "lora_name", "loras", "style.safetensors"),
                ("CreateHookModelAsLora", "ckpt_name", "checkpoints", "base.safetensors"),
            ],
        )

    def test_gligen_is_supported_as_remote_inventory_category(self):
        from cutlery_remote.inventory import CANONICAL_MODEL_TYPES, normalize_model_type

        self.assertIn("gligen", CANONICAL_MODEL_TYPES)
        self.assertEqual(normalize_model_type("gligen"), "gligen")


if __name__ == "__main__":
    unittest.main()
