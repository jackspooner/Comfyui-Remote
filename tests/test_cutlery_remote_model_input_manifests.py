import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutlery_remote import model_inputs  # noqa: E402


class ModelInputManifestTests(unittest.TestCase):
    def setUp(self):
        model_inputs.reset_manifest_cache()
        self._root = tempfile.TemporaryDirectory()
        self.addCleanup(self._root.cleanup)
        self.addCleanup(model_inputs.reset_manifest_cache)
        self.root = Path(self._root.name)
        patcher = mock.patch.object(model_inputs, "CUSTOM_NODES_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _write(self, package, payload):
        package_dir = self.root / package
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / model_inputs.MANIFEST_FILENAME).write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_manifest_adds_loader_and_overrides_builtin_inputs(self):
        self._write(
            "Cutlery-Nodes",
            {
                "version": 1,
                "inputs": {
                    "CutleryLoraLoaderClipOnly": {"lora_name": "loras"},
                    "UNETLoader": {"unet_name": "checkpoints"},
                },
                "peer_managed": ["CutleryAlphaFaceLoader"],
            },
        )

        registry = model_inputs.loader_model_inputs()

        self.assertEqual(registry["CutleryLoraLoaderClipOnly"], {"lora_name": "loras"})
        self.assertEqual(registry["UNETLoader"], {"unet_name": "checkpoints"})
        self.assertEqual(model_inputs.peer_managed_class_types(), frozenset({"CutleryAlphaFaceLoader"}))
        self.assertEqual(model_inputs.model_input_manifest().sources, ("Cutlery-Nodes",))

    def test_ordered_categories_become_primary_and_alternates(self):
        self._write(
            "Cutlery-Nodes",
            {"version": 1, "inputs": {"Loader": {"ckpt_name": ["checkpoints", "text_encoders", "clip"]}}},
        )

        spec = model_inputs.loader_model_inputs()["Loader"]["ckpt_name"]

        self.assertEqual(spec.primary, "checkpoints")
        self.assertEqual(spec.alternates, ("text_encoders", "clip"))

    def test_by_extension_rule_selects_category(self):
        self._write(
            "Cutlery-Nodes",
            {
                "version": 1,
                "inputs": {
                    "Loader": {
                        "clip_name": {"by_extension": {".gguf": "clip_gguf"}, "default": "text_encoders"}
                    }
                },
            },
        )

        rule = model_inputs.loader_model_inputs()["Loader"]["clip_name"]

        self.assertEqual(rule("model.gguf"), "clip_gguf")
        self.assertEqual(rule("MODEL.GGUF"), "clip_gguf")
        self.assertEqual(rule("model.safetensors"), "text_encoders")

    def test_invalid_entry_is_skipped_without_losing_valid_ones(self):
        self._write(
            "Cutlery-Nodes",
            {
                "version": 1,
                "inputs": {
                    "Good": {"lora_name": "loras"},
                    "BadPath": {"model": "/home/jack/models/x.safetensors"},
                    "BadTraversal": {"model": "../escape"},
                    "BadCategory": {"model": "Not A Category"},
                    "BadRule": {"model": {"by_extension": {"gguf": "clip_gguf"}, "default": "text_encoders"}},
                },
            },
        )

        registry = model_inputs.loader_model_inputs()

        self.assertIn("Good", registry)
        for rejected in ("BadPath", "BadTraversal", "BadCategory", "BadRule"):
            self.assertNotIn(rejected, registry)

    def test_malformed_manifests_are_ignored(self):
        bad_json = self.root / "Broken"
        bad_json.mkdir()
        (bad_json / model_inputs.MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
        self._write("WrongVersion", {"version": 2, "inputs": {"Loader": {"x": "loras"}}})
        self._write("UnknownKey", {"version": 1, "extra": True})
        self._write("NotObject", [1, 2, 3])

        manifest = model_inputs.model_input_manifest()

        self.assertEqual(manifest.inputs, {})
        self.assertEqual(manifest.sources, ())

    def test_only_the_manifest_filename_is_discovered(self):
        self._write("Cutlery-Nodes", {"version": 1, "inputs": {"Loader": {"lora_name": "loras"}}})
        decoy = self.root / "Other"
        decoy.mkdir()
        (decoy / "model_inputs.json").write_text(
            json.dumps({"version": 1, "inputs": {"Decoy": {"x": "loras"}}}), encoding="utf-8"
        )

        registry = model_inputs.loader_model_inputs()

        self.assertIn("Loader", registry)
        self.assertNotIn("Decoy", registry)

    def test_cache_refreshes_after_reset(self):
        self._write("Cutlery-Nodes", {"version": 1, "inputs": {"Loader": {"lora_name": "loras"}}})
        self.assertIn("Loader", model_inputs.loader_model_inputs())

        self._write("Cutlery-MOSS", {"version": 1, "peer_managed": ["MossTTSModelLoader"]})

        self.assertIn("Loader", model_inputs.loader_model_inputs())
        self.assertNotIn("MossTTSModelLoader", model_inputs.peer_managed_class_types())

        model_inputs.reset_manifest_cache()

        self.assertIn("MossTTSModelLoader", model_inputs.peer_managed_class_types())

    def test_iter_loader_model_inputs_reads_manifest_classes(self):
        self._write(
            "Cutlery-Nodes",
            {"version": 1, "inputs": {"CutleryLoraLoaderClipOnly": {"lora_name": "loras"}}},
        )
        workflow = {
            "7": {"class_type": "CutleryLoraLoaderClipOnly", "inputs": {"lora_name": "style.safetensors"}},
            "8": {"class_type": "CutleryLoraLoaderClipOnly", "inputs": {"lora_name": "none"}},
        }

        refs = list(model_inputs.iter_loader_model_inputs(workflow))

        self.assertEqual([(ref.node_id, ref.model_name) for ref in refs], [("7", "style.safetensors")])
        self.assertEqual(refs[0].model_type, "loras")

    def test_iter_peer_managed_nodes_reports_matching_nodes(self):
        self._write("Cutlery-MOSS", {"version": 1, "peer_managed": ["MossTTSModelLoader"]})
        workflow = {
            "3": {"class_type": "MossTTSModelLoader", "inputs": {}},
            "4": {"class_type": "SaveImage", "inputs": {}},
        }

        found = list(model_inputs.iter_peer_managed_nodes(workflow))

        self.assertEqual(found, [("3", "MossTTSModelLoader")])

if __name__ == "__main__":
    unittest.main()
