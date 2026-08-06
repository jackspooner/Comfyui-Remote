import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class RemoteInventoryTests(unittest.TestCase):
    def test_inventory_excludes_incomplete_model_transfer_staging_files(self):
        folder_paths = types.SimpleNamespace(
            get_filename_list=lambda key: {
                "checkpoints": [
                    "ready.safetensors",
                    ".cutlery-upload-deadbeef.part",
                    "subdir/.cutlery-upload-cancelled.part",
                ]
            }.get(key, []),
        )

        with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
            from cutlery_remote.inventory import list_model_names, resolve_model_name

            names = list_model_names("checkpoints")
            staged = resolve_model_name("checkpoints", ".cutlery-upload-deadbeef.part")

        self.assertEqual(names, ["ready.safetensors"])
        self.assertFalse(staged["ok"])

    def test_inventory_lists_canonical_model_categories_without_hashing(self):
        calls = []
        folder_paths = types.SimpleNamespace(
            get_filename_list=lambda key: calls.append(key)
            or {
                "checkpoints": ["zeta.safetensors", "alpha.safetensors"],
                "text_encoders": ["clip-l.safetensors"],
                "clip_gguf": ["qwen.gguf"],
                "unet_gguf": ["wan.gguf"],
                "vae_approx": ["taehv.safetensors"],
                "latent_upscale_models": ["latent-upscale.safetensors"],
                "geometry_estimation": ["moge.safetensors"],
                "audio_encoders": ["audio.safetensors"],
                "wav2vec2": ["wav2vec.safetensors"],
                "nlf": ["nlf_l_multi_0.3.2.torchscript"],
                "mmaudio": ["mmaudio_vae.safetensors"],
                "ipadapter": ["ip-adapter-plus.safetensors"],
                "loras": ["style.safetensors"],
            }.get(key, []),
            get_full_path_or_raise=mock.Mock(side_effect=AssertionError("inventory must not hash or resolve paths")),
        )

        with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
            from cutlery_remote.inventory import CANONICAL_MODEL_TYPES, local_model_inventory

            inventory = local_model_inventory(include_hashes=False)

        self.assertIn("checkpoints", CANONICAL_MODEL_TYPES)
        self.assertIn("diffusion_models", CANONICAL_MODEL_TYPES)
        self.assertIn("clip_gguf", CANONICAL_MODEL_TYPES)
        self.assertIn("unet_gguf", CANONICAL_MODEL_TYPES)
        self.assertIn("vae_approx", CANONICAL_MODEL_TYPES)
        self.assertIn("latent_upscale_models", CANONICAL_MODEL_TYPES)
        self.assertIn("geometry_estimation", CANONICAL_MODEL_TYPES)
        self.assertIn("audio_encoders", CANONICAL_MODEL_TYPES)
        self.assertIn("wav2vec2", CANONICAL_MODEL_TYPES)
        self.assertIn("nlf", CANONICAL_MODEL_TYPES)
        self.assertIn("mmaudio", CANONICAL_MODEL_TYPES)
        self.assertIn("ipadapter", CANONICAL_MODEL_TYPES)
        self.assertEqual(inventory["checkpoints"], ["alpha.safetensors", "zeta.safetensors"])
        self.assertEqual(inventory["text_encoders"], ["clip-l.safetensors"])
        self.assertEqual(inventory["clip_gguf"], ["qwen.gguf"])
        self.assertEqual(inventory["unet_gguf"], ["wan.gguf"])
        self.assertEqual(inventory["vae_approx"], ["taehv.safetensors"])
        self.assertEqual(inventory["latent_upscale_models"], ["latent-upscale.safetensors"])
        self.assertEqual(inventory["geometry_estimation"], ["moge.safetensors"])
        self.assertEqual(inventory["audio_encoders"], ["audio.safetensors"])
        self.assertEqual(inventory["wav2vec2"], ["wav2vec.safetensors"])
        self.assertEqual(inventory["nlf"], ["nlf_l_multi_0.3.2.torchscript"])
        self.assertEqual(inventory["mmaudio"], ["mmaudio_vae.safetensors"])
        self.assertEqual(inventory["ipadapter"], ["ip-adapter-plus.safetensors"])
        self.assertNotIn("hash", inventory["records"]["checkpoints"][0])
        self.assertIn("checkpoints", calls)
        self.assertIn("loras", calls)

    def test_gguf_and_approx_model_aliases_normalize_to_canonical_folder_keys(self):
        folder_paths = types.SimpleNamespace(
            get_filename_list=lambda key: {
                "clip_gguf": ["text/qwen.gguf"],
                "unet_gguf": ["wan.gguf"],
                "vae_approx": ["taehv.safetensors"],
            }.get(key, []),
        )

        with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
            from cutlery_remote.inventory import list_model_names, normalize_model_type, resolve_model_name

            self.assertEqual(normalize_model_type("clip-gguf"), "clip_gguf")
            self.assertEqual(normalize_model_type("unet gguf"), "unet_gguf")
            self.assertEqual(normalize_model_type("approx_vae"), "vae_approx")
            self.assertEqual(list_model_names("clip_gguf"), ["text/qwen.gguf"])
            self.assertTrue(resolve_model_name("unet_gguf", "wan.gguf")["ok"])

    def test_gguf_model_transfer_uses_loader_searchable_destination_folders(self):
        from cutlery_remote.model_transfer import remote_model_destination_folder

        self.assertEqual(
            remote_model_destination_folder("clip_gguf", "text/qwen.gguf", root="D:/ComfyUI/models"),
            "D:/ComfyUI/models/text_encoders/text",
        )
        self.assertEqual(
            remote_model_destination_folder("unet_gguf", "wan.gguf", root="D:/ComfyUI/models"),
            "D:/ComfyUI/models/diffusion_models",
        )

    def test_latent_upscale_model_alias_normalizes_to_canonical_folder_key(self):
        folder_paths = types.SimpleNamespace(
            get_filename_list=lambda key: {"latent_upscale_models": ["upscale.safetensors"]}.get(key, []),
        )

        with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
            from cutlery_remote.inventory import resolve_model_name

            result = resolve_model_name("latent_upscale_model", "upscale.safetensors")

        self.assertTrue(result["ok"])
        self.assertEqual(result["model_type"], "latent_upscale_models")

    def test_batch_6_model_type_aliases_normalize_to_canonical_folder_keys(self):
        folder_paths = types.SimpleNamespace(
            get_filename_list=lambda key: {
                "audio_encoders": ["audio.safetensors"],
                "background_removal": ["birefnet.safetensors"],
                "geometry_estimation": ["moge.safetensors"],
                "frame_interpolation": ["rife.pth"],
                "detection": ["face_landmarker.safetensors"],
                "model_patches": ["qwen_patch.safetensors"],
                "photomaker": ["photomaker-v1.bin"],
                "optical_flow": ["raft_large.pth"],
            }.get(key, []),
        )

        with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
            from cutlery_remote.inventory import resolve_model_name

            cases = [
                ("audio_encoder", "audio.safetensors", "audio_encoders"),
                ("background-removal", "birefnet.safetensors", "background_removal"),
                ("geometry", "moge.safetensors", "geometry_estimation"),
                ("frame_interpolation_model", "rife.pth", "frame_interpolation"),
                ("face_detection", "face_landmarker.safetensors", "detection"),
                ("model_patch", "qwen_patch.safetensors", "model_patches"),
                ("photomaker", "photomaker-v1.bin", "photomaker"),
                ("optical-flow", "raft_large.pth", "optical_flow"),
            ]
            results = [resolve_model_name(alias, name) for alias, name, _canonical in cases]

        self.assertEqual([result["model_type"] for result in results], [canonical for _alias, _name, canonical in cases])
        self.assertTrue(all(result["ok"] for result in results))

    def test_ipadapter_model_alias_normalizes_to_canonical_folder_key(self):
        folder_paths = types.SimpleNamespace(
            get_filename_list=lambda key: {"ipadapter": ["subdir/ip-adapter-plus.safetensors"]}.get(key, []),
        )

        with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
            from cutlery_remote.inventory import list_model_names, normalize_model_type, resolve_model_name

            self.assertEqual(normalize_model_type("ip_adapter"), "ipadapter")
            self.assertEqual(list_model_names("ipadapter"), ["subdir/ip-adapter-plus.safetensors"])
            self.assertTrue(resolve_model_name("ipadapter", "subdir/ip-adapter-plus.safetensors")["ok"])

    def test_wan_wrapper_model_aliases_normalize_to_canonical_folder_keys(self):
        folder_paths = types.SimpleNamespace(
            get_filename_list=lambda key: {
                "wav2vec2": ["wav2vec.safetensors"],
                "nlf": ["nlf_l_multi_0.3.2.torchscript"],
                "mmaudio": ["mmaudio_vae.safetensors"],
                "audio_encoders": ["whisper.safetensors"],
            }.get(key, []),
        )

        with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
            from cutlery_remote.inventory import list_model_names, normalize_model_type, resolve_model_name

            self.assertEqual(normalize_model_type("wav2vec"), "wav2vec2")
            self.assertEqual(normalize_model_type("nlf"), "nlf")
            self.assertEqual(normalize_model_type("mm_audio"), "mmaudio")
            self.assertEqual(normalize_model_type("audio_encoder"), "audio_encoders")
            self.assertEqual(list_model_names("mmaudio"), ["mmaudio_vae.safetensors"])
            self.assertTrue(resolve_model_name("wav2vec2", "wav2vec.safetensors")["ok"])

    def test_resolve_model_name_reports_presence_without_copying(self):
        folder_paths = types.SimpleNamespace(
            get_filename_list=lambda key: {"checkpoints": ["remote-only.safetensors"]}.get(key, []),
        )

        with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
            from cutlery_remote.inventory import resolve_model_name

            present = resolve_model_name("checkpoints", "remote-only.safetensors")
            missing = resolve_model_name("checkpoints", "missing.safetensors")

        self.assertTrue(present["ok"])
        self.assertEqual(present["model_type"], "checkpoints")
        self.assertEqual(present["model_name"], "remote-only.safetensors")
        self.assertFalse(missing["ok"])

    def test_resolve_model_name_matches_portable_separators_and_returns_registry_name(self):
        folder_paths = types.SimpleNamespace(
            get_filename_list=lambda key: {
                "loras": [r"styles\cinematic.safetensors"]
            }.get(key, []),
        )

        with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
            from cutlery_remote.inventory import resolve_model_name

            result = resolve_model_name(
                "loras",
                "styles/cinematic.safetensors",
            )

        self.assertTrue(result["ok"])
        self.assertEqual(
            result["model_name"],
            r"styles\cinematic.safetensors",
        )

    def test_resolve_model_name_uses_host_filesystem_case_rules(self):
        folder_paths = types.SimpleNamespace(
            get_filename_list=lambda key: {
                "loras": [r"Krea2\tentacles-krea2-v0.4.safetensors"]
            }.get(key, []),
        )

        with (
            mock.patch.dict(sys.modules, {"folder_paths": folder_paths}),
            mock.patch("cutlery_remote.inventory.os.path.normcase", side_effect=lambda value: value.casefold()),
        ):
            from cutlery_remote.inventory import resolve_model_name

            result = resolve_model_name(
                "loras",
                r"krea2\tentacles-krea2-v0.4.safetensors",
            )

        self.assertTrue(result["ok"])
        self.assertEqual(
            result["model_name"],
            r"Krea2\tentacles-krea2-v0.4.safetensors",
        )

    def test_find_local_model_by_filename_uses_model_category_and_returns_full_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            local_file = Path(temp_dir) / "clip_l.safetensors"
            local_file.write_bytes(b"clip")

            def full_path(folder_key, model_name):
                self.assertEqual(folder_key, "text_encoders")
                self.assertEqual(model_name, "sd3/clip_l.safetensors")
                return str(local_file)

            folder_paths = types.SimpleNamespace(
                get_filename_list=lambda key: {"text_encoders": ["sd3/clip_l.safetensors"]}.get(key, []),
                get_full_path_or_raise=full_path,
            )

            with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
                from cutlery_remote.inventory import find_local_model_by_filename

                result = find_local_model_by_filename("text_encoder", "clip_l.safetensors")

        self.assertTrue(result["ok"])
        self.assertEqual(result["model_type"], "text_encoders")
        self.assertEqual(result["model_name"], "sd3/clip_l.safetensors")
        self.assertEqual(result["filename"], "clip_l.safetensors")
        self.assertEqual(result["path"], str(local_file))

    def test_find_local_model_by_filename_supports_ipadapter_category(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            local_file = Path(temp_dir) / "ip-adapter-plus.safetensors"
            local_file.write_bytes(b"ipadapter")

            folder_paths = types.SimpleNamespace(
                get_filename_list=lambda key: {"ipadapter": ["subdir/ip-adapter-plus.safetensors"]}.get(key, []),
                get_full_path_or_raise=lambda folder_key, model_name: str(local_file),
            )

            with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
                from cutlery_remote.inventory import find_local_model_by_filename

                result = find_local_model_by_filename("ip_adapter", "ip-adapter-plus.safetensors")

        self.assertTrue(result["ok"])
        self.assertEqual(result["model_type"], "ipadapter")
        self.assertEqual(result["model_name"], "subdir/ip-adapter-plus.safetensors")

    def test_find_local_model_by_filename_reports_ambiguous_basename_matches(self):
        folder_paths = types.SimpleNamespace(
            get_filename_list=lambda key: {"loras": ["a/style.safetensors", "b/style.safetensors"]}.get(key, []),
            get_full_path_or_raise=mock.Mock(side_effect=AssertionError("ambiguous matches must not resolve paths")),
        )

        with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
            from cutlery_remote.inventory import find_local_model_by_filename

            result = find_local_model_by_filename("loras", "style.safetensors")

        self.assertFalse(result["ok"])
        self.assertEqual(result["matches"], ["a/style.safetensors", "b/style.safetensors"])
        self.assertIn("Multiple local", result["error"])

    def test_find_local_model_prefers_separator_normalized_exact_match_over_duplicate_basename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            local_file = Path(temp_dir) / "style.safetensors"
            local_file.write_bytes(b"style")

            def full_path(folder_key, model_name):
                self.assertEqual(folder_key, "loras")
                self.assertEqual(model_name, r"a\style.safetensors")
                return str(local_file)

            folder_paths = types.SimpleNamespace(
                get_filename_list=lambda key: {
                    "loras": [
                        r"a\style.safetensors",
                        r"b\style.safetensors",
                    ]
                }.get(key, []),
                get_full_path_or_raise=full_path,
            )

            with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
                from cutlery_remote.inventory import find_local_model_by_filename

                result = find_local_model_by_filename(
                    "loras",
                    "a/style.safetensors",
                )

        self.assertTrue(result["ok"])
        self.assertEqual(result["model_name"], r"a\style.safetensors")
        self.assertEqual(result["path"], str(local_file))


if __name__ == "__main__":
    unittest.main()
