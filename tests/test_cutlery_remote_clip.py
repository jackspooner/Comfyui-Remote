from __future__ import annotations

import asyncio
import hashlib
import json
import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


class _Routes:
    def __init__(self):
        self.handlers = {}

    def get(self, path):
        def register(fn):
            self.handlers[("GET", path)] = fn
            return fn

        return register

    def post(self, path):
        def register(fn):
            self.handlers[("POST", path)] = fn
            return fn

        return register


def _json_response(payload, status=200, **_kwargs):
    return {"payload": payload, "status": status}


def _load_remote_clip_module(routes: _Routes | None = None, folder_paths=None):
    routes = routes or _Routes()
    folder_paths = folder_paths or types.SimpleNamespace(
        base_path=str(COMFY_ROOT),
        get_filename_list=lambda key: {
            "text_encoders": ["remote-t5.safetensors", "remote-clip-l.safetensors"],
            "loras": ["style-a.safetensors", "characters/style-b.safetensors"],
        }.get(key, []),
        get_full_path_or_raise=lambda key, name: f"/models/{key}/{name}",
        get_folder_paths=lambda key: [f"/models/{key}"],
    )
    server = types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes)))
    aiohttp = types.SimpleNamespace(web=types.SimpleNamespace(json_response=_json_response))

    module_name = "cutlery_nodes_remote_clip_test"
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "nodes_remote_clip.py")
    assert spec is not None
    assert spec.loader is not None

    with mock.patch.dict(
        sys.modules,
        {
            "folder_paths": folder_paths,
            "server": server,
            "aiohttp": aiohttp,
            "aiohttp.web": aiohttp.web,
        },
    ):
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        module._remote_clip_server_disabled_response = lambda: None

    return module


class _FakeClip:
    def __init__(self):
        self.tokenized = []

    def clone(self):
        return self

    def tokenize(self, text):
        self.tokenized.append(text)
        return {"tokens": text}

    def encode_from_tokens_scheduled(self, tokens):
        return [
            [
                torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
                {"pooled_output": torch.ones((1, 3), dtype=torch.float32), "tokens": tokens["tokens"]},
            ]
        ]


class _FakeQwenClip:
    def __init__(self):
        self.tokenize_calls = []

    def clone(self):
        return self

    def tokenize(self, text, *, images=None, llama_template=None):
        self.tokenize_calls.append(
            {
                "text": text,
                "images": list(images or []),
                "llama_template": llama_template,
            }
        )
        return {"tokens": text, "image_count": len(images or [])}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros((1, 2, 3), dtype=torch.float32), {"tokens": tokens["tokens"], "image_count": tokens["image_count"]}]]


class _FakeVAE:
    def __init__(self):
        self.encoded = []

    def encode(self, image):
        self.encoded.append(image)
        return torch.full((1, 4, 4, 4), len(self.encoded), dtype=torch.float32)


class CutleryRemoteClipTests(unittest.TestCase):
    def test_disabled_server_gate_runs_before_authorization_and_body_read(self):
        routes = _Routes()
        module = _load_remote_clip_module(routes)
        disabled = {"payload": {"ok": False, "code": "remote_clip_server_disabled"}, "status": 403}
        module._remote_clip_server_disabled_response = lambda: disabled
        module._authorized = mock.Mock(side_effect=AssertionError("authorization must not run while disabled"))

        handler = routes.handlers[("POST", "/cutlery/remote/clip/text-encode")]
        result = asyncio.run(handler(types.SimpleNamespace()))

        self.assertIs(result, disabled)
        module._authorized.assert_not_called()

    def test_remote_clip_endpoint_queues_a_normal_comfyui_job_and_returns_its_conditioning(self):
        routes = _Routes()
        module = _load_remote_clip_module(routes)
        captured = {}

        class Queue:
            def put(self, item):
                captured["item"] = item
                ui = module.CutleryRemoteClipTextEncodeJob().execute(item[2]["1"]["inputs"]["payload_json"])["ui"]
                captured["history"] = {
                    item[1]: {
                        "status": {"status_str": "success", "completed": True, "messages": []},
                        "outputs": {"1": ui},
                    }
                }

            def get_history(self, prompt_id):
                return captured.get("history", {}).get(prompt_id) and captured["history"]

            def delete_queue_item(self, _predicate):
                return False

            def interrupt_if_running(self, _prompt_id):
                return False

        queue = Queue()
        module.PromptServer.instance.prompt_queue = queue
        module.PromptServer.instance.number = 4
        conditioning = {"schema": "cutlery.value_bundle.v1", "manifest": {"type": "list", "items": []}, "tensors": []}

        execution = types.ModuleType("execution")

        async def validate_prompt(prompt_id, prompt, targets):
            captured["validated"] = (prompt_id, prompt, targets)
            return True, None, ["1"], {}

        execution.validate_prompt = validate_prompt
        execution.SENSITIVE_EXTRA_DATA_KEYS = ()

        with (
            mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=True),
            mock.patch.dict(sys.modules, {"execution": execution}),
            mock.patch.object(module, "encode_remote_clip_text", return_value={"ok": True, "conditioning": conditioning}),
        ):
            response = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/clip/text-encode")](
                    types.SimpleNamespace(
                        headers={"Authorization": "Bearer abc123"},
                        json=lambda: {"prompt": "queued remotely", "text_encoder": "remote.safetensors"},
                    )
                )
            )

        self.assertEqual(response["status"], 200)
        self.assertEqual(response["payload"]["conditioning"], conditioning)
        self.assertEqual(captured["item"][0], 4.0)
        self.assertEqual(captured["item"][2]["1"]["class_type"], "CutleryRemoteClipTextEncodeJob")
        self.assertEqual(captured["item"][2]["1"]["_meta"]["title"], "Remote CLIP Text Encode")
        self.assertEqual(json.loads(captured["item"][2]["1"]["inputs"]["payload_json"]), {"prompt": "queued remotely", "text_encoder": "remote.safetensors"})
        self.assertEqual(captured["validated"][1], captured["item"][2])
        self.assertEqual(captured["validated"][2], None)

    def test_remote_clip_job_failure_is_returned_from_comfyui_history(self):
        module = _load_remote_clip_module()

        class Queue:
            def put(self, item):
                self.prompt_id = item[1]

            def get_history(self, prompt_id):
                return {
                    prompt_id: {
                        "status": {"status_str": "error", "completed": False, "messages": ["execution failed"]},
                        "outputs": {},
                    }
                }

            def delete_queue_item(self, _predicate):
                return False

            def interrupt_if_running(self, _prompt_id):
                return False

        module.PromptServer.instance.prompt_queue = Queue()
        module.PromptServer.instance.number = 0
        execution = types.ModuleType("execution")
        execution.SENSITIVE_EXTRA_DATA_KEYS = ()

        async def validate_prompt(_prompt_id, _prompt, _targets):
            return True, None, ["1"], {}

        execution.validate_prompt = validate_prompt
        with mock.patch.dict(sys.modules, {"execution": execution}):
            payload, status = asyncio.run(module._submit_remote_clip_job("single", {"prompt": "fails"}))

        self.assertEqual(status, 500)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"], "execution failed")

    def test_remote_clip_job_timeout_cancels_only_its_queued_prompt(self):
        module = _load_remote_clip_module()

        class Queue:
            def __init__(self):
                self.item = None
                self.deleted = None
                self.interrupted = None

            def put(self, item):
                self.item = item

            def get_history(self, prompt_id):
                return {}

            def delete_queue_item(self, predicate):
                self.deleted = predicate
                return True

            def interrupt_if_running(self, prompt_id):
                self.interrupted = prompt_id
                return False

        queue = Queue()
        module.PromptServer.instance.prompt_queue = queue
        module.PromptServer.instance.number = 0
        execution = types.ModuleType("execution")
        execution.SENSITIVE_EXTRA_DATA_KEYS = ()

        async def validate_prompt(_prompt_id, _prompt, _targets):
            return True, None, ["1"], {}

        execution.validate_prompt = validate_prompt
        with (
            mock.patch.dict(sys.modules, {"execution": execution}),
            mock.patch.object(module, "_remote_clip_encode_timeout", return_value=0),
        ):
            payload, status = asyncio.run(module._submit_remote_clip_job("single", {"prompt": "times out"}))

        self.assertEqual(status, 504)
        self.assertFalse(payload["ok"])
        self.assertEqual(queue.interrupted, queue.item[1])
        self.assertTrue(queue.deleted(queue.item))

    def test_streamed_materialization_enforces_exact_file_limit(self):
        module = _load_remote_clip_module()

        class Stream:
            def __init__(self, chunks):
                self.chunks = chunks

            async def iter_chunked(self, _size):
                for chunk in self.chunks:
                    yield chunk

        with tempfile.TemporaryDirectory() as temp_dir:
            module._primary_clip_root = lambda: Path(temp_dir)
            with self.assertRaises(module.RemoteClipUploadTooLarge):
                asyncio.run(module._materialize_clip_upload("encoder.safetensors", Stream([b"ab", b"c"]), limit_bytes=2))
            self.assertFalse(any(path.name.startswith(".upload-") for path in Path(temp_dir).rglob("*")))

    def test_streamed_lora_materialization_accepts_exact_limit_and_rejects_one_byte_over(self):
        module = _load_remote_clip_module()

        class Stream:
            def __init__(self, chunks):
                self.chunks = chunks

            async def iter_chunked(self, _size):
                for chunk in self.chunks:
                    yield chunk

        with tempfile.TemporaryDirectory() as temp_dir:
            module._primary_lora_root = lambda: Path(temp_dir)
            accepted = asyncio.run(
                module._materialize_lora_upload("style.safetensors", Stream([b"a", b"b"]), limit_bytes=2)
            )
            self.assertEqual(accepted["size"], 2)
            self.assertEqual(accepted["name"], "style.safetensors")
            with self.assertRaises(module.RemoteClipUploadTooLarge):
                asyncio.run(
                    module._materialize_lora_upload("too-large.safetensors", Stream([b"ab", b"c"]), limit_bytes=2)
                )
            self.assertFalse(any(path.name.startswith(".upload-") for path in Path(temp_dir).rglob("*")))

    def test_lora_materialization_preserves_relative_path_and_reuses_matching_file(self):
        module = _load_remote_clip_module()
        payload = b"same lora"
        digest = hashlib.sha256(payload).hexdigest()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            module._primary_lora_root = lambda: root

            first = module._materialize_lora_bytes(r"Krea2\realism.safetensors", payload, digest)
            second = module._materialize_lora_bytes("Krea2/realism.safetensors", payload, digest)

            self.assertEqual(first["name"], "Krea2/realism.safetensors")
            self.assertEqual(second["name"], "Krea2/realism.safetensors")
            self.assertEqual((root / "Krea2" / "realism.safetensors").read_bytes(), payload)
            self.assertFalse((root / "cutlery_remote").exists())

    def test_lora_materialization_rejects_conflicting_file_at_same_relative_path(self):
        module = _load_remote_clip_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "Krea2" / "realism.safetensors"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"existing")
            module._primary_lora_root = lambda: root

            with self.assertRaisesRegex(ValueError, "different contents"):
                module._materialize_lora_bytes("Krea2/realism.safetensors", b"replacement")

            self.assertEqual(target.read_bytes(), b"existing")

    def test_lora_materialization_rejects_unsafe_relative_paths(self):
        module = _load_remote_clip_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            module._primary_lora_root = lambda: Path(temp_dir)
            for source_name in ("../escape.safetensors", "/absolute.safetensors", r"C:\escape.safetensors"):
                with self.subTest(source_name=source_name), self.assertRaises(ValueError):
                    module._materialize_lora_bytes(source_name, b"payload")

    def test_dotenv_value_reads_comfy_root_env_file_after_process_environment(self):
        from cutlery_remote.dotenv import env_value

        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv = Path(temp_dir) / ".env"
            dotenv.write_text(
                "\n".join(
                    [
                        "CUTLERY_REMOTE_CLIP_BASE_URL=192.0.2.247:8188",
                        "CUTLERY_REMOTE_TOKEN=from-dotenv",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            folder_paths = types.SimpleNamespace(base_path=temp_dir)

            with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}), mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(env_value("CUTLERY_REMOTE_CLIP_BASE_URL"), "192.0.2.247:8188")
            with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}), mock.patch.dict(
                os.environ, {"CUTLERY_REMOTE_TOKEN": "from-process"}, clear=True
            ):
                self.assertEqual(env_value("CUTLERY_REMOTE_TOKEN"), "from-process")

    def test_input_types_use_name_only_local_inventory_for_text_encoder_dropdown(self):
        module = _load_remote_clip_module()
        module._remote_clip_mode = lambda: module.REMOTE_CLIP_MODE_REMOTE
        module.list_clip_text_encoder_names = lambda: ["local-t5.safetensors", "local-q4.gguf"]
        module.fetch_remote_clip_inventory = mock.Mock(side_effect=AssertionError("object_info must not probe remote CLIP"))
        module._local_clip_inventory = mock.Mock(side_effect=AssertionError("object_info must not hash local CLIP files"))
        module._local_lora_inventory = mock.Mock(side_effect=AssertionError("object_info must not hash local LoRA files"))

        inputs = module.CutleryRemoteClipTextEncode.INPUT_TYPES()

        self.assertEqual(inputs["required"]["text_encoder"][0], ["local-q4.gguf", "local-t5.safetensors"])
        self.assertTrue(inputs["required"]["text_encoder"][1]["defaultInput"])
        self.assertEqual(inputs["required"]["clip_type"][0], list(module.CLIP_TYPES))
        self.assertNotIn("lora_1", inputs["required"])
        self.assertNotIn("strength_clip_1", inputs["required"])
        self.assertEqual(inputs["optional"]["lora_chain"][0], "CUTLERY_LORA_CHAIN")

    def test_input_types_do_not_fetch_remote_inventory_during_object_info(self):
        module = _load_remote_clip_module()
        module.remote_clip_base_url = lambda: ""
        module.list_clip_text_encoder_names = lambda: []
        module.fetch_remote_clip_inventory = mock.Mock(side_effect=AssertionError("object_info must not probe remote CLIP"))
        module._local_clip_inventory = mock.Mock(side_effect=AssertionError("object_info must not hash local CLIP files"))
        module._local_lora_inventory = mock.Mock(side_effect=AssertionError("object_info must not hash local LoRA files"))

        inputs = module.CutleryRemoteClipTextEncode.INPUT_TYPES()

        module.fetch_remote_clip_inventory.assert_not_called()
        module._local_clip_inventory.assert_not_called()
        module._local_lora_inventory.assert_not_called()
        self.assertEqual(inputs["required"]["text_encoder"][0], [module.CONFIGURE_REMOTE_CHOICE])
        self.assertEqual(inputs["required"]["text_encoder"][1]["default"], module.CONFIGURE_REMOTE_CHOICE)
        self.assertEqual(inputs["optional"]["lora_chain"][0], "CUTLERY_LORA_CHAIN")

    def test_client_mode_remote_clip_model_fields_use_schema_stable_placeholder_combos(self):
        module = _load_remote_clip_module()
        module._remote_clip_mode = lambda: module.REMOTE_CLIP_MODE_DIRECT
        module.remote_clip_base_url = lambda: "http://remote.example:8188"
        module.fetch_remote_clip_inventory = mock.Mock(side_effect=AssertionError("object_info must not probe remote CLIP"))
        module._local_clip_inventory = mock.Mock(side_effect=AssertionError("object_info must not hash local CLIP files"))
        module._local_lora_inventory = mock.Mock(side_effect=AssertionError("object_info must not hash local LoRA files"))

        single_inputs = module.CutleryRemoteClipTextEncode.INPUT_TYPES()
        dual_inputs = module.CutleryRemoteDualClipTextEncode.INPUT_TYPES()
        qwen_inputs = module.CutleryRemoteTextEncodeQwenImageEditPlus.INPUT_TYPES()

        self.assertEqual(single_inputs["required"]["text_encoder"][0], [module.LOADING_REMOTE_CHOICES])
        self.assertEqual(single_inputs["required"]["text_encoder"][1]["default"], module.LOADING_REMOTE_CHOICES)
        self.assertEqual(dual_inputs["required"]["clip_name1"][0], [module.LOADING_REMOTE_CHOICES])
        self.assertEqual(dual_inputs["required"]["clip_name2"][0], [module.LOADING_REMOTE_CHOICES])
        self.assertEqual(qwen_inputs["required"]["text_encoder"][0], [module.LOADING_REMOTE_CHOICES])
        self.assertEqual(qwen_inputs["required"]["vae_name"][0], [module.LOADING_REMOTE_CHOICES])
        self.assertEqual(qwen_inputs["required"]["vae_name"][1]["default"], module.LOADING_REMOTE_CHOICES)
        module.fetch_remote_clip_inventory.assert_not_called()

    def test_remote_clip_text_encoder_validation_accepts_refreshed_remote_choice(self):
        module = _load_remote_clip_module()

        self.assertTrue(module.CutleryRemoteClipTextEncode.VALIDATE_INPUTS("remote-t5.safetensors"))
        self.assertEqual(
            module.CutleryRemoteClipTextEncode.VALIDATE_INPUTS(""),
            "A remote CLIP text encoder must be selected.",
        )

    def test_qwen_image_edit_input_types_expose_text_encoder_and_vae_dropdowns_without_remote_probe(self):
        module = _load_remote_clip_module()
        module._remote_clip_mode = lambda: module.REMOTE_CLIP_MODE_REMOTE
        module.list_clip_text_encoder_names = lambda: ["local-qwen.gguf"]
        module._vae_names = lambda: ["qwen-vae.safetensors"]
        module.fetch_remote_clip_inventory = mock.Mock(side_effect=AssertionError("object_info must not probe remote CLIP"))
        module._local_clip_inventory = mock.Mock(side_effect=AssertionError("object_info must not hash local CLIP files"))
        module._local_lora_inventory = mock.Mock(side_effect=AssertionError("object_info must not hash local LoRA files"))

        inputs = module.CutleryRemoteTextEncodeQwenImageEditPlus.INPUT_TYPES()

        self.assertEqual(inputs["required"]["text_encoder"][0], ["local-qwen.gguf"])
        self.assertEqual(inputs["required"]["vae_name"][0], [module.NONE_CHOICE, "qwen-vae.safetensors"])
        self.assertEqual(inputs["optional"]["image1"][0], "IMAGE")
        self.assertEqual(inputs["optional"]["image2"][0], "IMAGE")
        self.assertEqual(inputs["optional"]["image3"][0], "IMAGE")
        module.fetch_remote_clip_inventory.assert_not_called()

    def test_local_inventory_includes_regular_and_gguf_text_encoders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for folder_name, names in {
                "text_encoders": ["remote-t5.safetensors"],
                "clip_gguf": ["remote-q4.gguf", "shared.gguf"],
            }.items():
                folder = root / folder_name
                folder.mkdir()
                for name in names:
                    (folder / name).write_bytes(name.encode("utf-8"))

            folder_paths = types.SimpleNamespace(
                get_filename_list=lambda key: {
                    "text_encoders": ["remote-t5.safetensors", "shared.gguf"],
                    "clip_gguf": ["remote-q4.gguf", "shared.gguf"],
                    "loras": [],
                }.get(key, []),
                get_full_path_or_raise=lambda key, name: str(root / key / name),
                get_folder_paths=lambda key: [str(root / key)],
            )
            module = _load_remote_clip_module(folder_paths=folder_paths)

            with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
                inventory = module.local_remote_clip_inventory()

        self.assertEqual(inventory["text_encoders"], ["remote-q4.gguf", "remote-t5.safetensors", "shared.gguf"])

    def test_local_inventory_includes_vae_names(self):
        module = _load_remote_clip_module()
        module._local_lora_inventory = lambda: []
        module._local_clip_inventory = lambda: []
        module._vae_names = lambda: ["vae-a.safetensors", "pixel_space"]

        inventory = module.local_remote_clip_inventory()

        self.assertEqual(inventory["vaes"], ["vae-a.safetensors", "pixel_space"])

    def test_remote_clip_timeout_defaults_to_sixty_seconds(self):
        module = _load_remote_clip_module()

        self.assertEqual(module._remote_clip_timeout(), 60.0)

    def test_remote_clip_encode_timeout_defaults_to_ten_minutes_and_has_specific_override(self):
        module = _load_remote_clip_module()

        self.assertEqual(module._remote_clip_encode_timeout(), 600.0)

        values = {
            "CUTLERY_REMOTE_CLIP_TIMEOUT_S": "45",
            "CUTLERY_REMOTE_CLIP_ENCODE_TIMEOUT_S": "321",
        }
        module.env_value = lambda name, default="": values.get(name, default)

        self.assertEqual(module._remote_clip_encode_timeout(), 321.0)
        values.pop("CUTLERY_REMOTE_CLIP_ENCODE_TIMEOUT_S")
        self.assertEqual(module._remote_clip_encode_timeout(), 45.0)

    def test_remote_clip_encode_requests_use_longer_first_load_timeout(self):
        from cutlery_remote.serialization import encode_value_bundle

        module = _load_remote_clip_module()
        bundle = encode_value_bundle([[torch.zeros((1, 1, 1), dtype=torch.float32), {}]])
        module._remote_clip_encode_timeout = mock.Mock(return_value=321.0)
        module._post_json = mock.Mock(return_value={"ok": True, "conditioning": bundle})

        for post, path in (
            (module.post_remote_clip_encode, "/cutlery/remote/clip/text-encode"),
            (module.post_remote_dual_clip_encode, "/cutlery/remote/clip/dual-text-encode"),
            (module.post_remote_qwen_image_edit_plus_encode, "/cutlery/remote/clip/qwen-image-edit-plus"),
        ):
            with self.subTest(path=path):
                module._post_json.reset_mock()

                self.assertEqual(post({"prompt": "hello"}), bundle)

                module._post_json.assert_called_once_with(path, {"prompt": "hello"}, timeout=321.0)

    def test_remote_clip_base_url_defaults_to_direct_comfyui_target(self):
        module = _load_remote_clip_module()
        values = {
            "CUTLERY_REMOTE_CLIP_BASE_URL": "comfy-remote.example:8188",
        }
        module.env_value = lambda name, default="": values.get(name, default)

        self.assertEqual(module.remote_clip_base_url(), "http://comfy-remote.example:8188")

    def test_remote_clip_remote_mode_has_no_outbound_target(self):
        module = _load_remote_clip_module()
        values = {
            "CUTLERY_REMOTE_CLIP_MODE": "remote",
            "CUTLERY_REMOTE_CLIP_BASE_URL": "http://direct.example:8188",
        }
        module.env_value = lambda name, default="": values.get(name, default)

        self.assertEqual(module.remote_clip_base_url(), "")
        self.assertIn("CUTLERY_REMOTE_CLIP_MODE=direct", module._remote_clip_target_hint())

    def test_remote_clip_auth_token_uses_shared_remote_token(self):
        module = _load_remote_clip_module()
        values = {
            "CUTLERY_REMOTE_TOKEN": "shared-token",
        }
        module.env_value = lambda name, default="": values.get(name, default)

        self.assertEqual(module._remote_clip_auth_headers(), {"Authorization": "Bearer shared-token"})

    def test_remote_clip_auth_token_falls_back_to_configured_remote_token(self):
        module = _load_remote_clip_module()
        module.env_value = lambda _name, default="": default
        module.configured_remote_token = lambda: "configured-token"

        self.assertEqual(module._remote_clip_auth_headers(), {"Authorization": "Bearer configured-token"})

    def test_lora_inventory_merges_by_hash_and_prefers_local_display_path(self):
        module = _load_remote_clip_module()

        merged = module._merge_lora_inventories(
            local_entries=[
                {"name": "local/shared-style.safetensors", "sha256": "aaa", "size": 10},
                {"name": "local-only.safetensors", "sha256": "bbb", "size": 11},
            ],
            remote_entries=[
                {"name": "remote/different-name.safetensors", "sha256": "aaa", "size": 10},
                {"name": "remote-only.safetensors", "sha256": "ccc", "size": 12},
            ],
        )

        by_hash = {entry["sha256"]: entry for entry in merged}
        self.assertEqual(sorted(entry["display_name"] for entry in merged), ["local-only.safetensors", "local/shared-style.safetensors", "remote-only.safetensors"])
        self.assertEqual(by_hash["aaa"]["display_name"], "local/shared-style.safetensors")
        self.assertEqual(by_hash["aaa"]["local_name"], "local/shared-style.safetensors")
        self.assertEqual(by_hash["aaa"]["remote_name"], "remote/different-name.safetensors")
        self.assertEqual(by_hash["ccc"]["display_name"], "remote-only.safetensors")

    def test_input_types_do_not_expose_manual_lora_slots(self):
        module = _load_remote_clip_module()
        module._local_clip_inventory = lambda: []
        module._local_lora_inventory = lambda: [
            {"name": "local/shared-style.safetensors", "sha256": "aaa", "size": 10},
            {"name": "local-only.safetensors", "sha256": "bbb", "size": 11},
        ]
        module.fetch_remote_clip_inventory = lambda **_kwargs: {
            "text_encoders": ["remote-t5.safetensors"],
            "loras": ["remote/different-name.safetensors", "remote-only.safetensors"],
            "lora_inventory": [
                {"name": "remote/different-name.safetensors", "sha256": "aaa", "size": 10},
                {"name": "remote-only.safetensors", "sha256": "ccc", "size": 12},
            ],
            "clip_types": ["stable_diffusion", "flux"],
        }

        inputs = module.CutleryRemoteClipTextEncode.INPUT_TYPES()

        self.assertNotIn("lora_1", inputs["required"])
        self.assertNotIn("strength_clip_1", inputs["required"])
        self.assertNotIn("lora_name_1", inputs.get("optional", {}))
        self.assertEqual(inputs["optional"]["lora_chain"][0], "CUTLERY_LORA_CHAIN")

    def test_remote_clip_node_accepts_hidden_unique_id_for_materialization_progress(self):
        module = _load_remote_clip_module()
        module._local_clip_inventory = lambda: []
        module.fetch_remote_clip_inventory = lambda **_kwargs: {
            "text_encoders": ["remote-t5.safetensors"],
            "lora_inventory": [],
            "clip_types": ["stable_diffusion"],
        }

        inputs = module.CutleryRemoteClipTextEncode.INPUT_TYPES()

        self.assertEqual(inputs["hidden"]["unique_id"], "UNIQUE_ID")

    def test_inventory_route_runs_inventory_work_in_thread(self):
        routes = _Routes()
        module = _load_remote_clip_module(routes=routes)
        calls = []

        async def fake_to_thread(fn, *args, **kwargs):
            calls.append((fn, args, kwargs))
            return fn(*args, **kwargs)

        module.asyncio.to_thread = fake_to_thread
        module.local_remote_clip_inventory = mock.Mock(
            return_value={
                "ok": True,
                "text_encoders": ["remote-t5.safetensors"],
                "clip_inventory": [],
                "loras": [],
                "lora_inventory": [],
                "clip_types": ["stable_diffusion"],
                "vaes": [],
            }
        )

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=True):
            response = asyncio.run(
                routes.handlers[("GET", "/cutlery/remote/clip/inventory")](
                    types.SimpleNamespace(
                        headers={"Authorization": "Bearer abc123"},
                        query={"include_hashes": "0"},
                    )
                )
            )

        self.assertEqual(response["status"], 200)
        self.assertEqual(response["payload"]["text_encoders"], ["remote-t5.safetensors"])
        self.assertEqual(calls, [(module.local_remote_clip_inventory, (), {"include_hashes": False})])

    def test_remote_clip_choices_route_returns_remote_device_names_only(self):
        routes = _Routes()
        module = _load_remote_clip_module(routes=routes)
        module.fetch_remote_clip_inventory = mock.Mock(
            return_value={
                "text_encoders": ["remote-t5.gguf", "remote-projection.safetensors"],
                "clip_types": ["flux", "ltxv"],
                "vaes": ["remote-vae.safetensors"],
                "clip_inventory": [
                    {"name": "remote-t5.gguf", "sha256": "aaa", "size": 10},
                    {"name": "remote-projection.safetensors", "sha256": "bbb", "size": 11},
                ],
            }
        )
        module._local_clip_inventory = mock.Mock(side_effect=AssertionError("choices route must not hash local clips"))

        response = asyncio.run(routes.handlers[("GET", "/cutlery/remote/clip/choices")](types.SimpleNamespace()))

        self.assertEqual(response["status"], 200)
        self.assertEqual(response["payload"]["text_encoders"], ["remote-t5.gguf", "remote-projection.safetensors"])
        self.assertEqual(response["payload"]["clip_types"], ["flux", "ltxv"])
        self.assertEqual(response["payload"]["vaes"], ["remote-vae.safetensors"])
        module.fetch_remote_clip_inventory.assert_called_once_with(timeout=module.WIDGET_INVENTORY_TIMEOUT_SECONDS, include_hashes=False)
        module._local_clip_inventory.assert_not_called()

    def test_prepare_loras_uses_exact_remote_path_without_local_hashing(self):
        module = _load_remote_clip_module()
        module._local_lora_inventory = mock.Mock(side_effect=AssertionError("prepare must not hash all local LoRAs"))
        module.fetch_remote_clip_inventory = lambda **_kwargs: {
            "lora_inventory": [{"name": "remote/style.safetensors", "sha256": "", "size": None}]
        }
        materialize = mock.Mock()
        module._materialize_lora_to_remote = materialize

        prepared = module._prepare_remote_lora_entries(
            [{"lora_name": "remote/style.safetensors", "strength_clip": 0.65}]
        )

        self.assertEqual(prepared, [{"lora_name": "remote/style.safetensors", "strength_clip": 0.65}])
        materialize.assert_not_called()
        module._local_lora_inventory.assert_not_called()

    def test_prepare_loras_fetches_inventory_with_encode_timeout(self):
        module = _load_remote_clip_module()
        module._remote_clip_encode_timeout = mock.Mock(return_value=321.0)
        module.fetch_remote_clip_inventory = mock.Mock(
            return_value={"lora_inventory": [{"name": "remote/shared-style.safetensors", "sha256": "", "size": None}]}
        )

        module._prepare_remote_lora_entries([{"lora_name": "remote/shared-style.safetensors", "strength_clip": 0.65}])

        module.fetch_remote_clip_inventory.assert_called_once_with(timeout=321.0, include_hashes=False)

    def test_prepare_loras_materializes_local_only_lora_before_remote_encode(self):
        module = _load_remote_clip_module()
        module.fetch_remote_clip_inventory = lambda **_kwargs: {"lora_inventory": []}
        module._selected_local_lora_entry = mock.Mock(return_value={"local_name": "local-only.safetensors", "sha256": "bbb", "size": 11})
        module._materialize_lora_to_remote = mock.Mock(
            return_value={"name": "local-only.safetensors", "sha256": "bbb", "size": 11}
        )

        prepared = module._prepare_remote_lora_entries(
            [{"lora_name": "local-only.safetensors", "strength_clip": 0.8}]
        )

        self.assertEqual(prepared, [{"lora_name": "local-only.safetensors", "strength_clip": 0.8}])
        module._materialize_lora_to_remote.assert_called_once()

    def test_clip_inventory_merges_by_hash_and_prefers_local_display_path(self):
        module = _load_remote_clip_module()

        merged = module._merge_clip_inventories(
            local_entries=[
                {"name": "local/gemma.safetensors", "sha256": "aaa", "size": 10},
                {"name": "local-only.gguf", "sha256": "bbb", "size": 11},
            ],
            remote_entries=[
                {"name": "remote/different-gemma.safetensors", "sha256": "aaa", "size": 10},
                {"name": "remote-only.safetensors", "sha256": "ccc", "size": 12},
            ],
        )

        by_hash = {entry["sha256"]: entry for entry in merged}
        self.assertEqual(
            sorted(entry["display_name"] for entry in merged),
            ["local-only.gguf", "local/gemma.safetensors", "remote-only.safetensors"],
        )
        self.assertEqual(by_hash["aaa"]["display_name"], "local/gemma.safetensors")
        self.assertEqual(by_hash["aaa"]["local_name"], "local/gemma.safetensors")
        self.assertEqual(by_hash["aaa"]["remote_name"], "remote/different-gemma.safetensors")
        self.assertEqual(by_hash["ccc"]["display_name"], "remote-only.safetensors")

    def test_prepare_clips_uses_exact_remote_path_without_local_hashing(self):
        module = _load_remote_clip_module()
        module._local_clip_inventory = mock.Mock(side_effect=AssertionError("prepare must not hash all local CLIPs"))
        module.fetch_remote_clip_inventory = lambda **_kwargs: {
            "clip_inventory": [{"name": "remote/gemma.gguf", "sha256": "", "size": None}]
        }
        module._materialize_clip_to_remote = mock.Mock(side_effect=AssertionError("exact remote names must not be uploaded"))

        prepared = module._prepare_remote_clip_names(["remote/gemma.gguf"])

        self.assertEqual(prepared, ["remote/gemma.gguf"])
        module._materialize_clip_to_remote.assert_not_called()
        module._local_clip_inventory.assert_not_called()

    def test_prepare_clips_fetches_inventory_with_encode_timeout(self):
        module = _load_remote_clip_module()
        module._remote_clip_encode_timeout = mock.Mock(return_value=321.0)
        module.fetch_remote_clip_inventory = mock.Mock(
            return_value={"clip_inventory": [{"name": "remote/gemma.gguf", "sha256": "", "size": None}]}
        )

        module._prepare_remote_clip_names(["remote/gemma.gguf"])

        module.fetch_remote_clip_inventory.assert_called_once_with(timeout=321.0, include_hashes=False)

    def test_prepare_clips_materializes_local_only_clip_before_remote_encode(self):
        module = _load_remote_clip_module()
        module.fetch_remote_clip_inventory = lambda **_kwargs: {"clip_inventory": []}
        module._selected_local_clip_entry = mock.Mock(return_value={"local_name": "host/qwen.gguf", "sha256": "bbb", "size": 11})
        module._materialize_clip_to_remote = mock.Mock(
            return_value={"name": "cutlery_remote/bbbbbbbbbbbb-qwen.gguf", "sha256": "bbb", "size": 11}
        )

        prepared = module._prepare_remote_clip_names(["host/qwen.gguf"], progress_node_id="node-7")

        self.assertEqual(prepared, ["cutlery_remote/bbbbbbbbbbbb-qwen.gguf"])
        module._materialize_clip_to_remote.assert_called_once()
        self.assertEqual(module._materialize_clip_to_remote.call_args.kwargs["progress_node_id"], "node-7")

    def test_remote_dual_encode_loads_two_clip_paths_with_comfyui_gguf_loader(self):
        module = _load_remote_clip_module()
        path_requests = []
        folder_paths = types.SimpleNamespace(
            get_full_path_or_raise=lambda key, name: path_requests.append((key, name)) or f"/models/{key}/{name}",
            get_folder_paths=lambda key: [f"/models/{key}"],
        )
        comfy_sd = types.SimpleNamespace(
            CLIPType=types.SimpleNamespace(LTXV="LTXV", STABLE_DIFFUSION="STABLE_DIFFUSION"),
            load_clip=mock.Mock(side_effect=AssertionError("regular load_clip should not load GGUF dual clips")),
        )
        comfy = types.SimpleNamespace(sd=comfy_sd)
        gguf_loader = mock.Mock()
        gguf_loader.load_data.return_value = ["gguf-state", "projection-state"]
        gguf_loader.load_patcher.return_value = _FakeClip()

        class _GGUFLoader:
            def __new__(cls):
                return gguf_loader

        fake_gguf_module = types.SimpleNamespace(NODE_CLASS_MAPPINGS={"CLIPLoaderGGUF": _GGUFLoader})

        with mock.patch.dict(
            sys.modules,
            {
                "folder_paths": folder_paths,
                "comfy": comfy,
                "comfy.sd": comfy_sd,
                "_cutlery_comfyui_gguf": fake_gguf_module,
            },
        ):
            payload = module.encode_remote_dual_clip_text(
                {
                    "prompt": "remote prompt",
                    "clip_name1": "gemma-q4.gguf",
                    "clip_name2": "ltx-projection.safetensors",
                    "clip_type": "ltxv",
                }
            )

        self.assertTrue(payload["ok"])
        self.assertIn(("clip_gguf", "gemma-q4.gguf"), path_requests)
        self.assertIn(("text_encoders", "ltx-projection.safetensors"), path_requests)
        gguf_loader.load_data.assert_called_once_with(
            ("/models/clip_gguf/gemma-q4.gguf", "/models/text_encoders/ltx-projection.safetensors")
        )
        gguf_loader.load_patcher.assert_called_once_with(
            ("/models/clip_gguf/gemma-q4.gguf", "/models/text_encoders/ltx-projection.safetensors"),
            "LTXV",
            ["gguf-state", "projection-state"],
        )

    def test_dual_clip_node_posts_prepared_clips_loras_and_decodes_conditioning(self):
        from cutlery_remote.serialization import encode_value_bundle

        module = _load_remote_clip_module()
        conditioning = [[torch.ones((1, 2, 3), dtype=torch.float32), {"pooled_output": torch.zeros((1, 3), dtype=torch.float32)}]]
        bundle = encode_value_bundle(conditioning)
        module._prepare_remote_clip_names = mock.Mock(return_value=["remote/gemma.gguf", "remote/projection.safetensors"])
        module._prepare_remote_lora_entries = mock.Mock(return_value=[{"lora_name": "remote/style.safetensors", "strength_clip": 0.65}])

        with mock.patch.object(module, "post_remote_dual_clip_encode", return_value=bundle) as post:
            (decoded,) = module.CutleryRemoteDualClipTextEncode().encode(
                prompt="a sharp studio portrait",
                clip_name1="host/gemma.gguf",
                clip_name2="host/projection.safetensors",
                clip_type="ltxv",
                lora_chain={"loras": [{"lora_name": "host/style.safetensors", "strength_clip": 0.65}]},
                unique_id="node-42",
            )

        self.assertTrue(torch.equal(decoded[0][0], conditioning[0][0]))
        payload = post.call_args.args[0]
        self.assertEqual(payload["prompt"], "a sharp studio portrait")
        self.assertEqual(payload["clip_name1"], "remote/gemma.gguf")
        self.assertEqual(payload["clip_name2"], "remote/projection.safetensors")
        self.assertEqual(payload["clip_type"], "ltxv")
        self.assertEqual(payload["loras"], [{"lora_name": "remote/style.safetensors", "strength_clip": 0.65}])
        module._prepare_remote_clip_names.assert_called_once_with(
            ["host/gemma.gguf", "host/projection.safetensors"],
            progress_node_id="node-42",
        )

    def test_clear_remote_clips_route_proxies_without_auth_and_clears_with_auth(self):
        routes = _Routes()
        module = _load_remote_clip_module(routes=routes)
        module._post_json = mock.Mock(return_value={"ok": True, "deleted_count": 2, "proxied": True})
        module._clear_materialized_clips = mock.Mock(return_value={"ok": True, "deleted_count": 1})

        with mock.patch.dict(
            os.environ,
            {"CUTLERY_REMOTE_TOKEN": "abc123", "CUTLERY_REMOTE_CLIP_BASE_URL": "http://remote.example:8188"},
            clear=True,
        ):
            proxied = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/clip/clips/clear")](
                    types.SimpleNamespace(headers={}, json=lambda: {})
                )
            )
            cleared = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/clip/clips/clear")](
                    types.SimpleNamespace(headers={"Authorization": "Bearer abc123"}, json=lambda: {})
                )
            )

        self.assertEqual(proxied["status"], 200)
        self.assertEqual(proxied["payload"]["deleted_count"], 2)
        module._post_json.assert_called_once_with("/cutlery/remote/clip/clips/clear", {})
        self.assertEqual(cleared["status"], 200)
        self.assertEqual(cleared["payload"]["deleted_count"], 1)

    def test_remote_materialize_clip_route_streams_payload_to_materialized_folder(self):
        routes = _Routes()
        payload = b"local clip bytes"
        digest = hashlib.sha256(payload).hexdigest()

        class _FakeStream:
            async def iter_chunked(self, _chunk_size):
                yield payload[:5]
                yield payload[5:]

        with tempfile.TemporaryDirectory() as temp_dir:
            text_encoder_root = Path(temp_dir) / "text_encoders"
            text_encoder_root.mkdir()
            folder_paths = types.SimpleNamespace(
                get_filename_list=lambda _key: [],
                get_full_path_or_raise=lambda key, name: str(Path(temp_dir) / key / name),
                get_folder_paths=lambda key: [str(text_encoder_root)] if key == "text_encoders" else [str(Path(temp_dir) / key)],
            )
            module = _load_remote_clip_module(routes=routes, folder_paths=folder_paths)
            request = types.SimpleNamespace(
                headers={
                    "Authorization": "Bearer abc123",
                    "X-Cutlery-Clip-Name": "host/gemma.gguf",
                    "X-Cutlery-Clip-SHA256": digest,
                },
                content=_FakeStream(),
                _client_max_size=1,
            )

            with (
                mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=True),
                mock.patch.dict(sys.modules, {"folder_paths": folder_paths}),
            ):
                response = asyncio.run(routes.handlers[("POST", "/cutlery/remote/clip/clips/materialize")](request))

            materialized_name = response["payload"]["name"]
            materialized_path = text_encoder_root / materialized_name
            materialized_bytes = materialized_path.read_bytes()

            self.assertEqual(response["status"], 200)
            self.assertEqual(response["payload"]["sha256"], digest)
            self.assertEqual(response["payload"]["size"], len(payload))
            self.assertTrue(materialized_name.startswith("cutlery_remote/"))
            self.assertGreaterEqual(request._client_max_size, module.REMOTE_CLIP_FILE_UPLOAD_LIMIT_BYTES)
            self.assertEqual(materialized_bytes, payload)

    def test_remote_materialize_qwen_image_route_streams_payload_to_input_folder(self):
        routes = _Routes()
        payload = b"png image bytes"
        digest = hashlib.sha256(payload).hexdigest()

        class _FakeStream:
            async def iter_chunked(self, _chunk_size):
                yield payload[:4]
                yield payload[4:]

        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir) / "input"
            input_root.mkdir()
            folder_paths = types.SimpleNamespace(
                get_filename_list=lambda _key: [],
                get_full_path_or_raise=lambda key, name: str(Path(temp_dir) / key / name),
                get_folder_paths=lambda key: [str(Path(temp_dir) / key)],
                get_input_directory=lambda: str(input_root),
                get_annotated_filepath=lambda name: str(input_root / name),
            )
            module = _load_remote_clip_module(routes=routes, folder_paths=folder_paths)
            request = types.SimpleNamespace(
                headers={
                    "Authorization": "Bearer abc123",
                    "X-Cutlery-Image-Name": "image1_0001.png",
                    "X-Cutlery-Image-SHA256": digest,
                },
                content=_FakeStream(),
                _client_max_size=1,
            )

            with (
                mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=True),
                mock.patch.dict(sys.modules, {"folder_paths": folder_paths}),
            ):
                response = asyncio.run(routes.handlers[("POST", "/cutlery/remote/clip/images/materialize")](request))

            materialized_name = response["payload"]["name"]
            materialized_path = input_root / materialized_name

            self.assertEqual(response["status"], 200)
            self.assertEqual(response["payload"]["sha256"], digest)
            self.assertEqual(response["payload"]["size"], len(payload))
            self.assertEqual(response["payload"]["type"], "input")
            self.assertEqual(response["payload"]["subfolder"], "cutlery_remote/qwen")
            self.assertTrue(materialized_name.startswith("cutlery_remote/qwen/"))
            self.assertGreaterEqual(request._client_max_size, module.REMOTE_CLIP_FILE_UPLOAD_LIMIT_BYTES)
            self.assertEqual(materialized_path.read_bytes(), payload)

    def test_remote_materialize_qwen_image_route_rejects_hash_mismatch(self):
        routes = _Routes()
        payload = b"png image bytes"

        class _FakeStream:
            async def iter_chunked(self, _chunk_size):
                yield payload

        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir) / "input"
            input_root.mkdir()
            folder_paths = types.SimpleNamespace(
                get_filename_list=lambda _key: [],
                get_full_path_or_raise=lambda key, name: str(Path(temp_dir) / key / name),
                get_folder_paths=lambda key: [str(Path(temp_dir) / key)],
                get_input_directory=lambda: str(input_root),
            )
            module = _load_remote_clip_module(routes=routes, folder_paths=folder_paths)

            with (
                mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=True),
                mock.patch.dict(sys.modules, {"folder_paths": folder_paths}),
            ):
                response = asyncio.run(
                    routes.handlers[("POST", "/cutlery/remote/clip/images/materialize")](
                        types.SimpleNamespace(
                            headers={
                                "Authorization": "Bearer abc123",
                                "X-Cutlery-Image-Name": "../bad.png",
                                "X-Cutlery-Image-SHA256": "0" * 64,
                            },
                            content=_FakeStream(),
                            _client_max_size=1,
                        )
                    )
                )

        self.assertEqual(response["status"], 400)
        self.assertIn("SHA-256", response["payload"]["error"])

    def test_clear_remote_qwen_images_route_proxies_without_auth_and_clears_with_auth(self):
        routes = _Routes()
        module = _load_remote_clip_module(routes=routes)
        module._post_json = mock.Mock(return_value={"ok": True, "deleted_count": 2, "proxied": True})
        module._clear_materialized_qwen_images = mock.Mock(return_value={"ok": True, "deleted_count": 1})

        with mock.patch.dict(
            os.environ,
            {"CUTLERY_REMOTE_TOKEN": "abc123", "CUTLERY_REMOTE_CLIP_BASE_URL": "http://remote.example:8188"},
            clear=True,
        ):
            proxied = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/clip/images/clear")](
                    types.SimpleNamespace(headers={}, json=lambda: {})
                )
            )
            cleared = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/clip/images/clear")](
                    types.SimpleNamespace(headers={"Authorization": "Bearer abc123"}, json=lambda: {})
                )
            )

        self.assertEqual(proxied["status"], 200)
        self.assertEqual(proxied["payload"]["deleted_count"], 2)
        module._post_json.assert_called_once_with("/cutlery/remote/clip/images/clear", {})
        self.assertEqual(cleared["status"], 200)
        self.assertEqual(cleared["payload"]["deleted_count"], 1)

    def test_materialize_lora_to_remote_streams_upload_with_progress_and_console_logs(self):
        module = _load_remote_clip_module()
        payload = b"abcde"
        digest = hashlib.sha256(payload).hexdigest()
        updates = []

        class _FakeProgressBar:
            def __init__(self, total, node_id=None):
                self.total = total
                self.node_id = node_id

            def update_absolute(self, value, total=None):
                updates.append((value, total or self.total, self.node_id))

        fake_comfy = types.ModuleType("comfy")
        fake_comfy.__path__ = []
        fake_utils = types.ModuleType("comfy.utils")
        fake_utils.ProgressBar = _FakeProgressBar
        fake_comfy.utils = fake_utils

        class _FakeResponse:
            status = 200
            reason = "OK"

            def read(self):
                return json.dumps(
                    {
                        "ok": True,
                        "name": "local/style.safetensors",
                        "sha256": digest,
                        "size": len(payload),
                    }
                ).encode("utf-8")

        class _FakeConnection:
            instances = []

            def __init__(self, host, port=None, timeout=None):
                self.host = host
                self.port = port
                self.timeout = timeout
                self.headers = {}
                self.sent = []
                _FakeConnection.instances.append(self)

            def putrequest(self, method, target):
                self.method = method
                self.target = target

            def putheader(self, name, value):
                self.headers[name] = value

            def endheaders(self):
                self.headers_ended = True

            def send(self, data):
                self.sent.append(bytes(data))

            def getresponse(self):
                return _FakeResponse()

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as temp_dir:
            lora_path = Path(temp_dir) / "style.safetensors"
            lora_path.write_bytes(payload)
            module._lora_full_path = lambda _name: str(lora_path)
            module.remote_clip_base_url = lambda: "http://remote.example:8188"
            module._remote_clip_auth_headers = lambda: {"Authorization": "Bearer abc123"}
            module._remote_clip_timeout = lambda: 12.0

            with (
                mock.patch.dict(sys.modules, {"comfy": fake_comfy, "comfy.utils": fake_utils}),
                mock.patch("http.client.HTTPConnection", _FakeConnection),
                mock.patch.object(module, "REMOTE_CLIP_UPLOAD_CHUNK_SIZE", 2),
                self.assertLogs("cutlery.remote.clip", level="INFO") as logs,
            ):
                response = module._materialize_lora_to_remote(
                    {"local_name": "local/style.safetensors", "sha256": digest, "size": len(payload)},
                    progress_node_id="node-42",
                )

        connection = _FakeConnection.instances[0]
        self.assertEqual(response["name"], "local/style.safetensors")
        self.assertEqual(connection.host, "remote.example")
        self.assertEqual(connection.port, 8188)
        self.assertEqual(connection.method, "POST")
        self.assertEqual(connection.target, "/cutlery/remote/clip/loras/materialize")
        self.assertEqual(connection.headers["Content-Length"], str(len(payload)))
        self.assertEqual(connection.headers["Authorization"], "Bearer abc123")
        self.assertEqual(connection.sent, [b"ab", b"cd", b"e"])
        self.assertEqual(updates[-1], (len(payload), len(payload), "node-42"))
        self.assertTrue(any("copied=5 B/5 B (100.0%)" in line for line in logs.output), logs.output)
        self.assertTrue(any("Remote LoRA materialization complete" in line and "copied=5 B" in line for line in logs.output), logs.output)

    def test_remote_materialize_lora_route_writes_payload_and_reports_relative_name(self):
        routes = _Routes()
        module = _load_remote_clip_module(routes=routes)
        payload = b"local lora bytes"
        digest = hashlib.sha256(payload).hexdigest()
        module._materialize_lora_upload = mock.AsyncMock(
            return_value={"name": "local/test-style.safetensors", "sha256": digest, "size": len(payload), "materialized": True}
        )

        class Stream:
            async def iter_chunked(self, _size):
                yield payload

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=True):
            response = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/clip/loras/materialize")](
                    types.SimpleNamespace(
                        headers={
                            "Authorization": "Bearer abc123",
                            "X-Cutlery-Lora-Name": "local/test-style.safetensors",
                            "X-Cutlery-Lora-SHA256": digest,
                        },
                        content=Stream(),
                    )
                )
            )

        self.assertEqual(response["status"], 200)
        self.assertEqual(response["payload"]["name"], "local/test-style.safetensors")
        module._materialize_lora_upload.assert_awaited_once_with(
            "local/test-style.safetensors",
            mock.ANY,
            digest,
            module.REMOTE_CLIP_LORA_UPLOAD_LIMIT_BYTES,
        )

    def test_remote_materialize_lora_route_rejects_one_byte_over_stream_limit(self):
        routes = _Routes()
        module = _load_remote_clip_module(routes=routes)
        payload = b"x" * (1024 * 1024 + 1)
        digest = hashlib.sha256(payload).hexdigest()
        module._primary_lora_root = lambda: Path(tempfile.gettempdir()) / "cutlery-test-loras"
        module._remote_clip_lora_upload_limit_bytes = lambda: 1024 * 1024

        class _LimitedRequest:
            def __init__(self):
                self.headers = {
                    "Authorization": "Bearer abc123",
                    "X-Cutlery-Lora-Name": "large.safetensors",
                    "X-Cutlery-Lora-SHA256": digest,
                }
                self._client_max_size = 1024 * 1024
                self.content = self

            async def iter_chunked(self, _size):
                yield payload

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=True):
            response = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/clip/loras/materialize")](_LimitedRequest())
            )

        self.assertEqual(response["status"], 413)
        self.assertIn("exceeds", response["payload"]["error"])

    def test_clear_remote_loras_route_proxies_without_auth_and_clears_with_auth(self):
        routes = _Routes()
        module = _load_remote_clip_module(routes=routes)
        module._post_json = mock.Mock(return_value={"ok": True, "deleted_count": 2, "proxied": True})
        module._clear_materialized_loras = mock.Mock(return_value={"ok": True, "deleted_count": 1})

        with mock.patch.dict(
            os.environ,
            {"CUTLERY_REMOTE_TOKEN": "abc123", "CUTLERY_REMOTE_CLIP_BASE_URL": "http://remote.example:8188"},
            clear=True,
        ):
            proxied = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/clip/loras/clear")](
                    types.SimpleNamespace(headers={}, json=lambda: {})
                )
            )
            cleared = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/clip/loras/clear")](
                    types.SimpleNamespace(headers={"Authorization": "Bearer abc123"}, json=lambda: {})
                )
            )

        self.assertEqual(proxied["status"], 200)
        self.assertEqual(proxied["payload"]["deleted_count"], 2)
        module._post_json.assert_called_once_with("/cutlery/remote/clip/loras/clear", {})
        self.assertEqual(cleared["status"], 200)
        self.assertEqual(cleared["payload"]["deleted_count"], 1)

    def test_remote_encode_loads_gguf_text_encoder_with_comfyui_gguf_loader(self):
        module = _load_remote_clip_module()
        path_requests = []
        folder_paths = types.SimpleNamespace(
            get_full_path_or_raise=lambda key, name: path_requests.append((key, name)) or f"/models/{key}/{name}",
            get_folder_paths=lambda key: [f"/models/{key}"],
        )
        comfy_sd = types.SimpleNamespace(
            CLIPType=types.SimpleNamespace(FLUX="FLUX", STABLE_DIFFUSION="STABLE_DIFFUSION"),
            load_clip=mock.Mock(side_effect=AssertionError("regular load_clip should not load GGUF text encoders")),
        )
        comfy = types.SimpleNamespace(sd=comfy_sd)
        gguf_loader = mock.Mock()
        gguf_loader.load_data.return_value = ["gguf-state"]
        gguf_loader.load_patcher.return_value = "clip"

        class _GGUFLoader:
            def __new__(cls):
                return gguf_loader

        fake_gguf_module = types.SimpleNamespace(NODE_CLASS_MAPPINGS={"CLIPLoaderGGUF": _GGUFLoader})

        with mock.patch.dict(
            sys.modules,
            {
                "folder_paths": folder_paths,
                "comfy": comfy,
                "comfy.sd": comfy_sd,
                "_cutlery_comfyui_gguf": fake_gguf_module,
            },
        ):
            clip = module._load_clip_for_remote_encode("remote-q4.gguf", "flux")

        self.assertEqual(clip, "clip")
        self.assertIn(("clip_gguf", "remote-q4.gguf"), path_requests)
        gguf_loader.load_data.assert_called_once_with(["/models/clip_gguf/remote-q4.gguf"])
        gguf_loader.load_patcher.assert_called_once_with(["/models/clip_gguf/remote-q4.gguf"], "FLUX", ["gguf-state"])

    def test_node_posts_prompt_and_decodes_remote_conditioning_bundle(self):
        from cutlery_remote.serialization import encode_value_bundle

        module = _load_remote_clip_module()
        conditioning = [[torch.ones((1, 2, 3), dtype=torch.float32), {"pooled_output": torch.zeros((1, 3), dtype=torch.float32)}]]
        bundle = encode_value_bundle(conditioning)
        node = module.CutleryRemoteClipTextEncode()
        module._prepare_remote_clip_names = lambda names, **_kwargs: names
        module._prepare_remote_lora_entries = lambda entries, **_kwargs: entries

        with mock.patch.object(module, "post_remote_clip_encode", return_value=bundle) as post:
            (decoded,) = node.encode(
                prompt="a sharp studio portrait",
                text_encoder="remote-t5.safetensors",
                clip_type="flux",
                lora_chain={
                    "loras": [
                        {"lora_name": "style-a.safetensors", "strength_model": 0.25, "strength_clip": 0.65}
                    ]
                },
            )

        self.assertTrue(torch.equal(decoded[0][0], conditioning[0][0]))
        self.assertTrue(torch.equal(decoded[0][1]["pooled_output"], conditioning[0][1]["pooled_output"]))
        payload = post.call_args.args[0]
        self.assertEqual(payload["prompt"], "a sharp studio portrait")
        self.assertEqual(payload["text_encoder"], "remote-t5.safetensors")
        self.assertEqual(payload["clip_type"], "flux")
        self.assertEqual(payload["loras"], [{"lora_name": "style-a.safetensors", "strength_clip": 0.65}])

    def test_qwen_node_uses_selected_remote_text_encoder_without_hash_inventory(self):
        from cutlery_remote.serialization import encode_value_bundle

        module = _load_remote_clip_module()
        conditioning = [[torch.ones((1, 2, 3), dtype=torch.float32), {"pooled_output": torch.zeros((1, 3), dtype=torch.float32)}]]
        bundle = encode_value_bundle(conditioning)
        module._prepare_remote_clip_names = mock.Mock(side_effect=AssertionError("Qwen must not use hash-aware CLIP prep"))
        module._local_clip_inventory = mock.Mock(side_effect=AssertionError("Qwen must not hash local CLIP files"))
        module.fetch_remote_clip_inventory = mock.Mock(side_effect=AssertionError("Qwen encode must not fetch hash inventory"))

        with mock.patch.object(module, "post_remote_qwen_image_edit_plus_encode", return_value=bundle) as post:
            module.CutleryRemoteTextEncodeQwenImageEditPlus().encode(
                prompt="make it brighter",
                text_encoder="remote/qwen.gguf",
                vae_name=module.NONE_CHOICE,
                unique_id="node-9",
            )

        self.assertEqual(post.call_args.args[0]["text_encoder"], "remote/qwen.gguf")
        module._prepare_remote_clip_names.assert_not_called()
        module._local_clip_inventory.assert_not_called()
        module.fetch_remote_clip_inventory.assert_not_called()

    def test_qwen_image_edit_node_posts_prompt_vae_and_materialized_image_refs(self):
        from cutlery_remote.serialization import encode_value_bundle

        module = _load_remote_clip_module()
        conditioning = [[torch.ones((1, 2, 3), dtype=torch.float32), {"pooled_output": torch.zeros((1, 3), dtype=torch.float32)}]]
        bundle = encode_value_bundle(conditioning)
        image1 = torch.ones((1, 8, 8, 3), dtype=torch.float32)
        image3 = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        module._prepare_remote_clip_names = mock.Mock(side_effect=AssertionError("Qwen must not use hash-aware CLIP prep"))

        def fake_image_ref(image, *, name: str = "image", progress_node_id: str | None = None):
            if image is None:
                return None
            return {
                "schema": module.REMOTE_CLIP_IMAGE_FILE_REF_BUNDLE_SCHEMA,
                "format": "png",
                "mime_type": "image/png",
                "width": int(image.shape[2]),
                "height": int(image.shape[1]),
                "channels": 3,
                "frames": [
                    {
                        "filename": f"{name}_0001.png",
                        "name": f"cutlery_remote/qwen/aaa-{name}_0001.png",
                        "subfolder": "cutlery_remote/qwen",
                        "type": "input",
                        "byte_count": 12,
                        "sha256": f"{name}-sha",
                    }
                ],
            }

        module._encode_optional_image_value = mock.Mock(side_effect=fake_image_ref)

        with mock.patch.object(module, "post_remote_qwen_image_edit_plus_encode", return_value=bundle) as post:
            (decoded,) = module.CutleryRemoteTextEncodeQwenImageEditPlus().encode(
                prompt="make it brighter",
                text_encoder="remote/qwen.gguf",
                vae_name="qwen-vae.safetensors",
                image1=image1,
                image3=image3,
                unique_id="node-9",
            )

        self.assertTrue(torch.equal(decoded[0][0], conditioning[0][0]))
        payload = post.call_args.args[0]
        self.assertEqual(payload["prompt"], "make it brighter")
        self.assertEqual(payload["text_encoder"], "remote/qwen.gguf")
        self.assertEqual(payload["clip_type"], "qwen_image")
        self.assertEqual(payload["vae_name"], "qwen-vae.safetensors")
        self.assertEqual(sorted(payload["images"]), ["image1", "image3"])
        self.assertEqual(payload["images"]["image1"]["schema"], module.REMOTE_CLIP_IMAGE_FILE_REF_BUNDLE_SCHEMA)
        self.assertEqual(payload["images"]["image1"]["mime_type"], "image/png")
        self.assertEqual(payload["images"]["image1"]["frames"][0]["filename"], "image1_0001.png")
        self.assertEqual(payload["images"]["image1"]["frames"][0]["type"], "input")
        self.assertNotIn("data", payload["images"]["image1"]["frames"][0])
        self.assertNotIn("manifest", payload["images"]["image1"])
        image_calls = [
            (call.args[0], call.kwargs.get("name"), call.kwargs.get("progress_node_id"))
            for call in module._encode_optional_image_value.call_args_list
        ]
        self.assertEqual(
            [(item[1], item[2]) for item in image_calls],
            [("image1", "node-9"), ("image2", "node-9"), ("image3", "node-9")],
        )
        self.assertIs(image_calls[0][0], image1)
        self.assertIs(image_calls[2][0], image3)
        module._prepare_remote_clip_names.assert_not_called()

    def test_qwen_image_transport_materializes_batched_image_tensor_to_remote_file_refs(self):
        module = _load_remote_clip_module()
        image = torch.tensor(
            [
                [
                    [[0.0, 64 / 255, 128 / 255], [1.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                ],
                [
                    [[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]],
                    [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
                ],
            ],
            dtype=torch.float32,
        )
        uploads = []

        def fake_materialize(filename, payload, *, progress_node_id=None):
            uploads.append((filename, payload, progress_node_id))
            sha256 = hashlib.sha256(payload).hexdigest()
            return {
                "ok": True,
                "name": f"cutlery_remote/qwen/{sha256[:12]}-{filename}",
                "subfolder": "cutlery_remote/qwen",
                "type": "input",
                "sha256": sha256,
                "size": len(payload),
                "materialized": True,
            }

        module._materialize_qwen_png_bytes_to_remote = mock.Mock(side_effect=fake_materialize)
        payload = module._encode_optional_image_value(image, progress_node_id="node-77")

        self.assertEqual(payload["schema"], module.REMOTE_CLIP_IMAGE_FILE_REF_BUNDLE_SCHEMA)
        self.assertEqual(payload["format"], "png")
        self.assertEqual(payload["width"], 2)
        self.assertEqual(payload["height"], 2)
        self.assertEqual([frame["filename"] for frame in payload["frames"]], ["image_0001.png", "image_0002.png"])
        self.assertEqual([frame["type"] for frame in payload["frames"]], ["input", "input"])
        self.assertNotIn("data", payload["frames"][0])
        self.assertEqual([call[0] for call in uploads], ["image_0001.png", "image_0002.png"])
        self.assertEqual([call[2] for call in uploads], ["node-77", "node-77"])

    def test_qwen_file_ref_bundle_decodes_from_remote_input_folder(self):
        module = _load_remote_clip_module()
        image = torch.ones((1, 3, 3, 3), dtype=torch.float32)
        png_bytes = module._image_tensor_to_png_bytes(image[0])
        digest = hashlib.sha256(png_bytes).hexdigest()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_root = Path(temp_dir) / "input"
            image_path = input_root / "cutlery_remote" / "qwen" / "abc-image.png"
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(png_bytes)
            folder_paths = types.SimpleNamespace(
                get_annotated_filepath=lambda name: str(input_root / name),
            )
            bundle = {
                "schema": module.REMOTE_CLIP_IMAGE_FILE_REF_BUNDLE_SCHEMA,
                "format": "png",
                "mime_type": "image/png",
                "width": 3,
                "height": 3,
                "channels": 3,
                "frames": [
                    {
                        "filename": "image_0001.png",
                        "name": "cutlery_remote/qwen/abc-image.png",
                        "subfolder": "cutlery_remote/qwen",
                        "type": "input",
                        "byte_count": len(png_bytes),
                        "sha256": digest,
                    }
                ],
            }

            with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
                decoded = module._decode_optional_image_bundle(bundle)

        self.assertEqual(tuple(decoded.shape), tuple(image.shape))
        self.assertTrue(torch.allclose(decoded, image, atol=1.0 / 255.0))

    def test_qwen_image_transport_still_decodes_legacy_inline_bundle(self):
        module = _load_remote_clip_module()
        image = torch.ones((1, 2, 2, 3), dtype=torch.float32)

        payload = module._encode_image_file_bundle(image)
        decoded = module._decode_optional_image_bundle(payload)

        self.assertEqual(payload["schema"], module.REMOTE_CLIP_IMAGE_BUNDLE_SCHEMA)
        self.assertTrue(torch.allclose(decoded, image, atol=1.0 / 255.0))

    def test_qwen_node_reuses_conditioning_cache_for_identical_inputs(self):
        from cutlery_remote.serialization import encode_value_bundle

        module = _load_remote_clip_module()
        conditioning = [[torch.ones((1, 2, 3), dtype=torch.float32), {"pooled_output": torch.zeros((1, 3), dtype=torch.float32)}]]
        bundle = encode_value_bundle(conditioning)
        node = module.CutleryRemoteTextEncodeQwenImageEditPlus()
        module._encode_optional_image_value = mock.Mock(return_value=None)

        with mock.patch.object(module, "post_remote_qwen_image_edit_plus_encode", return_value=bundle) as post:
            first = node.encode(
                prompt="same prompt",
                text_encoder="remote/qwen.gguf",
                vae_name=module.NONE_CHOICE,
                unique_id="node-100",
            )[0]
            second = node.encode(
                prompt="same prompt",
                text_encoder="remote/qwen.gguf",
                vae_name=module.NONE_CHOICE,
                unique_id="node-100",
            )[0]

        post.assert_called_once()
        self.assertTrue(torch.equal(first[0][0], second[0][0]))
        self.assertIsNot(first[0][0], second[0][0])

    def test_node_forwards_unique_id_to_lora_materialization_progress(self):
        from cutlery_remote.serialization import encode_value_bundle

        module = _load_remote_clip_module()
        bundle = encode_value_bundle([[torch.zeros((1, 1, 1)), {}]])
        module._prepare_remote_clip_names = mock.Mock(return_value=["remote-t5.safetensors"])
        module._prepare_remote_lora_entries = mock.Mock(return_value=[])

        with mock.patch.object(module, "post_remote_clip_encode", return_value=bundle):
            module.CutleryRemoteClipTextEncode().encode(
                prompt="hello world",
                text_encoder="remote-t5.safetensors",
                clip_type="flux",
                lora_chain={"loras": [{"lora_name": "style-a.safetensors", "strength_clip": 0.65}]},
                unique_id="node-42",
            )

        module._prepare_remote_lora_entries.assert_called_once_with(
            [{"lora_name": "style-a.safetensors", "strength_clip": 0.65}],
            progress_node_id="node-42",
        )
        module._prepare_remote_clip_names.assert_called_once_with(["remote-t5.safetensors"], progress_node_id="node-42")

    def test_node_materializes_chain_lora_name_before_remote_encode(self):
        from cutlery_remote.serialization import encode_value_bundle

        module = _load_remote_clip_module()
        module.fetch_remote_clip_inventory = lambda **_kwargs: {"lora_inventory": []}
        module._selected_local_lora_entry = mock.Mock(return_value={"local_name": "host/style-a.safetensors", "sha256": "bbb", "size": 11})
        module._prepare_remote_clip_names = lambda names, **_kwargs: names
        module._materialize_lora_to_remote = mock.Mock(
            return_value={"name": "host/style-a.safetensors", "sha256": "bbb", "size": 11}
        )
        bundle = encode_value_bundle([[torch.ones((1, 1, 1), dtype=torch.float32), {}]])

        with mock.patch.object(module, "post_remote_clip_encode", return_value=bundle) as post:
            module.CutleryRemoteClipTextEncode().encode(
                prompt="portrait",
                text_encoder="remote-t5.safetensors",
                clip_type="flux",
                lora_chain={
                    "loras": [
                        {"lora_name": "host/style-a.safetensors", "strength_model": 0.4, "strength_clip": 0.75}
                    ]
                },
            )

        self.assertEqual(
            post.call_args.args[0]["loras"],
            [{"lora_name": "host/style-a.safetensors", "strength_clip": 0.75}],
        )
        module._materialize_lora_to_remote.assert_called_once()

    def test_node_posts_no_loras_when_lora_chain_is_empty(self):
        from cutlery_remote.serialization import encode_value_bundle

        module = _load_remote_clip_module()
        bundle = encode_value_bundle([[torch.ones((1, 1, 1), dtype=torch.float32), {}]])
        node = module.CutleryRemoteClipTextEncode()
        module._prepare_remote_clip_names = lambda names, **_kwargs: names
        module._prepare_remote_lora_entries = lambda entries, **_kwargs: entries

        with mock.patch.object(module, "post_remote_clip_encode", return_value=bundle) as post:
            node.encode(
                prompt="portrait",
                text_encoder="remote-t5.safetensors",
                clip_type="flux",
                lora_chain={"loras": []},
            )

        self.assertEqual(post.call_args.args[0]["loras"], [])

    def test_node_reuses_cached_remote_conditioning_for_identical_payload(self):
        from cutlery_remote.serialization import encode_value_bundle

        module = _load_remote_clip_module()
        conditioning = [[torch.ones((1, 2, 3), dtype=torch.float32), {"pooled_output": torch.zeros((1, 3))}]]
        bundle = encode_value_bundle(conditioning)
        node = module.CutleryRemoteClipTextEncode()
        module._prepare_remote_clip_names = lambda names, **_kwargs: names
        module._prepare_remote_lora_entries = lambda entries, **_kwargs: entries

        with mock.patch.object(module, "post_remote_clip_encode", return_value=bundle) as post:
            first = node.encode(
                prompt="same prompt",
                text_encoder="remote-t5.safetensors",
                clip_type="flux",
                lora_chain={"loras": []},
                unique_id="node-100",
            )[0]
            second = node.encode(
                prompt="same prompt",
                text_encoder="remote-t5.safetensors",
                clip_type="flux",
                lora_chain={"loras": []},
                unique_id="node-100",
            )[0]

        post.assert_called_once()
        self.assertTrue(torch.equal(first[0][0], second[0][0]))
        self.assertIsNot(first[0][0], second[0][0])

    def test_remote_encode_loads_clip_applies_loras_and_returns_conditioning_bundle(self):
        from cutlery_remote.serialization import decode_value_bundle

        module = _load_remote_clip_module()
        fake_clip = _FakeClip()
        applied = []
        module._load_clip_for_remote_encode = lambda text_encoder, clip_type, device: fake_clip
        module._apply_clip_loras = lambda clip, entries: applied.append(entries) or clip

        payload = module.encode_remote_clip_text(
            {
                "prompt": "remote prompt",
                "text_encoder": "remote-t5.safetensors",
                "clip_type": "flux",
                "loras": [{"lora_name": "style-a.safetensors", "strength_clip": 0.5}],
            }
        )
        decoded = decode_value_bundle(payload["conditioning"])

        self.assertEqual(fake_clip.tokenized, ["remote prompt"])
        self.assertEqual(applied, [[{"lora_name": "style-a.safetensors", "strength_clip": 0.5}]])
        self.assertEqual(decoded[0][1]["tokens"], "remote prompt")

    def test_remote_qwen_image_edit_encode_loads_clip_vae_and_decodes_images(self):
        from cutlery_remote.serialization import decode_value_bundle

        module = _load_remote_clip_module()
        fake_clip = _FakeQwenClip()
        fake_vae = _FakeVAE()
        module._load_clip_for_remote_encode = lambda text_encoder, clip_type, device: fake_clip
        module._load_vae_for_remote_encode = lambda vae_name: fake_vae

        fake_comfy = types.ModuleType("comfy")
        fake_comfy.__path__ = []
        fake_utils = types.ModuleType("comfy.utils")
        fake_utils.common_upscale = lambda samples, _width, _height, _mode, _crop: samples
        fake_comfy.utils = fake_utils
        fake_node_helpers = types.SimpleNamespace(
            conditioning_set_values=lambda conditioning, values, append=True: (
                conditioning[0][1].update(values) or conditioning
            )
        )
        image1 = torch.ones((1, 8, 8, 3), dtype=torch.float32)
        image3 = torch.zeros((1, 4, 4, 3), dtype=torch.float32)

        with mock.patch.dict(
            sys.modules,
            {
                "comfy": fake_comfy,
                "comfy.utils": fake_utils,
                "node_helpers": fake_node_helpers,
            },
        ):
            payload = module.encode_remote_qwen_image_edit_plus_text(
                {
                    "prompt": "paint it blue",
                    "text_encoder": "remote/qwen.gguf",
                    "clip_type": "qwen_image",
                    "vae_name": "qwen-vae.safetensors",
                    "images": {
                        "image1": module._encode_image_file_bundle(image1),
                        "image3": module._encode_image_file_bundle(image3),
                    },
                }
            )

        decoded = decode_value_bundle(payload["conditioning"])
        self.assertTrue(payload["ok"])
        self.assertEqual(len(fake_clip.tokenize_calls), 1)
        call = fake_clip.tokenize_calls[0]
        self.assertIn("Picture 1", call["text"])
        self.assertIn("Picture 3", call["text"])
        self.assertTrue(call["text"].endswith("paint it blue"))
        self.assertEqual(len(call["images"]), 2)
        self.assertIn("<|im_start|>system", call["llama_template"])
        self.assertEqual(len(fake_vae.encoded), 2)
        self.assertEqual(decoded[0][1]["image_count"], 2)
        self.assertEqual(len(decoded[0][1]["reference_latents"]), 2)

    def test_remote_encode_unloads_previous_clip_cache_before_switching_text_encoder(self):
        module = _load_remote_clip_module()
        old_key = ("old-t5.safetensors", "flux", "default")
        new_key = ("remote-t5.safetensors", "flux", "default")
        module._CLIP_CACHE[old_key] = _FakeClip()
        module._ACTIVE_CLIP_KEY = old_key
        cleanup_calls = []
        load_calls = []
        new_clip = _FakeClip()

        def fake_collect_and_empty_cache():
            cleanup_calls.append(list(module._CLIP_CACHE.keys()))

        def fake_load_clip(text_encoder, clip_type, device):
            load_calls.append((text_encoder, clip_type, device, list(module._CLIP_CACHE.keys())))
            module._CLIP_CACHE[new_key] = new_clip
            return new_clip

        module._collect_and_empty_cache = fake_collect_and_empty_cache
        module._load_clip_for_remote_encode = fake_load_clip
        module._apply_clip_loras = lambda clip, entries: clip

        module.encode_remote_clip_text(
            {
                "prompt": "new prompt",
                "text_encoder": "remote-t5.safetensors",
                "clip_type": "flux",
            }
        )

        self.assertEqual(cleanup_calls, [[]])
        self.assertEqual(load_calls, [("remote-t5.safetensors", "flux", "default", [])])
        self.assertNotIn(old_key, module._CLIP_CACHE)
        self.assertEqual(module._ACTIVE_CLIP_KEY, new_key)

    def test_remote_clip_unload_helper_clears_caches_and_active_key(self):
        module = _load_remote_clip_module()
        module._ACTIVE_CLIP_KEY = ("remote-t5.safetensors", "flux", "default")
        module._CLIP_CACHE[module._ACTIVE_CLIP_KEY] = _FakeClip()
        module._LORA_CACHE["/models/loras/style-a.safetensors"] = ({"weights": 1}, {"meta": 1})
        cleanup_calls = []
        module._collect_and_empty_cache = lambda: cleanup_calls.append(True)

        payload = module.unload_remote_clip_cache()

        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["unloaded"], True)
        self.assertEqual(payload["previous_clip_key"], ["remote-t5.safetensors", "flux", "default"])
        self.assertEqual(module._CLIP_CACHE, {})
        self.assertEqual(module._LORA_CACHE, {})
        self.assertIsNone(module._ACTIVE_CLIP_KEY)
        self.assertEqual(cleanup_calls, [True])

    def test_remote_clip_unload_route_requires_auth_and_returns_payload(self):
        routes = _Routes()
        module = _load_remote_clip_module(routes=routes)
        module._ACTIVE_CLIP_KEY = ("remote-t5.safetensors", "flux", "default")
        module._CLIP_CACHE[module._ACTIVE_CLIP_KEY] = _FakeClip()
        module._collect_and_empty_cache = lambda: None

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=True):
            rejected = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/clip/unload")](
                    types.SimpleNamespace(headers={"Authorization": "Bearer wrong"}, json=lambda: {})
                )
            )
            unloaded = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/clip/unload")](
                    types.SimpleNamespace(headers={"Authorization": "Bearer abc123"}, json=lambda: {})
                )
            )

        self.assertEqual(rejected["status"], 401)
        self.assertEqual(unloaded["status"], 200)
        self.assertEqual(unloaded["payload"]["ok"], True)
        self.assertEqual(unloaded["payload"]["unloaded"], True)

    def test_inventory_and_encode_routes_require_auth(self):
        routes = _Routes()
        module = _load_remote_clip_module(routes=routes)
        fake_clip = _FakeClip()
        module._local_clip_inventory = lambda: [
            {"name": "remote-t5.safetensors", "sha256": "aaa", "size": 10},
            {"name": "remote-clip-l.safetensors", "sha256": "bbb", "size": 11},
        ]
        module._local_lora_inventory = lambda: [
            {"name": "style-a.safetensors", "sha256": "aaa", "size": 10},
            {"name": "characters/style-b.safetensors", "sha256": "bbb", "size": 11},
        ]
        module._folder_filename_list = lambda key: {
            "loras": ["style-a.safetensors", "characters/style-b.safetensors"],
            "vae": ["remote-vae.safetensors"],
            "vae_approx": [],
        }.get(key, [])
        module._load_clip_for_remote_encode = lambda text_encoder, clip_type, device: fake_clip
        module._apply_clip_loras = lambda clip, entries: clip

        with (
            mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=True),
            mock.patch.object(module, "_submit_remote_clip_job", new=mock.AsyncMock(return_value=({"ok": True}, 200))),
        ):
            inventory = asyncio.run(
                routes.handlers[("GET", "/cutlery/remote/clip/inventory")](
                    types.SimpleNamespace(headers={"Authorization": "Bearer abc123"})
                )
            )
            with mock.patch.object(module.LOGGER, "warning"):
                rejected = asyncio.run(
                    routes.handlers[("POST", "/cutlery/remote/clip/text-encode")](
                        types.SimpleNamespace(headers={"Authorization": "Bearer wrong"}, json=lambda: {})
                    )
                )
            encoded = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/clip/text-encode")](
                    types.SimpleNamespace(
                        headers={"Authorization": "Bearer abc123"},
                        json=lambda: {"prompt": "route prompt", "text_encoder": "remote-t5.safetensors", "clip_type": "stable_diffusion"},
                    )
                )
            )

        self.assertEqual(inventory["status"], 200)
        self.assertEqual(inventory["payload"]["text_encoders"], ["remote-t5.safetensors", "remote-clip-l.safetensors"])
        self.assertEqual(rejected["status"], 401)
        self.assertEqual(encoded["status"], 200)
        self.assertTrue(encoded["payload"]["ok"])


if __name__ == "__main__":
    unittest.main()
