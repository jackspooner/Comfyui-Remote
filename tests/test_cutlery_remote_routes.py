import asyncio
import base64
import importlib.util
import logging
import os
import sys
import threading
import types
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _Routes:
    def __init__(self):
        self.handlers = {}

    def get(self, path):
        def decorator(fn):
            self.handlers[("GET", path)] = fn
            return fn

        return decorator

    def post(self, path):
        def decorator(fn):
            self.handlers[("POST", path)] = fn
            return fn

        return decorator


def _load_nodes_remote(routes, folder_paths=None):
    package_name = "cutlery_nodes_remote_test_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPO_ROOT)]
    sys.modules[package_name] = package

    server = types.ModuleType("server")
    server.PromptServer = type(
        "PromptServer",
        (),
        {"instance": types.SimpleNamespace(routes=routes)},
    )
    sys.modules["server"] = server

    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web = types.SimpleNamespace(
        json_response=lambda payload, status=200, **_kwargs: {"payload": payload, "status": status}
    )
    sys.modules["aiohttp"] = aiohttp
    if folder_paths is not None:
        sys.modules["folder_paths"] = folder_paths

    module_name = f"{package_name}.nodes_remote"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "nodes_remote.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    module._remote_server_disabled_response = lambda: None

    trusted_origin = "http://127.0.0.1:8189"

    def resolve_test_target(value):
        text = str(value or "").strip().rstrip("/")
        if text not in {
            "renderhost",
            "cutlery://renderhost",
            "127.0.0.1:8189",
            trusted_origin,
        }:
            raise ValueError(f"Cutlery remote target {text!r} is not trusted in this test.")
        return module.TrustedRemoteTarget(
            name="renderhost",
            base_url=trusted_origin,
            canonical="cutlery://renderhost",
            display_label="renderhost",
            copy_host="renderhost",
            copy_root="D:/ComfyUI/models",
        )

    module.resolve_trusted_remote_target = resolve_test_target
    return module


class RemoteRoutesTests(unittest.TestCase):
    def test_disabled_server_gate_runs_before_authorization(self):
        routes = _Routes()
        module = _load_nodes_remote(routes)
        disabled = {"payload": {"ok": False, "code": "remote_server_disabled"}, "status": 403}
        module._remote_server_disabled_response = lambda: disabled
        module._authorized = mock.Mock(side_effect=AssertionError("authorization must not run while disabled"))

        handler = routes.handlers[("POST", "/cutlery/remote/group/run")]
        result = asyncio.run(handler(types.SimpleNamespace()))

        self.assertIs(result, disabled)
        module._authorized.assert_not_called()

    def test_capabilities_route_requires_matching_bearer_token(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            _load_nodes_remote(routes)

            handler = routes.handlers[("GET", "/cutlery/remote/capabilities")]
            with mock.patch.object(logging.getLogger("cutlery.remote.routes"), "warning"):
                rejected = asyncio.run(handler(types.SimpleNamespace(headers={"Authorization": "Bearer wrong"})))
            accepted = asyncio.run(handler(types.SimpleNamespace(headers={"Authorization": "Bearer abc123"})))

        self.assertEqual(rejected["status"], 401)
        self.assertFalse(rejected["payload"]["ok"])
        self.assertEqual(accepted["status"], 200)
        self.assertTrue(accepted["payload"]["ok"])

    def test_node_definitions_route_is_authenticated_and_returns_combo_options(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            module.build_node_definitions_payload = mock.Mock(
                return_value={
                    "ok": True,
                    "requested_count": 1,
                    "definition_count": 1,
                    "definitions": {
                        "CLIPLoader": {
                            "ok": True,
                            "missing": False,
                            "class_type": "CLIPLoader",
                            "source": "INPUT_TYPES",
                            "inputs": {
                                "required": {
                                    "clip_name": {
                                        "kind": "combo",
                                        "type": "COMBO",
                                        "options": ["remote.safetensors"],
                                        "materializable": True,
                                    }
                                },
                                "optional": {},
                                "hidden": {},
                            },
                            "outputs": [],
                            "signature": {"inputs": {}, "outputs": []},
                            "errors": [],
                        }
                    },
                }
            )
            handler = routes.handlers[("POST", "/cutlery/remote/node-definitions")]
            rejected = asyncio.run(handler(types.SimpleNamespace(headers={}, json=lambda: {"class_types": ["CLIPLoader"]})))
            accepted = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={"Authorization": "Bearer abc123"},
                        json=lambda: {"class_types": ["CLIPLoader"]},
                    )
                )
            )

        self.assertEqual(rejected["status"], 401)
        self.assertEqual(accepted["status"], 200)
        clip_definition = accepted["payload"]["nodes"]["CLIPLoader"]
        self.assertTrue(clip_definition["available"])
        self.assertEqual(clip_definition["input_options"]["clip_name"]["options"], ["remote.safetensors"])

    def test_node_definitions_proxy_uses_only_trusted_target_and_shared_token(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            module._post_remote_json = mock.Mock(return_value={"ok": True, "nodes": {}})
            handler = routes.handlers[("POST", "/cutlery/remote/proxy/node-definitions")]

            rejected = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={},
                        json=lambda: {"target": "https://attacker.example:443", "class_types": ["CLIPLoader"]},
                    )
                )
            )
            rejected_loopback = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={},
                        json=lambda: {"target": "127.0.0.1:8190", "class_types": ["CLIPLoader"]},
                    )
                )
            )
            accepted = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={},
                        json=lambda: {"target": "127.0.0.1:8189", "class_types": ["CLIPLoader"]},
                    )
                )
            )

        self.assertEqual(rejected["status"], 403)
        self.assertEqual(rejected_loopback["status"], 403)
        self.assertEqual(accepted["status"], 200)
        module._post_remote_json.assert_called_once_with(
            "http://127.0.0.1:8189",
            "/cutlery/remote/node-definitions",
            {"class_types": ["CLIPLoader"]},
            token="abc123",
            timeout_seconds=30.0,
        )

    def test_node_definitions_proxy_preserves_upstream_bad_request_status(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            module._post_remote_json = mock.Mock(
                side_effect=module.RemoteHttpError(
                    "class_types must be an array.",
                    status_code=400,
                )
            )
            handler = routes.handlers[("POST", "/cutlery/remote/proxy/node-definitions")]

            response = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={},
                        json=lambda: {
                            "target": "127.0.0.1:8189",
                            "class_types": "CLIPLoader",
                        },
                    )
                )
            )

        self.assertEqual(response["status"], 400)
        self.assertIn("class_types must be an array", response["payload"]["error"])

    def test_registry_proxy_rejects_unknown_registry_before_outbound_request(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            module._get_remote_json = mock.Mock(side_effect=AssertionError("unknown registry must not be contacted"))
            module._post_remote_json = mock.Mock(side_effect=AssertionError("unknown registry must not be contacted"))
            handler = routes.handlers[("POST", "/cutlery/remote/proxy/registry")]

            response = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={},
                        json=lambda: {
                            "target": "127.0.0.1:8189",
                            "registry": "/arbitrary/path",
                            "payload": {},
                        },
                    )
                )
            )

        self.assertEqual(response["status"], 400)
        self.assertEqual(response["payload"]["code"], "unknown_registry")
        module._get_remote_json.assert_not_called()
        module._post_remote_json.assert_not_called()

    def test_registry_proxy_rejects_arbitrary_top_level_path_before_outbound_request(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            module._get_remote_json = mock.Mock(side_effect=AssertionError("arbitrary path must not be contacted"))
            handler = routes.handlers[("POST", "/cutlery/remote/proxy/registry")]

            response = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={},
                        json=lambda: {
                            "target": "127.0.0.1:8189",
                            "registry": "remote_clip.choices",
                            "payload": {},
                            "path": "/cutlery/restart",
                        },
                    )
                )
            )

        self.assertEqual(response["status"], 400)
        self.assertEqual(
            response["payload"]["code"],
            "unsupported_registry_request_fields",
        )
        module._get_remote_json.assert_not_called()

    def test_registry_proxy_get_uses_exact_allowlisted_path_and_shared_token(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            module._get_remote_json = mock.Mock(
                return_value={"ok": True, "text_encoders": ["remote-t5.safetensors"]}
            )
            handler = routes.handlers[("POST", "/cutlery/remote/proxy/registry")]

            response = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={},
                        json=lambda: {
                            "target": "127.0.0.1:8189",
                            "registry": "remote_clip.choices",
                            "payload": {},
                        },
                    )
                )
            )

        self.assertEqual(response["status"], 200)
        self.assertEqual(response["payload"]["registry"], "remote_clip.choices")
        self.assertEqual(
            response["payload"]["payload"]["text_encoders"],
            ["remote-t5.safetensors"],
        )
        module._get_remote_json.assert_called_once_with(
            "http://127.0.0.1:8189",
            "/cutlery/remote/clip/choices",
            token="abc123",
            timeout_seconds=30.0,
        )

    def test_registry_proxy_gets_remote_clip_choices_from_target_machine(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            module._get_remote_json = mock.Mock(
                return_value={
                    "ok": True,
                    "text_encoders": ["remote-t5.safetensors"],
                    "clip_types": ["stable_diffusion"],
                    "vaes": ["remote-vae.safetensors"],
                }
            )
            handler = routes.handlers[("POST", "/cutlery/remote/proxy/registry")]

            response = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={},
                        json=lambda: {
                            "target": "127.0.0.1:8189",
                            "registry": "remote_clip.choices",
                            "payload": {},
                        },
                    )
                )
            )

        self.assertEqual(response["status"], 200)
        self.assertEqual(
            response["payload"]["payload"]["text_encoders"],
            ["remote-t5.safetensors"],
        )
        module._get_remote_json.assert_called_once_with(
            "http://127.0.0.1:8189",
            "/cutlery/remote/clip/choices",
            token="abc123",
            timeout_seconds=30.0,
        )

    def test_registry_proxy_rejects_browser_controlled_upstream_urls_and_keys_before_outbound_request(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            module._post_remote_json = mock.Mock(
                side_effect=AssertionError("unsafe registry payload must not be contacted")
            )
            handler = routes.handlers[("POST", "/cutlery/remote/proxy/registry")]

            response = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={},
                        json=lambda: {
                            "target": "127.0.0.1:8189",
                            "registry": "remote_clip.choices",
                            "payload": {
                                "upstream_url": "https://attacker.example",
                                "api_key": "must-not-be-forwarded",
                            },
                        },
                    )
                )
            )
            self.assertEqual(response["status"], 400)
            self.assertEqual(
                response["payload"]["code"],
                "unsupported_registry_payload_fields",
            )

        module._post_remote_json.assert_not_called()

    def test_registry_proxy_rejects_untrusted_target_before_outbound_request(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            module._post_remote_json = mock.Mock(side_effect=AssertionError("untrusted target must not be contacted"))
            handler = routes.handlers[("POST", "/cutlery/remote/proxy/registry")]

            response = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={},
                        json=lambda: {
                            "target": "https://attacker.example:443",
                            "registry": "remote_clip.choices",
                            "payload": {},
                        },
                    )
                )
            )

        self.assertEqual(response["status"], 403)
        module._post_remote_json.assert_not_called()

    def test_blob_routes_upload_and_report_existing_hashes(self):
        routes = _Routes()
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
                module = _load_nodes_remote(routes)
                module.default_blob_store = lambda: module.BlobStore(Path(temp_dir))

                upload_handler = routes.handlers[("POST", "/cutlery/remote/blobs")]
                exists_handler = routes.handlers[("POST", "/cutlery/remote/blobs/exists")]
                payload = b"remote blob payload"
                upload_request = types.SimpleNamespace(
                    headers={"Authorization": "Bearer abc123"},
                    json=lambda: {"data_b64": base64.b64encode(payload).decode("ascii")},
                )
                exists_request = types.SimpleNamespace(
                    headers={"Authorization": "Bearer abc123"},
                    json=lambda: {"hashes": [module.sha256_bytes(payload), "0" * 64]},
                )

                uploaded = asyncio.run(upload_handler(upload_request))
                checked = asyncio.run(exists_handler(exists_request))

        self.assertEqual(uploaded["status"], 200)
        self.assertEqual(uploaded["payload"]["blob"]["hash"], module.sha256_bytes(payload))
        self.assertEqual(checked["status"], 200)
        self.assertEqual(checked["payload"]["present"], [module.sha256_bytes(payload)])
        self.assertEqual(checked["payload"]["missing"], ["0" * 64])

    def test_model_inventory_and_resolve_routes_are_authenticated_and_cheap(self):
        routes = _Routes()
        folder_paths = types.SimpleNamespace(
            get_filename_list=lambda key: {"checkpoints": ["remote.safetensors"]}.get(key, []),
            get_full_path_or_raise=mock.Mock(side_effect=AssertionError("model inventory route must not hash or resolve paths")),
        )
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            _load_nodes_remote(routes, folder_paths=folder_paths)

            inventory_handler = routes.handlers[("GET", "/cutlery/remote/models")]
            resolve_handler = routes.handlers[("POST", "/cutlery/remote/models/resolve")]
            inventory = asyncio.run(
                inventory_handler(
                    types.SimpleNamespace(
                        headers={"Authorization": "Bearer abc123"},
                        query={"model_type": "checkpoints", "include_hashes": "0"},
                    )
                )
            )
            resolved = asyncio.run(
                resolve_handler(
                    types.SimpleNamespace(
                        headers={"Authorization": "Bearer abc123"},
                        json=lambda: {"model_type": "checkpoints", "model_name": "remote.safetensors"},
                    )
                )
            )

        self.assertEqual(inventory["status"], 200)
        self.assertEqual(inventory["payload"]["models"], ["remote.safetensors"])
        self.assertEqual(resolved["status"], 200)
        self.assertTrue(resolved["payload"]["ok"])

    def test_model_inventory_proxy_rejects_untrusted_target_before_outbound_request(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            module._get_remote_json = mock.Mock(side_effect=AssertionError("untrusted target must not be contacted"))
            handler = routes.handlers[("GET", "/cutlery/remote/models")]

            response = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={},
                        query={
                            "target": "https://attacker.example:443",
                            "model_type": "checkpoints",
                            "include_hashes": "0",
                        },
                    )
                )
            )

        self.assertEqual(response["status"], 403)
        self.assertFalse(response["payload"]["ok"])
        module._get_remote_json.assert_not_called()

    def test_model_inventory_proxy_allows_registered_loopback_target(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            module._get_remote_json = mock.Mock(return_value={"ok": True, "models": ["remote.safetensors"]})
            handler = routes.handlers[("GET", "/cutlery/remote/models")]

            response = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={},
                        query={
                            "target": "127.0.0.1:8189",
                            "model_type": "checkpoints",
                            "include_hashes": "0",
                        },
                    )
                )
            )

        self.assertEqual(response["status"], 200)
        module._get_remote_json.assert_called_once()
        self.assertEqual(module._get_remote_json.call_args.args[0], "http://127.0.0.1:8189")
        self.assertEqual(module._get_remote_json.call_args.kwargs["token"], "abc123")

    def test_model_inventory_proxy_rejects_hash_requests_before_remote_call(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            module._get_remote_json = mock.Mock(
                side_effect=AssertionError("browser hash request must not reach the remote")
            )
            handler = routes.handlers[("GET", "/cutlery/remote/models")]

            response = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={},
                        query={
                            "target": "127.0.0.1:8189",
                            "model_type": "checkpoints",
                            "include_hashes": "1",
                        },
                    )
                )
            )

        self.assertEqual(response["status"], 400)
        self.assertIn("not available", response["payload"]["error"])
        module._get_remote_json.assert_not_called()

    def test_model_inventory_proxy_yields_event_loop_while_remote_request_blocks(self):
        routes = _Routes()
        started = threading.Event()
        release = threading.Event()

        def blocking_get(*_args, **_kwargs):
            started.set()
            if not release.wait(timeout=2.0):
                raise TimeoutError("test did not regain control of the event loop")
            return {"ok": True, "models": ["remote.safetensors"]}

        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            module._get_remote_json = blocking_get
            handler = routes.handlers[("GET", "/cutlery/remote/models")]
            request = types.SimpleNamespace(
                headers={},
                query={
                    "target": "127.0.0.1:8189",
                    "model_type": "checkpoints",
                    "include_hashes": "0",
                },
            )

            async def exercise():
                task = asyncio.create_task(handler(request))
                observed = await asyncio.wait_for(
                    asyncio.to_thread(started.wait),
                    timeout=0.5,
                )
                self.assertTrue(observed)
                await asyncio.sleep(0)
                release.set()
                return await asyncio.wait_for(task, timeout=0.5)

            response = asyncio.run(exercise())

        self.assertEqual(response["status"], 200)
        self.assertEqual(response["payload"]["models"], ["remote.safetensors"])

    def test_remote_group_route_decodes_bundled_values_and_delegates_to_group_runner(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            delegated = {}

            async def fake_run_group_body(body):
                delegated.update(body)
                return {"ok": True, "outputs": {"caption": "done"}}, 200

            module._run_remote_group_body = fake_run_group_body
            handler = routes.handlers[("POST", "/cutlery/remote/group/run")]
            result = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={"Authorization": "Bearer abc123"},
                        json=lambda: {
                            "workflow": {"1": {"class_type": "NoOp", "inputs": {}}},
                            "values": {"prompt": module.encode_value_bundle("hello")},
                            "timeout_seconds": 3,
                        },
                    )
                )
            )

        self.assertEqual(result["status"], 200)
        self.assertEqual(delegated["values"], {"prompt": "hello"})
        self.assertEqual(module.decode_value_bundle(result["payload"]["outputs"]["caption"]), "done")

    def test_streamed_group_runner_restores_trellis_progress_adapter_after_worker_error(self):
        routes = _Routes()
        module = _load_nodes_remote(routes)
        adapter_events = []

        class Adapter:
            def __enter__(self):
                adapter_events.append("enter")
                return self

            def __exit__(self, exc_type, _exc_value, _traceback):
                adapter_events.append(exc_type)

        async def failing_workflow(_body):
            raise RuntimeError("worker failed")

        boundary = types.ModuleType(f"{module.__package__}.nodes_wf3_boundary")
        boundary._run_workflow = failing_workflow
        module.TrellisTqdmProgress = lambda prompt_id: Adapter()
        with mock.patch.dict(sys.modules, {boundary.__name__: boundary}):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                asyncio.run(module._run_remote_group_body({"prompt_id": "stream-1"}, stream_trellis_progress=True))

        self.assertEqual(adapter_events, ["enter", RuntimeError])

    def test_normal_group_runner_does_not_enable_trellis_progress_adapter(self):
        routes = _Routes()
        module = _load_nodes_remote(routes)

        class UnexpectedAdapter:
            def __enter__(self):
                raise AssertionError("normal remote execution must not patch Trellis tqdm")

        async def workflow(_body):
            return {"ok": True, "outputs": {}}, 200

        boundary = types.ModuleType(f"{module.__package__}.nodes_wf3_boundary")
        boundary._run_workflow = workflow
        module.TrellisTqdmProgress = UnexpectedAdapter
        with mock.patch.dict(sys.modules, {boundary.__name__: boundary}):
            result = asyncio.run(module._run_remote_group_body({"prompt_id": "normal-1"}))

        self.assertEqual(result, ({"ok": True, "outputs": {}}, 200))

    def test_remote_group_route_rejects_malformed_input_bundle_before_runner(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            runner = mock.AsyncMock(
                side_effect=AssertionError("malformed input must not reach the workflow runner")
            )
            module._run_remote_group_body = runner
            handler = routes.handlers[("POST", "/cutlery/remote/group/run")]
            with mock.patch.object(module.LOGGER, "warning"):
                result = asyncio.run(
                    handler(
                        types.SimpleNamespace(
                            headers={"Authorization": "Bearer abc123"},
                            json=lambda: {
                                "workflow": {"1": {"class_type": "NoOp", "inputs": {}}},
                                "values": {
                                    "payload": {
                                        "schema": module.VALUE_BUNDLE_SCHEMA,
                                        "manifest": {},
                                        "blobs": "not-an-object",
                                    }
                                },
                            },
                        )
                    )
                )

        self.assertEqual(result["status"], 400)
        self.assertFalse(result["payload"]["ok"])
        self.assertIn("Invalid remote group input values", result["payload"]["error"])
        self.assertIn("missing blobs", result["payload"]["error"])
        runner.assert_not_awaited()

    def test_remote_group_route_decodes_each_input_bundle_exactly_once(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            inner_json = {
                "schema": module.VALUE_BUNDLE_SCHEMA,
                "manifest": "user-owned-json",
                "blobs": "not-a-value-bundle",
            }
            captured = {}

            async def fake_run_workflow(body):
                captured.update(body)
                return {
                    "ok": True,
                    "prompt_id": "single-decode",
                    "outputs": {
                        "payload": {
                            "type": "json",
                            "value": body["values"]["payload"],
                        }
                    },
                }, 200

            boundary_module_name = f"{module.__package__}.nodes_wf3_boundary"
            previous = sys.modules.get(boundary_module_name)
            sys.modules[boundary_module_name] = types.SimpleNamespace(
                _run_workflow=fake_run_workflow
            )
            try:
                handler = routes.handlers[("POST", "/cutlery/remote/group/run")]
                result = asyncio.run(
                    handler(
                        types.SimpleNamespace(
                            headers={"Authorization": "Bearer abc123"},
                            json=lambda: {
                                "workflow": {"1": {"class_type": "NoOp", "inputs": {}}},
                                "values": {
                                    "payload": module.encode_value_bundle(inner_json)
                                },
                            },
                        )
                    )
                )
            finally:
                if previous is None:
                    sys.modules.pop(boundary_module_name, None)
                else:
                    sys.modules[boundary_module_name] = previous

        self.assertEqual(result["status"], 200)
        self.assertEqual(captured["values"]["payload"], inner_json)
        self.assertEqual(
            module.decode_value_bundle(result["payload"]["outputs"]["payload"]),
            inner_json,
        )

    def test_remote_group_route_returns_structured_output_transport_failure(self):
        routes = _Routes()
        missing = str(Path(tempfile.gettempdir()) / f"missing-route-{os.getpid()}.mp4")
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)

            async def fake_run_group_body(_body):
                return {
                    "ok": True,
                    "prompt_id": "output-transport-failure",
                    "outputs": {
                        "video": {
                            "type": "video",
                            "value": {
                                "path": missing,
                                "filename": "missing.mp4",
                            },
                        }
                    },
                }, 200

            module._run_remote_group_body = fake_run_group_body
            handler = routes.handlers[("POST", "/cutlery/remote/group/run")]
            with mock.patch.object(module.LOGGER, "exception"):
                result = asyncio.run(
                    handler(
                        types.SimpleNamespace(
                            headers={"Authorization": "Bearer abc123"},
                            json=lambda: {
                                "workflow": {"1": {"class_type": "NoOp", "inputs": {}}},
                                "values": {},
                            },
                        )
                    )
                )

        self.assertEqual(result["status"], 500)
        self.assertFalse(result["payload"]["ok"])
        self.assertEqual(result["payload"]["prompt_id"], "output-transport-failure")
        self.assertIn("Remote group output transport failed", result["payload"]["error"])
        self.assertIn("could not read VIDEO file", result["payload"]["error"])

    def test_remote_group_route_preserves_image_tensor_through_bundle_and_runner_boundary(self):
        routes = _Routes()
        tensor = torch.rand((1, 3, 2, 3))
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            delegated = {}

            async def fake_run_group_body(body):
                delegated.update(body)
                return {
                    "ok": True,
                    "outputs": {
                        "image": {
                            "type": "image",
                            "value": body["values"]["image"],
                        }
                    },
                }, 200

            module._run_remote_group_body = fake_run_group_body
            handler = routes.handlers[("POST", "/cutlery/remote/group/run")]
            result = asyncio.run(
                handler(
                    types.SimpleNamespace(
                        headers={"Authorization": "Bearer abc123"},
                        json=lambda: {
                            "workflow": {"1": {"class_type": "CutleryWorkflowInput", "inputs": {}}},
                            "values": {"image": module.encode_value_bundle(tensor)},
                        },
                    )
                )
            )

        decoded_output = module.decode_value_bundle(result["payload"]["outputs"]["image"])
        self.assertEqual(result["status"], 200)
        self.assertIsInstance(delegated["values"]["image"], torch.Tensor)
        self.assertTrue(torch.equal(delegated["values"]["image"], tensor))
        self.assertTrue(torch.equal(decoded_output, tensor))

    def test_remote_group_image_output_file_refs_are_encoded_as_tensors(self):
        routes = _Routes()
        module = _load_nodes_remote(routes)
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "remote.png"
            from PIL import Image

            Image.new("RGB", (2, 3), (255, 0, 0)).save(image_path)
            encoded = module._encode_remote_outputs(
                {
                    "image": {
                        "type": "image",
                        "value": {"path": str(image_path), "filename": "remote.png", "type": "output"},
                    }
                }
            )
            decoded = module.decode_value_bundle(encoded["image"])

        self.assertEqual(tuple(decoded.shape), (1, 3, 2, 3))
        self.assertTrue(torch.allclose(decoded[0, :, :, 0], torch.ones((3, 2))))

    def test_remote_group_audio_video_output_file_refs_are_encoded_as_media_bytes(self):
        routes = _Routes()
        module = _load_nodes_remote(routes)
        with tempfile.TemporaryDirectory() as temp_dir:
            media_path = Path(temp_dir) / "remote.mp4"
            media_path.write_bytes(b"video bytes")

            encoded = module._encode_remote_outputs(
                {
                    "video": {
                        "type": "video",
                        "value": {"path": str(media_path), "filename": "remote.mp4", "contentType": "video/mp4"},
                    }
                }
            )
            decoded = module.decode_value_bundle(encoded["video"])

        self.assertTrue(decoded["__cutlery_remote_media__"])
        self.assertEqual(decoded["media_type"], "video")
        self.assertEqual(decoded["filename"], "remote.mp4")
        self.assertEqual(decoded["content_type"], "video/mp4")
        self.assertEqual(decoded["data"], b"video bytes")

    def test_remote_group_batched_audio_video_refs_are_encoded_recursively(self):
        routes = _Routes()
        module = _load_nodes_remote(routes)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.wav"
            second = root / "second.wav"
            first.write_bytes(b"first audio")
            second.write_bytes(b"second audio")

            encoded = module._encode_remote_outputs(
                {
                    "audio": {
                        "type": "audio",
                        "value": [
                            {"path": str(first), "filename": first.name, "contentType": "audio/wav"},
                            {"path": str(second), "filename": second.name, "contentType": "audio/wav"},
                        ],
                    }
                }
            )
            decoded = module.decode_value_bundle(encoded["audio"])

        self.assertEqual([item["media_type"] for item in decoded], ["audio", "audio"])
        self.assertEqual([item["filename"] for item in decoded], ["first.wav", "second.wav"])
        self.assertEqual([item["data"] for item in decoded], [b"first audio", b"second audio"])

    def test_remote_group_media_limits_fail_before_read_or_base64_encoding(self):
        routes = _Routes()
        module = _load_nodes_remote(routes)
        with tempfile.TemporaryDirectory() as temp_dir:
            media_path = Path(temp_dir) / "too-large.mp4"
            media_path.write_bytes(b"12345")
            outputs = {
                "video": {
                    "type": "video",
                    "value": {"path": str(media_path), "filename": media_path.name},
                }
            }

            with (
                mock.patch.object(module, "MAX_REMOTE_MEDIA_ITEM_BYTES", 4),
                mock.patch.object(module, "encode_value_bundle") as encode_bundle,
                mock.patch.object(Path, "open", side_effect=AssertionError("file must not be read")),
            ):
                with self.assertRaisesRegex(ValueError, "outbound media item limit"):
                    module._encode_remote_outputs(outputs)

            encode_bundle.assert_not_called()

    def test_remote_group_media_aggregate_limit_fails_before_bundle_base64_encoding(self):
        routes = _Routes()
        module = _load_nodes_remote(routes)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.mp4"
            second = root / "second.mp4"
            first.write_bytes(b"123")
            second.write_bytes(b"456")
            outputs = {
                "first": {
                    "type": "video",
                    "value": {"path": str(first), "filename": first.name},
                },
                "second": {
                    "type": "video",
                    "value": {"path": str(second), "filename": second.name},
                },
            }

            with (
                mock.patch.object(module, "MAX_REMOTE_MEDIA_ITEM_BYTES", 10),
                mock.patch.object(module, "MAX_REMOTE_MEDIA_TOTAL_BYTES", 5),
                mock.patch.object(module, "encode_value_bundle") as encode_bundle,
            ):
                with self.assertRaisesRegex(ValueError, "outbound media total"):
                    module._encode_remote_outputs(outputs)

            encode_bundle.assert_not_called()

    def test_remote_group_unreadable_media_and_image_outputs_fail_loudly(self):
        routes = _Routes()
        module = _load_nodes_remote(routes)
        missing = str(Path(tempfile.gettempdir()) / f"missing-{os.getpid()}.bin")

        with self.assertRaisesRegex(RuntimeError, "could not read VIDEO file"):
            module._encode_remote_outputs(
                {
                    "video": {
                        "type": "video",
                        "value": {"path": missing, "filename": "missing.mp4"},
                    }
                }
            )
        with self.assertRaisesRegex(RuntimeError, "could not read image file"):
            module._encode_remote_outputs(
                {
                    "image": {
                        "type": "image",
                        "value": {"path": missing, "filename": "missing.png"},
                    }
                }
            )

    def test_remote_group_interrupt_cancels_only_supplied_prompt_id(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            cancellation = mock.Mock(
                return_value={
                    "ok": True,
                    "prompt_id": "remote-123",
                    "removed_from_queue": False,
                    "interrupted_running": True,
                    "cancelled": True,
                }
            )
            record_cancellation = mock.Mock(return_value="remote-123")
            boundary_module_name = f"{module.__package__}.nodes_wf3_boundary"
            previous = sys.modules.get(boundary_module_name)
            sys.modules[boundary_module_name] = types.SimpleNamespace(
                cancel_prompt=cancellation,
                record_prompt_cancellation=record_cancellation,
            )
            try:
                handler = routes.handlers[("POST", "/cutlery/remote/group/{remote_prompt_id}/interrupt")]
                response = asyncio.run(
                    handler(
                        types.SimpleNamespace(
                            headers={"Authorization": "Bearer abc123"},
                            match_info={"remote_prompt_id": "remote-123"},
                        )
                    )
                )
            finally:
                if previous is None:
                    sys.modules.pop(boundary_module_name, None)
                else:
                    sys.modules[boundary_module_name] = previous

        self.assertEqual(response["status"], 200)
        self.assertEqual(response["payload"]["remote_prompt_id"], "remote-123")
        self.assertTrue(response["payload"]["interrupted_running"])
        self.assertTrue(response["payload"]["cancellation_recorded"])
        record_cancellation.assert_called_once_with("remote-123")
        cancellation.assert_called_once_with("remote-123")

    def test_canonical_compile_route_returns_unchanged_prompt_without_groups(self):
        routes = _Routes()
        module = _load_nodes_remote(routes)
        prompt = {"1": {"class_type": "NoOp", "inputs": {}}}

        response = asyncio.run(
            routes.handlers[("POST", "/cutlery/remote/compile")](
                types.SimpleNamespace(json=lambda: {"workflow": {"nodes": [], "groups": []}, "prompt": prompt})
            )
        )

        self.assertEqual(response["status"], 200)
        self.assertEqual(response["payload"]["prompt"], prompt)
        self.assertEqual(response["payload"]["targets"], [])

    def test_canonical_compile_route_compiles_nested_remote_group(self):
        routes = _Routes()
        module = _load_nodes_remote(routes)
        workflow = {
            "nodes": [{"id": 30, "type": "inner", "pos": [0, 0], "size": [100, 100]}],
            "links": [],
            "groups": [],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "inner",
                        "nodes": [
                            {
                                "id": 2,
                                "type": "RemoteNode",
                                "pos": [100, 100],
                                "size": [160, 100],
                                "inputs": [{"name": "image", "type": "IMAGE"}],
                                "outputs": [{"name": "image", "type": "IMAGE"}],
                            }
                        ],
                        "links": [],
                        "groups": [{"title": "127.0.0.1:8889", "bounding": [50, 50, 260, 200]}],
                    }
                ]
            },
        }
        prompt = {
            "1": {"class_type": "Source", "inputs": {}},
            "30:2": {"class_type": "RemoteNode", "inputs": {"image": ["1", 0]}},
            "3": {"class_type": "Consumer", "inputs": {"image": ["30:2", 0]}},
        }
        definitions = {
            class_type: {"ok": True, "inputs": {}, "outputs": []}
            for class_type in ("Source", "RemoteNode", "Consumer")
        }

        with (
            mock.patch.object(module, "build_node_definitions_payload", return_value={"definitions": definitions}),
            mock.patch.object(
                module,
                "_post_remote_json_async",
                new=mock.AsyncMock(return_value={"ok": True, "nodes": definitions}),
            ),
            mock.patch.object(module, "find_local_model_by_filename", return_value={"ok": False}),
        ):
            response = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/compile")](
                    types.SimpleNamespace(json=lambda: {"workflow": workflow, "prompt": prompt})
                )
            )

        self.assertEqual(response["status"], 200)
        payload = response["payload"]
        self.assertEqual(payload["targets"], ["127.0.0.1:8889"])
        wrapper_id = payload["remaps"]["30:2"]
        self.assertEqual(payload["prompt"][wrapper_id]["class_type"], "CutleryRemoteGroupValueExecutor")
        self.assertEqual(payload["prompt"]["3"]["inputs"]["image"], [wrapper_id, 0])

    def test_preload_route_uses_the_normal_remote_workflow_runner(self):
        routes = _Routes()
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes)
            captured = {}

            async def runner(body):
                captured.update(body)
                return {"ok": True, "outputs": {}}, 200

            module._run_remote_group_body = runner
            response = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/group/preload")](
                    types.SimpleNamespace(
                        headers={"Authorization": "Bearer abc123"},
                        json=lambda: {"prompt_id": "preload-1", "workflow": {"loader": {}}, "values": {"ignored": 1}},
                    )
                )
            )

        self.assertEqual(response["status"], 200)
        self.assertEqual(captured["values"], {})
        self.assertEqual(captured["prompt_id"], "preload-1")

    def test_batch_model_resolution_returns_verified_identity(self):
        routes = _Routes()
        folder_paths = types.SimpleNamespace(get_full_path_or_raise=lambda _category, _name: __file__)
        with mock.patch.dict(os.environ, {"CUTLERY_REMOTE_TOKEN": "abc123"}, clear=False):
            module = _load_nodes_remote(routes, folder_paths=folder_paths)
            module.resolve_model_name = mock.Mock(return_value={"ok": True, "model_name": "clip.safetensors"})
            module._REMOTE_MODEL_DIGEST_CACHE.digest_for = mock.Mock(return_value=(123, "a" * 64))
            response = asyncio.run(
                routes.handlers[("POST", "/cutlery/remote/models/resolve-batch")](
                    types.SimpleNamespace(
                        headers={"Authorization": "Bearer abc123"},
                        json=lambda: {
                            "models": [
                                {
                                    "category": "text_encoders",
                                    "canonical_name": "clip.safetensors",
                                    "size": 123,
                                    "sha256": "a" * 64,
                                }
                            ]
                        },
                    )
                )
            )

        self.assertEqual(response["status"], 200)
        self.assertTrue(response["payload"]["models"][0]["present"])
        self.assertEqual(response["payload"]["models"][0]["sha256"], "a" * 64)


if __name__ == "__main__":
    unittest.main()
