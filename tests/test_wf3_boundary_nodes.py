from __future__ import annotations

import asyncio
import builtins
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from io import BytesIO
from tempfile import TemporaryDirectory
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_boundary_module(input_dir: str | None = None, output_dir: str | None = None, comfy_load_image=None):
    module_name = "cutlery_nodes_wf3_boundary_test"

    class _Routes:
        def __init__(self):
            self.handlers = {}

        def get(self, _path):
            def decorator(fn):
                self.handlers[("GET", _path)] = fn
                return fn

            return decorator

        def post(self, _path):
            def decorator(fn):
                self.handlers[("POST", _path)] = fn
                return fn

            return decorator

    server_stub = types.ModuleType("server")
    server_stub.PromptServer = type(
        "PromptServer",
        (),
        {"instance": type("PromptServerInstance", (), {"routes": _Routes(), "prompt_queue": None, "number": 0})()},
    )

    aiohttp_stub = types.ModuleType("aiohttp")
    aiohttp_stub.web = types.SimpleNamespace(json_response=lambda payload, **kwargs: {"payload": payload, **kwargs})

    folder_paths_stub = types.ModuleType("folder_paths")
    folder_paths_stub.get_input_directory = lambda: input_dir or "/tmp/cutlery-input"
    folder_paths_stub.get_output_directory = lambda: output_dir or "/tmp/cutlery-output"
    folder_paths_stub.get_save_image_path = lambda prefix, root, width=None, height=None: (
        root,
        prefix,
        1,
        "",
        prefix,
    )

    comfy_nodes_stub = types.ModuleType("nodes")

    class _LoadImage:
        def load_image(self, image):
            if comfy_load_image is not None:
                return comfy_load_image(image)
            return (f"loaded:{image}", None)

    comfy_nodes_stub.LoadImage = _LoadImage

    previous = {name: sys.modules.get(name) for name in ("server", "aiohttp", "folder_paths", "nodes")}
    sys.modules["server"] = server_stub
    sys.modules["aiohttp"] = aiohttp_stub
    sys.modules["folder_paths"] = folder_paths_stub
    sys.modules["nodes"] = comfy_nodes_stub

    try:
        spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "nodes_wf3_boundary.py")
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def _execution_stub(validate_result=(True, None, [], {}), *, sensitive_keys=()):
    execution_stub = types.ModuleType("execution")

    async def validate_prompt(_prompt_id, _workflow, _partial_execution_targets):
        if isinstance(validate_result, BaseException):
            raise validate_result
        return validate_result

    execution_stub.validate_prompt = validate_prompt
    execution_stub.SENSITIVE_EXTRA_DATA_KEYS = sensitive_keys
    return execution_stub


def _image_input_workflow():
    return {
        "1": {
            "class_type": "CutleryWorkflowInput",
            "inputs": {
                "ports_json": '[{"name":"image","type":"image","required":true}]',
            },
        }
    }


def _editor_remote_workflow():
    return {
        "nodes": [
            {
                "id": 1,
                "type": "CutleryWorkflowInput",
                "pos": [-200, 100],
                "size": [160, 100],
                "widgets_values": ['[{"name":"prompt","type":"string"}]'],
                "outputs": [{"name": "value_1", "type": "STRING", "links": [1]}],
            },
            {
                "id": 2,
                "type": "RemoteStringNode",
                "pos": [100, 100],
                "size": [160, 100],
                "inputs": [{"name": "text", "type": "STRING", "link": 1}],
                "outputs": [{"name": "text", "type": "STRING", "links": [2]}],
            },
            {
                "id": 3,
                "type": "CutleryWorkflowOutput",
                "pos": [500, 100],
                "size": [160, 100],
                "widgets_values": ['[{"name":"result","type":"string"}]'],
                "inputs": [{"name": "value_1", "type": "STRING", "link": 2}],
            },
        ],
        "links": [
            [1, 1, 0, 2, 0, "STRING"],
            [2, 2, 0, 3, 0, "STRING"],
        ],
        "groups": [{"title": "127.0.0.1:8889", "bounding": [50, 50, 260, 200]}],
    }


class WorkflowBoundaryNodeTests(unittest.TestCase):
    def test_input_node_reads_configured_typed_ports_from_wf3_inputs(self):
        image_loads = []

        def load_image(image):
            image_loads.append(image)
            return ("image-tensor", None)

        module = _load_boundary_module(comfy_load_image=load_image)
        ports_json = """[
            {"name": "prompt", "type": "string"},
            {"name": "steps", "type": "int"},
            {"name": "cfg", "type": "float"},
            {"name": "enabled", "type": "bool"},
            {"name": "init_image", "type": "image"}
        ]"""

        outputs = module.CutleryWorkflowInput().read(
            ports_json=ports_json,
            wf3={
                "inputs": {
                    "prompt": "hello",
                    "steps": "12",
                    "cfg": "7.5",
                    "enabled": "false",
                    "init_image": "wf3/request/image.png",
                }
            },
        )

        self.assertEqual(outputs[:5], ("hello", 12, 7.5, False, "image-tensor"))
        self.assertEqual(image_loads, ["wf3/request/image.png"])

    def test_boolean_alias_is_normalized_to_bool_port_type(self):
        module = _load_boundary_module()

        ports = module.parse_port_specs('[{"name":"flag","type":"boolean","default":true}]')
        outputs = module.CutleryWorkflowInput().read(
            ports_json='[{"name":"flag","type":"boolean","default":true}]',
            wf3={"inputs": {}},
        )

        self.assertEqual(ports[0]["type"], "bool")
        self.assertEqual(ports[0]["socket_type"], "BOOLEAN")
        self.assertIs(outputs[0], True)

    def test_port_specs_reject_more_than_sixty_four_ports_instead_of_truncating(self):
        module = _load_boundary_module()
        ports = [
            {"name": f"port_{index}", "type": "string"}
            for index in range(module.MAX_WF3_PORTS + 1)
        ]

        with self.assertRaisesRegex(ValueError, "maximum is 64"):
            module.parse_port_specs(ports)

    def test_json_port_preserves_schema_and_coerces_json_values(self):
        module = _load_boundary_module()
        ports_json = """[
            {
                "name": "settings",
                "type": "json",
                "required": true,
                "schema": {
                    "type": "object",
                    "required": ["mode"],
                    "properties": {
                        "mode": { "type": "string" },
                        "threshold": { "type": "number" }
                    },
                    "additionalProperties": false
                }
            }
        ]"""

        ports = module.parse_port_specs(ports_json)
        outputs = module.CutleryWorkflowInput().read(
            ports_json=ports_json,
            wf3={"inputs": {"settings": '{"mode":"fast","threshold":0.75}'}},
        )

        self.assertEqual(ports[0]["type"], "json")
        self.assertEqual(ports[0]["socket_type"], "*")
        self.assertEqual(ports[0]["schema"]["required"], ["mode"])
        self.assertEqual(outputs[0], {"mode": "fast", "threshold": 0.75})

    def test_json_port_accepts_scalar_strings_and_markdown_fenced_json(self):
        module = _load_boundary_module()

        scalar_outputs = module.CutleryWorkflowInput().read(
            ports_json='[{"name":"prompt","type":"json"}]',
            wf3={"inputs": {"prompt": "plain prompt text"}},
        )
        fenced_outputs = module.CutleryWorkflowInput().read(
            ports_json='[{"name":"prompt","type":"json"}]',
            wf3={"inputs": {"prompt": '```json\n{"mode":"fast"}\n```'}},
        )

        self.assertEqual(scalar_outputs[0], "plain prompt text")
        self.assertEqual(fenced_outputs[0], {"mode": "fast"})

    def test_json_port_accepts_openai_style_schema_wrapper(self):
        module = _load_boundary_module()
        ports = module.parse_port_specs(
            [
                {
                    "name": "settings",
                    "type": "json",
                    "schema": {
                        "name": "settings_schema",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "required": ["mode"],
                            "properties": {"mode": {"type": "string"}},
                            "additionalProperties": False,
                        },
                    },
                }
            ]
        )

        self.assertEqual(ports[0]["schema"]["type"], "object")
        self.assertEqual(ports[0]["schema"]["required"], ["mode"])
        self.assertNotIn("strict", ports[0]["schema"])

    def test_remote_boundary_runtime_and_lora_chain_types_pass_through_native_values(self):
        module = _load_boundary_module()
        latent = {"samples": object()}
        conditioning = [[object(), {"pooled_output": object()}]]
        mask = object()
        lora_chain = {
            "loras": [
                {
                    "lora_name": "styles/character.safetensors",
                    "strength_model": 0.8,
                    "strength_clip": 0.6,
                }
            ]
        }
        ports_json = """[
            {"name": "latent", "type": "latent"},
            {"name": "conditioning", "type": "conditioning"},
            {"name": "mask", "type": "mask"},
            {"name": "lora_chain", "type": "cutlery_lora_chain"}
        ]"""

        ports = module.parse_port_specs(ports_json)
        input_values = module.CutleryWorkflowInput().read(
            ports_json=ports_json,
            wf3={
                "inputs": {
                    "latent": latent,
                    "conditioning": conditioning,
                    "mask": mask,
                    "lora_chain": lora_chain,
                }
            },
        )
        output_events = module.CutleryWorkflowOutput().emit(
            ports_json=ports_json,
            value_1=latent,
            value_2=conditioning,
            value_3=mask,
            value_4=lora_chain,
        )["ui"]["wf3"]

        self.assertEqual(
            [port["socket_type"] for port in ports],
            ["LATENT", "CONDITIONING", "MASK", "CUTLERY_LORA_CHAIN"],
        )
        self.assertIs(input_values[0], latent)
        self.assertIs(input_values[1], conditioning)
        self.assertIs(input_values[2], mask)
        self.assertIs(input_values[3], lora_chain)
        self.assertIs(output_events[0]["value"], latent)
        self.assertIs(output_events[1]["value"], conditioning)
        self.assertIs(output_events[2]["value"], mask)
        self.assertEqual(output_events[3]["value"], lora_chain)

    def test_input_node_uses_connected_passthrough_values_when_wf3_input_is_missing(self):
        module = _load_boundary_module()
        ports_json = '[{"name":"prompt","type":"string","required":true},{"name":"steps","type":"int","default":30}]'

        outputs = module.CutleryWorkflowInput().read(
            ports_json=ports_json,
            wf3={"inputs": {}},
            value_1="debug prompt",
            value_2="12",
        )

        self.assertEqual(outputs[:2], ("debug prompt", 12))

    def test_input_node_prefers_wf3_values_over_connected_passthrough_values(self):
        module = _load_boundary_module()
        ports_json = '[{"name":"prompt","type":"string","required":true}]'

        outputs = module.CutleryWorkflowInput().read(
            ports_json=ports_json,
            wf3={"inputs": {"prompt": "request prompt"}},
            value_1="debug prompt",
        )

        self.assertEqual(outputs[0], "request prompt")

    def test_input_node_cache_identity_ignores_values_declared_by_other_boundary_nodes(self):
        module = _load_boundary_module()
        ports_json = '[{"name":"lora_chain","type":"cutlery_lora_chain","required":true}]'
        lora_chain = [{"name": "style.safetensors", "strength_model": 0.8, "strength_clip": 0.7}]

        original = module.CutleryWorkflowInput.IS_CHANGED(
            ports_json,
            wf3={"inputs": {"image": "first.png", "lora_chain": lora_chain}},
        )
        unrelated_changed = module.CutleryWorkflowInput.IS_CHANGED(
            ports_json,
            wf3={"inputs": {"image": "second.png", "lora_chain": lora_chain}},
        )
        declared_changed = module.CutleryWorkflowInput.IS_CHANGED(
            ports_json,
            wf3={"inputs": {"image": "second.png", "lora_chain": [{**lora_chain[0], "strength_clip": 0.5}]}},
        )

        self.assertEqual(original, unrelated_changed)
        self.assertNotEqual(original, declared_changed)

    def test_output_node_emits_configured_typed_values_by_name(self):
        module = _load_boundary_module()
        ports_json = """[
            {"name": "caption", "type": "string"},
            {"name": "seed", "type": "int"},
            {"name": "score", "type": "float"},
            {"name": "accepted", "type": "bool"}
        ]"""

        result = module.CutleryWorkflowOutput().emit(
            ports_json=ports_json,
            value_1="done",
            value_2=123,
            value_3=0.75,
            value_4="true",
        )

        self.assertEqual(
            result["ui"]["wf3"],
            [
                {"key": "caption", "type": "string", "value": "done"},
                {"key": "seed", "type": "int", "value": 123},
                {"key": "score", "type": "float", "value": 0.75},
                {"key": "accepted", "type": "bool", "value": True},
            ],
        )

    def test_wildcard_json_output_rejects_runtime_custom_objects_with_exact_path(self):
        module = _load_boundary_module()

        class RuntimeOnlyValue:
            pass

        with self.assertRaisesRegex(
            TypeError,
            r"Workflow output 'payload'\['items'\]\[1\].*RuntimeOnlyValue",
        ):
            module.CutleryWorkflowOutput().emit(
                ports_json='[{"name":"payload","type":"json"}]',
                value_1={"items": ["ok", RuntimeOnlyValue()]},
            )

    def test_wildcard_json_output_rejects_nonfinite_numbers_and_non_string_keys(self):
        module = _load_boundary_module()
        cases = (
            ({"score": float("nan")}, ValueError, r"\['score'\].*non-finite"),
            ({"items": {1: "bad"}}, TypeError, r"\['items'\].*key 1.*int"),
            (("tuple",), TypeError, r"unsupported value type tuple"),
        )

        for value, error_type, expected in cases:
            with self.subTest(value=value):
                with self.assertRaisesRegex(error_type, expected):
                    module.CutleryWorkflowOutput().emit(
                        ports_json='[{"name":"payload","type":"json"}]',
                        value_1=value,
                    )

    def test_output_node_saves_audio_and_video_outputs(self):
        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            module = _load_boundary_module(output_dir=str(output_dir))
            saved_audio = output_dir / "cutlery" / "speech_00001.flac"

            class FakeFolderType:
                output = "output"

            class FakeAudioSaveHelper:
                @staticmethod
                def save_audio(audio, filename_prefix, folder_type, cls, format="flac", quality="128k"):
                    self.assertEqual(audio, {"waveform": "tensor", "sample_rate": 44100})
                    self.assertEqual(filename_prefix, "cutlery/speech")
                    self.assertEqual(folder_type, FakeFolderType.output)
                    saved_audio.parent.mkdir(parents=True, exist_ok=True)
                    saved_audio.write_bytes(b"flac")
                    return [{"filename": "speech_00001.flac", "subfolder": "cutlery", "type": "output"}]

            class FakeVideo:
                def __init__(self):
                    self.calls = []

                def get_dimensions(self):
                    return (640, 360)

                def save_to(self, path, format=None, codec=None, metadata=None):
                    self.calls.append({"path": path, "format": format, "codec": codec, "metadata": metadata})
                    Path(path).parent.mkdir(parents=True, exist_ok=True)
                    Path(path).write_bytes(b"mp4")

            fake_video = FakeVideo()
            fake_ui = types.SimpleNamespace(AudioSaveHelper=FakeAudioSaveHelper)
            fake_io = types.SimpleNamespace(FolderType=FakeFolderType)
            fake_types = types.SimpleNamespace(
                VideoContainer=types.SimpleNamespace(MP4="mp4"),
                VideoCodec=types.SimpleNamespace(H264="h264"),
            )
            fake_latest = types.SimpleNamespace(ui=fake_ui, io=fake_io, Types=fake_types)

            with mock.patch.dict(sys.modules, {"comfy_api.latest": fake_latest}):
                result = module.CutleryWorkflowOutput().emit(
                    ports_json='[{"name":"speech","type":"audio"},{"name":"clip","type":"video"}]',
                    value_1={"waveform": "tensor", "sample_rate": 44100},
                    value_2=fake_video,
                )

            self.assertEqual(fake_video.calls[0]["format"], "mp4")
            self.assertEqual(fake_video.calls[0]["codec"], "h264")
            video_path = Path(fake_video.calls[0]["path"])
            self.assertTrue(video_path.exists())
            self.assertEqual(
                result["ui"]["wf3"],
                [
                    {
                        "key": "speech",
                        "type": "audio",
                        "value": {
                            "filename": "speech_00001.flac",
                            "subfolder": "cutlery",
                            "type": "output",
                            "path": str(saved_audio),
                            "contentType": "audio/flac",
                        },
                    },
                    {
                        "key": "clip",
                        "type": "video",
                        "value": {
                            "filename": "clip_00001_.mp4",
                            "subfolder": "cutlery",
                            "type": "output",
                            "path": str(video_path),
                            "contentType": "video/mp4",
                        },
                    },
                ],
            )

    def test_output_node_normalizes_existing_audio_and_video_file_references(self):
        module = _load_boundary_module()

        result = module.CutleryWorkflowOutput().emit(
            ports_json='[{"name":"speech","type":"audio"},{"name":"clip","type":"video"}]',
            value_1="C:/ComfyUI/output/audio/speech.wav",
            value_2={"filename": "clip.mp4", "subfolder": "video", "type": "output"},
        )

        expected_audio_path = str(Path("C:/ComfyUI/output/audio/speech.wav").resolve())
        expected_video_path = str((Path("/tmp/cutlery-output") / "video" / "clip.mp4").resolve())
        self.assertEqual(
            result["ui"]["wf3"],
            [
                {
                    "key": "speech",
                    "type": "audio",
                    "value": {
                        "filename": "speech.wav",
                        "subfolder": "",
                        "type": "output",
                        "path": expected_audio_path,
                        "contentType": "audio/wav",
                    },
                },
                {
                    "key": "clip",
                    "type": "video",
                    "value": {
                        "filename": "clip.mp4",
                        "subfolder": "video",
                        "type": "output",
                        "path": expected_video_path,
                        "contentType": "video/mp4",
                    },
                },
            ],
        )

    def test_materialize_run_values_copies_image_paths_to_request_input_folder(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.png"
            source.write_bytes(b"png")
            input_dir = root / "input"
            input_dir.mkdir()

            module = _load_boundary_module(input_dir=str(input_dir))
            ports = module.parse_port_specs([{"name": "prompt", "type": "string"}, {"name": "image", "type": "image"}])

            normalized = module.materialize_run_values(
                {"prompt": "hello", "image": str(source)},
                ports=ports,
                input_dir=input_dir,
                request_id="req-1",
            )

            self.assertEqual(normalized["prompt"], "hello")
            self.assertEqual(normalized["image"], "cutlery/req-1/source.png")
            self.assertEqual((input_dir / "cutlery" / "req-1" / "source.png").read_bytes(), b"png")

    def test_materialize_run_values_preserves_native_image_tensors(self):
        with TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            input_dir.mkdir()
            module = _load_boundary_module(input_dir=str(input_dir))
            tensor = torch.zeros((1, 2, 3, 3))

            normalized = module.materialize_run_values(
                {"image": tensor},
                ports=module.parse_port_specs([{"name": "image", "type": "image"}]),
                input_dir=input_dir,
                request_id="tensor-image",
            )

            self.assertIs(normalized["image"], tensor)
            self.assertFalse(module._request_input_dir(input_dir, "tensor-image").exists())

    def test_materialize_run_values_copies_video_paths_to_request_input_folder(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.mp4"
            source.write_bytes(b"mp4")
            input_dir = root / "input"
            input_dir.mkdir()

            module = _load_boundary_module(input_dir=str(input_dir))
            ports = module.parse_port_specs([{"name": "clip", "type": "video"}])

            normalized = module.materialize_run_values(
                {"clip": str(source)},
                ports=ports,
                input_dir=input_dir,
                request_id="req-1",
            )

            target = input_dir / "cutlery" / "req-1" / "source.mp4"
            self.assertEqual(normalized["clip"], str(target.resolve()))
            self.assertEqual(target.read_bytes(), b"mp4")

    def test_materialize_run_values_stringifies_json_inputs_before_queueing(self):
        with TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            input_dir.mkdir()

            module = _load_boundary_module(input_dir=str(input_dir))
            ports = module.parse_port_specs([{"name": "spec", "type": "json"}])
            value = {
                "nodes": [{"id": 1, "type": "NotAWorkflow"}],
                "links": [],
                "prompt": {"description": "workflow-shaped user JSON"},
            }

            normalized = module.materialize_run_values(
                {"spec": value},
                ports=ports,
                input_dir=input_dir,
                request_id="req-json",
            )
            outputs = module.CutleryWorkflowInput().read(
                ports_json='[{"name":"spec","type":"json"}]',
                wf3={"inputs": normalized},
            )

            self.assertIsInstance(normalized["spec"], str)
            self.assertEqual(outputs[0], value)

    def test_materialize_run_values_stringifies_json_scalar_strings_before_queueing(self):
        with TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            input_dir.mkdir()

            module = _load_boundary_module(input_dir=str(input_dir))
            ports = module.parse_port_specs([{"name": "prompt", "type": "json"}])

            normalized = module.materialize_run_values(
                {"prompt": "plain prompt text"},
                ports=ports,
                input_dir=input_dir,
                request_id="req-json-string",
            )
            outputs = module.CutleryWorkflowInput().read(
                ports_json='[{"name":"prompt","type":"json"}]',
                wf3={"inputs": normalized},
            )

            self.assertEqual(normalized["prompt"], '"plain prompt text"')
            self.assertEqual(outputs[0], "plain prompt text")

    def test_materialize_run_values_downloads_video_urls_to_request_input_folder(self):
        with TemporaryDirectory() as tmp:
            input_dir = Path(tmp) / "input"
            input_dir.mkdir()

            module = _load_boundary_module(input_dir=str(input_dir))
            ports = module.parse_port_specs([{"name": "clip", "type": "video"}])

            class FakeResponse:
                headers = {"Content-Type": "video/webm"}

                def __init__(self):
                    self.body = BytesIO(b"webm")

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def read(self, size=-1):
                    return self.body.read(size)

            opened = []

            def open_url(request, timeout=0):
                opened.append((request.full_url, timeout))
                return FakeResponse()

            normalized = module.materialize_run_values(
                {"clip": "https://example.test/rendered/video"},
                ports=ports,
                input_dir=input_dir,
                request_id="req-url",
                open_url=open_url,
            )

            target = input_dir / "cutlery" / "req-url" / "video.webm"
            self.assertEqual(opened, [("https://example.test/rendered/video", 60)])
            self.assertEqual(normalized["clip"], str(target.resolve()))
            self.assertEqual(target.read_bytes(), b"webm")

    def test_input_node_coerces_video_value_to_native_video(self):
        module = _load_boundary_module()

        class FakeVideo:
            def __init__(self, path):
                self.path = path

        fake_latest = types.SimpleNamespace(InputImpl=types.SimpleNamespace(VideoFromFile=FakeVideo))

        with mock.patch.dict(sys.modules, {"comfy_api.latest": fake_latest}):
            outputs = module.CutleryWorkflowInput().read(
                ports_json='[{"name":"clip","type":"video"}]',
                wf3={"inputs": {"clip": "C:/videos/source.mp4"}},
            )

        self.assertIsInstance(outputs[0], FakeVideo)
        self.assertEqual(outputs[0].path, "C:/videos/source.mp4")

    def test_editor_workflow_json_converts_to_api_prompt_json(self):
        module = _load_boundary_module()
        input_ports = '[{"name":"prompt","type":"string"}]'
        output_ports = '[{"name":"caption","type":"string"}]'

        prompt = module.workflow_to_api_prompt(
            {
                "nodes": [
                    {
                        "id": 1,
                        "type": "CutleryWorkflowInput",
                        "widgets_values": [input_ports],
                        "inputs": [{"name": "prompt", "link": 9}],
                        "outputs": [{"name": "value_1", "links": [10]}],
                    },
                    {
                        "id": 3,
                        "type": "SomePromptSource",
                        "widgets_values": ["debug prompt"],
                        "inputs": [],
                        "outputs": [{"name": "text", "links": [9]}],
                    },
                    {
                        "id": 4,
                        "type": "Reroute",
                        "inputs": [{"name": "", "link": 9}],
                        "outputs": [{"name": "", "links": [11]}],
                    },
                    {
                        "id": 5,
                        "type": "Note",
                        "inputs": [],
                        "outputs": [],
                        "widgets_values": ["Documentation only"],
                    },
                    {
                        "id": 2,
                        "type": "CutleryWorkflowOutput",
                        "widgets_values": [output_ports],
                        "inputs": [{"name": "caption", "link": 10}],
                        "outputs": [],
                    },
                ],
                "links": [
                    [9, 3, 0, 4, 0, "STRING"],
                    [11, 4, 0, 1, 0, "STRING"],
                    [10, 1, 0, 2, 0, "STRING"],
                ],
            }
        )

        self.assertEqual(set(prompt), {"1", "2", "3"})
        self.assertEqual(prompt["1"]["class_type"], "CutleryWorkflowInput")
        self.assertEqual(prompt["1"]["inputs"]["ports_json"], input_ports)
        self.assertEqual(prompt["1"]["inputs"]["value_1"], ["3", 0])
        self.assertEqual(prompt["2"]["class_type"], "CutleryWorkflowOutput")
        self.assertEqual(prompt["2"]["inputs"]["ports_json"], output_ports)
        self.assertEqual(prompt["2"]["inputs"]["value_1"], ["1", 0])

    def test_editor_widget_values_follow_widget_metadata_and_skip_seed_control(self):
        module = _load_boundary_module()

        class TestSampler:
            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {
                        "model": ("MODEL",),
                        "width": ("INT",),
                        "seed": ("INT", {"control_after_generate": True}),
                        "steps": ("INT",),
                        "sampler_name": (["euler", "heun"],),
                    },
                    "optional": {
                        "reference": ("IMAGE",),
                        "fit_mode": (["fit", "crop"],),
                    },
                }

        workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "TestSampler",
                    "inputs": [
                        {"name": "model", "type": "MODEL", "link": 1},
                        {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": 2},
                        {"name": "seed", "type": "INT", "widget": {"name": "seed"}, "link": None},
                        {"name": "steps", "type": "INT", "widget": {"name": "steps"}, "link": None},
                        {"name": "reference", "type": "IMAGE", "link": None},
                        {"name": "fit_mode", "type": "COMBO", "widget": {"name": "fit_mode"}, "link": None},
                        {"name": "sampler_name", "type": "COMBO", "widget": {"name": "sampler_name"}, "link": None},
                    ],
                    "widgets_values": [1024, 178, "increment", 10, "fit", "euler"],
                },
                {"id": 2, "type": "Source", "inputs": [], "outputs": [{"links": [1]}, {"links": [2]}]},
            ],
            "links": [[1, 2, 0, 1, 0, "MODEL"], [2, 2, 1, 1, 1, "INT"]],
        }

        with mock.patch.object(module, "_node_class", side_effect=lambda name: TestSampler if name == "TestSampler" else None):
            prompt = module.workflow_to_api_prompt(workflow)

        self.assertEqual(
            prompt["1"]["inputs"],
            {
                "model": ["2", 0],
                "width": ["2", 1],
                "seed": 178,
                "steps": 10,
                "fit_mode": "fit",
                "sampler_name": "euler",
            },
        )

    def test_normalize_workflow_materializes_save_image_for_unsaved_image_output(self):
        module = _load_boundary_module()
        workflow = {
            "1": {
                "class_type": "ImageSource",
                "inputs": {},
            },
            "2": {
                "class_type": "CutleryWorkflowOutput",
                "inputs": {
                    "ports_json": '[{"name":"preview","type":"image"}]',
                    "value_1": ["1", 0],
                },
            },
        }

        prompt = module.normalize_workflow_json(workflow)

        self.assertEqual(set(prompt), {"1", "2", "3"})
        self.assertEqual(
            prompt["3"],
            {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["1", 0],
                    "filename_prefix": "cutlery/preview",
                },
                "_meta": {
                    "title": "Save Workflow Output: preview",
                    "cutlery_materialized": "workflow_output_image",
                },
            },
        )
        self.assertEqual(set(workflow), {"1", "2"})

    def test_normalize_api_workflow_maps_friendly_output_socket_name(self):
        module = _load_boundary_module()
        workflow = {
            "1": {"class_type": "ImageSource", "inputs": {}},
            "2": {
                "class_type": "CutleryWorkflowOutput",
                "inputs": {
                    "ports_json": '[{"name":"result","type":"image"}]',
                    "result": ["1", 0],
                },
            },
        }

        prompt = module.normalize_workflow_json(workflow)

        self.assertEqual(prompt["2"]["inputs"]["value_1"], ["1", 0])
        self.assertNotIn("result", prompt["2"]["inputs"])
        self.assertIn("3", prompt)
        self.assertEqual(workflow["2"]["inputs"]["result"], ["1", 0])

    def test_normalize_workflow_reuses_save_image_on_the_same_source(self):
        module = _load_boundary_module()
        workflow = {
            "1": {
                "class_type": "ImageSource",
                "inputs": {},
            },
            "2": {
                "class_type": "CutleryWorkflowOutput",
                "inputs": {
                    "ports_json": '[{"name":"first","type":"image"},{"name":"second","type":"image"}]',
                    "value_1": ["1", 0],
                    "value_2": ["1", 0],
                },
            },
            "3": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["1", 0],
                    "filename_prefix": "existing",
                },
            },
        }

        prompt = module.normalize_workflow_json(workflow)

        self.assertIs(prompt, workflow)
        self.assertEqual(set(prompt), {"1", "2", "3"})

    def test_normalize_editor_workflow_materializes_only_missing_image_sources(self):
        module = _load_boundary_module()
        ports_json = '[{"name":"saved","type":"image"},{"name":"missing","type":"image"},{"name":"caption","type":"string"}]'
        workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "ImageSource",
                    "inputs": [],
                    "outputs": [
                        {"name": "first", "links": [10, 12]},
                        {"name": "second", "links": [11]},
                    ],
                },
                {
                    "id": 2,
                    "type": "CutleryWorkflowOutput",
                    "widgets_values": [ports_json],
                    "inputs": [
                        {"name": "saved", "link": 10},
                        {"name": "missing", "link": 11},
                        {"name": "caption", "link": None},
                    ],
                    "outputs": [],
                },
                {
                    "id": 3,
                    "type": "SaveImage",
                    "widgets_values": ["existing"],
                    "inputs": [{"name": "images", "link": 12}],
                    "outputs": [],
                },
            ],
            "links": [
                [10, 1, 0, 2, 0, "IMAGE"],
                [11, 1, 1, 2, 1, "IMAGE"],
                [12, 1, 0, 3, 0, "IMAGE"],
            ],
        }

        prompt = module.normalize_workflow_json(workflow)

        save_nodes = [node for node in prompt.values() if node["class_type"] == "SaveImage"]
        self.assertEqual(len(save_nodes), 2)
        self.assertIn(
            {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["1", 1],
                    "filename_prefix": "cutlery/missing",
                },
                "_meta": {
                    "title": "Save Workflow Output: missing",
                    "cutlery_materialized": "workflow_output_image",
                },
            },
            save_nodes,
        )

    def test_run_workflow_includes_materialized_save_in_partial_execution_targets(self):
        module = _load_boundary_module()
        workflow = {
            "1": {"class_type": "ImageSource", "inputs": {}},
            "2": {
                "class_type": "CutleryWorkflowOutput",
                "inputs": {
                    "ports_json": '[{"name":"preview","type":"image"}]',
                    "value_1": ["1", 0],
                },
            },
        }
        captured = {}

        async def validate_prompt(_prompt_id, prompt, partial_execution_targets):
            captured["prompt"] = prompt
            captured["targets"] = partial_execution_targets
            return True, "", partial_execution_targets, {}

        class FakeQueue:
            def put(self, item):
                captured["queued"] = item

            def get_history(self, prompt_id=None):
                return {
                    prompt_id: {
                        "outputs": {},
                        "status": {"status_str": "success"},
                    }
                }

        module.PromptServer.instance.prompt_queue = FakeQueue()
        execution_stub = types.SimpleNamespace(
            validate_prompt=validate_prompt,
            SENSITIVE_EXTRA_DATA_KEYS=(),
        )

        with mock.patch.dict(sys.modules, {"execution": execution_stub}):
            result, status = asyncio.run(
                module._run_workflow(
                    {
                        "prompt_id": "partial-image-output",
                        "workflow": workflow,
                        "partial_execution_targets": ["2"],
                    }
                )
            )

        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        self.assertEqual(captured["targets"], ["2", "3"])
        self.assertEqual(captured["queued"][4], ["2", "3"])
        self.assertEqual(captured["prompt"]["3"]["class_type"], "SaveImage")

    def test_collect_wf3_outputs_flattens_history_events(self):
        module = _load_boundary_module()

        outputs = module.collect_wf3_outputs(
            {
                "abc": {
                    "outputs": {
                        "10": {"wf3": [{"key": "caption", "type": "string", "value": "hello"}]},
                        "11": {"wf3": [{"key": "score", "type": "float", "value": 0.5}]},
                    }
                }
            },
            "abc",
        )

        self.assertEqual(
            outputs,
            {
                "caption": {"type": "string", "value": "hello"},
                "score": {"type": "float", "value": 0.5},
            },
        )

    def test_cancel_prompt_removes_only_matching_queue_item_or_interrupts_matching_run(self):
        module = _load_boundary_module()

        class FakeQueue:
            def __init__(self):
                self.queued = [(1, "queued-id", {}, {}, [], {})]
                self.running = {"running-id"}

            def delete_queue_item(self, predicate):
                for index, item in enumerate(self.queued):
                    if predicate(item):
                        self.queued.pop(index)
                        return True
                return False

            def interrupt_if_running(self, prompt_id):
                return prompt_id in self.running

        queue = FakeQueue()
        queued = module.cancel_prompt("queued-id", queue)
        running = module.cancel_prompt("running-id", queue)
        absent = module.cancel_prompt("other-id", queue)

        self.assertTrue(queued["removed_from_queue"])
        self.assertFalse(queued["interrupted_running"])
        self.assertTrue(running["interrupted_running"])
        self.assertFalse(running["removed_from_queue"])
        self.assertFalse(absent["cancelled"])

    def test_cancel_prompt_cleans_materialized_inputs_when_queued_item_is_removed(self):
        with TemporaryDirectory() as input_dir:
            module = _load_boundary_module(input_dir=input_dir)
            request_dir = module._request_input_dir(Path(input_dir), "queued-id")
            module._mark_materialized_request_dir(request_dir)
            (request_dir / "input.png").write_bytes(b"materialized")

            class FakeQueue:
                def __init__(self):
                    self.queued = [(1, "queued-id", {}, {}, [], {})]

                def delete_queue_item(self, predicate):
                    for index, item in enumerate(self.queued):
                        if predicate(item):
                            self.queued.pop(index)
                            return True
                    return False

                def interrupt_if_running(self, _prompt_id):
                    return False

            queue = FakeQueue()
            result = module.cancel_prompt("queued-id", queue)

            self.assertTrue(result["removed_from_queue"])
            self.assertFalse(request_dir.exists())

            prequeue_dir = module._request_input_dir(Path(input_dir), "prequeue-id")
            module._mark_materialized_request_dir(prequeue_dir)
            (prequeue_dir / "input.png").write_bytes(b"materialized")
            module.record_prompt_cancellation("prequeue-id")
            prequeue_result = module.cancel_prompt("prequeue-id", queue)

            self.assertFalse(prequeue_result["cancelled"])
            self.assertFalse(prequeue_dir.exists())

            legacy_dir = Path(input_dir) / "cutlery_wf3" / "legacy-id"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / module.LEGACY_MATERIALIZED_MARKER_NAME).write_text("1", encoding="utf-8")
            (legacy_dir / "input.png").write_bytes(b"materialized")
            queue.queued.append((1, "legacy-id", {}, {}, [], {}))

            legacy_result = module.cancel_prompt("legacy-id", queue)

            self.assertTrue(legacy_result["removed_from_queue"])
            self.assertFalse(legacy_dir.exists())

    def test_run_workflow_consumes_cancellation_recorded_before_queueing(self):
        module = _load_boundary_module()
        module.record_prompt_cancellation("cancel-before-queue")

        result, status = asyncio.run(
            module._run_workflow(
                {
                    "prompt_id": "cancel-before-queue",
                    "workflow": {},
                }
            )
        )

        self.assertEqual(status, 409)
        self.assertFalse(result["ok"])
        self.assertTrue(result["cancelled"])
        self.assertEqual(result["prompt_id"], "cancel-before-queue")
        self.assertIsNone(module.PromptServer.instance.prompt_queue)
        self.assertFalse(module.consume_prompt_cancellation("cancel-before-queue"))

    def test_run_workflow_preserves_native_image_tensor_in_queued_wf3_inputs(self):
        with TemporaryDirectory() as input_dir:
            module = _load_boundary_module(input_dir=input_dir)
            tensor = torch.zeros((1, 2, 3, 3))

            class FakeQueue:
                def __init__(self):
                    self.item = None

                def put(self, item):
                    self.item = item

                def get_history(self, prompt_id=None):
                    return {
                        prompt_id: {
                            "outputs": {},
                            "status": {"status_str": "success"},
                        }
                    }

            queue = FakeQueue()
            module.PromptServer.instance.prompt_queue = queue
            with mock.patch.dict(sys.modules, {"execution": _execution_stub()}):
                result, status = asyncio.run(
                    module._run_workflow(
                        {
                            "prompt_id": "tensor-round-trip",
                            "workflow": _image_input_workflow(),
                            "values": {"image": tensor},
                        }
                    )
                )

            self.assertEqual(status, 200)
            self.assertTrue(result["ok"])
            self.assertIsNotNone(queue.item)
            self.assertIs(queue.item[3]["wf3"]["inputs"]["image"], tensor)
            self.assertFalse(module._request_input_dir(Path(input_dir), "tensor-round-trip").exists())

    def test_run_workflow_compiles_editor_remote_groups_before_queueing(self):
        module = _load_boundary_module()
        compile_calls = []

        async def compile_remote_groups(body):
            compile_calls.append(body)
            prompt = {node_id: dict(node) for node_id, node in body["prompt"].items() if node_id != "2"}
            wrapper_id = "cutlery_remote_group_1"
            prompt["3"] = {
                "class_type": "CutleryWorkflowOutput",
                "inputs": {"ports_json": '[{"name":"result","type":"string"}]', "value_1": [wrapper_id, 0]},
            }
            prompt[wrapper_id] = {
                "class_type": (
                    "CutleryRemoteGroupExecutor"
                    if body.get("partial_execution_targets")
                    else "CutleryRemoteGroupValueExecutor"
                ),
                "inputs": {"remote_base_url": "127.0.0.1:8889"},
            }
            return {
                "prompt": prompt,
                "remaps": {"2": wrapper_id},
                "targets": ["127.0.0.1:8889"],
            }

        class FakeQueue:
            def __init__(self):
                self.item = None

            def put(self, item):
                self.item = item

            def get_history(self, prompt_id=None):
                return {
                    prompt_id: {
                        "outputs": {},
                        "status": {"status_str": "success"},
                    }
                }

        validated_targets = []

        async def validate_prompt(_prompt_id, workflow, partial_execution_targets):
            validated_targets.append(partial_execution_targets)
            if partial_execution_targets:
                self.assertEqual(workflow[partial_execution_targets[0]]["class_type"], "CutleryRemoteGroupExecutor")
            return True, None, partial_execution_targets, {}

        queue = FakeQueue()
        module.PromptServer.instance.prompt_queue = queue
        with mock.patch.dict(
            sys.modules,
            {
                "execution": types.SimpleNamespace(validate_prompt=validate_prompt, SENSITIVE_EXTRA_DATA_KEYS=()),
                "nodes_remote": types.SimpleNamespace(_compile_remote_groups_request=compile_remote_groups),
            },
        ):
            result, status = asyncio.run(
                module._run_workflow(
                    {
                        "prompt_id": "remote-editor",
                        "workflow": _editor_remote_workflow(),
                        "values": {"prompt": "hello"},
                    }
                )
            )
            normal_wrapper = queue.item[2]["cutlery_remote_group_1"]

            partial_result, partial_status = asyncio.run(
                module._run_workflow(
                    {
                        "prompt_id": "remote-editor-partial",
                        "workflow": _editor_remote_workflow(),
                        "values": {"prompt": "hello"},
                        "partial_execution_targets": ["2"],
                    }
                )
            )

        self.assertEqual(status, 200)
        self.assertTrue(result["ok"])
        queued_prompt = queue.item[2]
        self.assertNotIn("2", queued_prompt)
        wrapper = queued_prompt["cutlery_remote_group_1"]
        self.assertEqual(compile_calls[0]["partial_execution_targets"], None)
        self.assertEqual(normal_wrapper["class_type"], "CutleryRemoteGroupValueExecutor")
        self.assertEqual(wrapper["class_type"], "CutleryRemoteGroupExecutor")
        self.assertEqual(wrapper["inputs"]["remote_base_url"], "127.0.0.1:8889")
        self.assertEqual(partial_status, 200)
        self.assertTrue(partial_result["ok"])
        self.assertEqual(compile_calls[1]["partial_execution_targets"], ["2"])
        self.assertEqual(validated_targets[1], ["cutlery_remote_group_1"])

    def test_run_workflow_transfers_materialized_directory_ownership_after_queue_put(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            source = root / "source.png"
            source.write_bytes(b"png")
            module = _load_boundary_module(input_dir=str(input_dir))

            class FakeQueue:
                def __init__(self):
                    self.item = None

                def put(self, item):
                    self.item = item

                def get_history(self, prompt_id=None):
                    return {
                        prompt_id: {
                            "outputs": {},
                            "status": {"status_str": "success"},
                        }
                    }

            queue = FakeQueue()
            module.PromptServer.instance.prompt_queue = queue
            with mock.patch.dict(sys.modules, {"execution": _execution_stub()}):
                result, status = asyncio.run(
                    module._run_workflow(
                        {
                            "prompt_id": "queue-success",
                            "workflow": _image_input_workflow(),
                            "values": {"image": str(source)},
                        }
                    )
                )

            request_dir = module._request_input_dir(input_dir, "queue-success").resolve()
            self.assertEqual(status, 200)
            self.assertTrue(result["ok"])
            self.assertTrue((request_dir / module.MATERIALIZED_MARKER_NAME).is_file())
            self.assertEqual(
                queue.item[3]["wf3"]["_materialized_request_dir"],
                str(request_dir),
            )

    def test_run_workflow_cleans_materialized_inputs_when_execution_import_fails(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            source = root / "source.png"
            source.write_bytes(b"png")
            module = _load_boundary_module(input_dir=str(input_dir))
            original_import = builtins.__import__

            def fail_execution_import(name, *args, **kwargs):
                if name == "execution":
                    raise ImportError("execution unavailable")
                return original_import(name, *args, **kwargs)

            with mock.patch.object(builtins, "__import__", side_effect=fail_execution_import):
                result, status = asyncio.run(
                    module._run_workflow(
                        {
                            "prompt_id": "import-failure",
                            "workflow": _image_input_workflow(),
                            "values": {"image": str(source)},
                        }
                    )
                )

            self.assertEqual(status, 500)
            self.assertIn("execution unavailable", result["error"])
            self.assertFalse(module._request_input_dir(input_dir, "import-failure").exists())

    def test_run_workflow_cleans_partially_materialized_inputs_on_input_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            source = root / "source.png"
            source.write_bytes(b"png")
            module = _load_boundary_module(input_dir=str(input_dir))
            workflow = {
                "1": {
                    "class_type": "CutleryWorkflowInput",
                    "inputs": {
                        "ports_json": (
                            '[{"name":"first","type":"image","required":true},'
                            '{"name":"second","type":"image","required":true}]'
                        ),
                    },
                }
            }

            result, status = asyncio.run(
                module._run_workflow(
                    {
                        "prompt_id": "partial-materialization",
                        "workflow": workflow,
                        "values": {
                            "first": str(source),
                            "second": str(root / "missing.png"),
                        },
                    }
                )
            )

            self.assertEqual(status, 400)
            self.assertIn("does not point to a readable file", result["error"])
            self.assertFalse(module._request_input_dir(input_dir, "partial-materialization").exists())

    def test_run_workflow_cleans_materialized_inputs_when_replacement_raises(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            source = root / "source.png"
            source.write_bytes(b"png")
            module = _load_boundary_module(input_dir=str(input_dir))
            module.PromptServer.instance.node_replace_manager = types.SimpleNamespace(
                apply_replacements=mock.Mock(side_effect=RuntimeError("replacement failed"))
            )

            with mock.patch.dict(sys.modules, {"execution": _execution_stub()}):
                with self.assertRaisesRegex(RuntimeError, "replacement failed"):
                    asyncio.run(
                        module._run_workflow(
                            {
                                "prompt_id": "replacement-failure",
                                "workflow": _image_input_workflow(),
                                "values": {"image": str(source)},
                            }
                        )
                    )

            self.assertFalse(module._request_input_dir(input_dir, "replacement-failure").exists())

    def test_run_workflow_cleans_materialized_inputs_when_validation_raises(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            source = root / "source.png"
            source.write_bytes(b"png")
            module = _load_boundary_module(input_dir=str(input_dir))

            with mock.patch.dict(
                sys.modules,
                {"execution": _execution_stub(RuntimeError("validation failed"))},
            ):
                with self.assertRaisesRegex(RuntimeError, "validation failed"):
                    asyncio.run(
                        module._run_workflow(
                            {
                                "prompt_id": "validation-failure",
                                "workflow": _image_input_workflow(),
                                "values": {"image": str(source)},
                            }
                        )
                    )

            self.assertFalse(module._request_input_dir(input_dir, "validation-failure").exists())

    def test_run_workflow_cleans_materialized_inputs_when_number_parsing_raises(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            source = root / "source.png"
            source.write_bytes(b"png")
            module = _load_boundary_module(input_dir=str(input_dir))

            with mock.patch.dict(sys.modules, {"execution": _execution_stub()}):
                with self.assertRaises(ValueError):
                    asyncio.run(
                        module._run_workflow(
                            {
                                "prompt_id": "number-failure",
                                "workflow": _image_input_workflow(),
                                "values": {"image": str(source)},
                                "number": "not-a-number",
                            }
                        )
                    )

            self.assertFalse(module._request_input_dir(input_dir, "number-failure").exists())

    def test_run_workflow_cleans_materialized_inputs_when_sensitive_extraction_raises(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            source = root / "source.png"
            source.write_bytes(b"png")
            module = _load_boundary_module(input_dir=str(input_dir))

            class FailingSensitiveKeys:
                def __iter__(self):
                    raise RuntimeError("sensitive extraction failed")

            with mock.patch.dict(
                sys.modules,
                {"execution": _execution_stub(sensitive_keys=FailingSensitiveKeys())},
            ):
                with self.assertRaisesRegex(RuntimeError, "sensitive extraction failed"):
                    asyncio.run(
                        module._run_workflow(
                            {
                                "prompt_id": "sensitive-failure",
                                "workflow": _image_input_workflow(),
                                "values": {"image": str(source)},
                            }
                        )
                    )

            self.assertFalse(module._request_input_dir(input_dir, "sensitive-failure").exists())

    def test_run_workflow_cleans_materialized_inputs_when_queue_put_raises(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            source = root / "source.png"
            source.write_bytes(b"png")
            module = _load_boundary_module(input_dir=str(input_dir))

            class FailingQueue:
                def put(self, _item):
                    raise RuntimeError("queue insertion failed")

            module.PromptServer.instance.prompt_queue = FailingQueue()
            with mock.patch.dict(sys.modules, {"execution": _execution_stub()}):
                with self.assertRaisesRegex(RuntimeError, "queue insertion failed"):
                    asyncio.run(
                        module._run_workflow(
                            {
                                "prompt_id": "put-failure",
                                "workflow": _image_input_workflow(),
                                "values": {"image": str(source)},
                            }
                        )
                    )

            self.assertFalse(module._request_input_dir(input_dir, "put-failure").exists())

    def test_run_workflow_rejects_invalid_wait_values_before_enqueueing(self):
        invalid_cases = (
            ("timeout_seconds", 0),
            ("timeout_seconds", float("nan")),
            ("timeout_seconds", 86401),
            ("poll_seconds", float("inf")),
            ("poll_seconds", -0.1),
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            input_dir.mkdir()
            source = root / "source.png"
            source.write_bytes(b"png")
            module = _load_boundary_module(input_dir=str(input_dir))

            class RecordingQueue:
                def __init__(self):
                    self.items = []

                def put(self, item):
                    self.items.append(item)

            queue = RecordingQueue()
            module.PromptServer.instance.prompt_queue = queue
            with mock.patch.dict(sys.modules, {"execution": _execution_stub()}):
                for index, (field_name, value) in enumerate(invalid_cases):
                    with self.subTest(field_name=field_name, value=value):
                        prompt_id = f"invalid-wait-{index}"
                        result, status = asyncio.run(
                            module._run_workflow(
                                {
                                    "prompt_id": prompt_id,
                                    "workflow": _image_input_workflow(),
                                    "values": {"image": str(source)},
                                    field_name: value,
                                }
                            )
                        )

                        self.assertEqual(status, 400)
                        self.assertFalse(result["ok"])
                        self.assertIn(field_name, result["error"])
                        self.assertFalse(module._request_input_dir(input_dir, prompt_id).exists())

            self.assertEqual(queue.items, [])

    def test_package_init_merges_boundary_node_mappings(self):
        package_name = "cutlery_nodes_wf3_init_test"
        submodules = [
            "nodes",
            "nodes_3d",
            "nodes_image_crop",
            "nodes_lora",
            "nodes_attention_switch",
            "nodes_klein",
            "nodes_template",
            "nodes_any",
            "nodes_mask",
            "nodes_caption",
            "nodes_hidream_o1",
            "nodes_yingmusic",
            "nodes_transcription",
            "nodes_video",
            "nodes_lrclib",
            "nodes_audio",
            "nodes_higgs_audio",
            "nodes_alphaface",
            "nodes_xvc",
            "nodes_fish_s2",
            "nodes_face",
            "nodes_remote",
            "nodes_remote_clip",
            "nodes_blender",
        ]
        runtime_modules = ("server", "aiohttp", "folder_paths", "nodes")
        previous = {
            name: sys.modules.get(name)
            for name in [
                package_name,
                *(f"{package_name}.{item}" for item in submodules),
                *runtime_modules,
            ]
        }
        try:
            for item in submodules:
                module = types.ModuleType(f"{package_name}.{item}")
                module.NODE_CLASS_MAPPINGS = {}
                module.NODE_DISPLAY_NAME_MAPPINGS = {}
                sys.modules[f"{package_name}.{item}"] = module

            class PackageRoutes:
                def get(self, _path):
                    return lambda fn: fn

                def post(self, _path):
                    return lambda fn: fn

            server_stub = types.ModuleType("server")
            server_stub.PromptServer = type(
                "PromptServer",
                (),
                {
                    "instance": types.SimpleNamespace(
                        routes=PackageRoutes(),
                        prompt_queue=None,
                        number=0,
                    )
                },
            )
            aiohttp_stub = types.ModuleType("aiohttp")
            aiohttp_stub.web = types.SimpleNamespace(json_response=lambda payload, **_kwargs: payload)
            folder_paths_stub = types.ModuleType("folder_paths")
            comfy_nodes_stub = types.ModuleType("nodes")
            sys.modules.update(
                {
                    "server": server_stub,
                    "aiohttp": aiohttp_stub,
                    "folder_paths": folder_paths_stub,
                    "nodes": comfy_nodes_stub,
                }
            )

            spec = importlib.util.spec_from_file_location(
                package_name,
                REPO_ROOT / "__init__.py",
                submodule_search_locations=[str(REPO_ROOT)],
            )
            assert spec is not None
            assert spec.loader is not None
            package = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = package
            spec.loader.exec_module(package)

            self.assertIn("CutleryWorkflowInput", package.NODE_CLASS_MAPPINGS)
            self.assertIn("CutleryWorkflowOutput", package.NODE_CLASS_MAPPINGS)
            self.assertEqual(package.NODE_DISPLAY_NAME_MAPPINGS["CutleryWorkflowInput"], "Workflow Input")
        finally:
            for name, value in previous.items():
                if value is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
            sys.modules.pop(f"{package_name}.nodes_wf3_boundary", None)

    def test_run_endpoint_is_registered(self):
        module = _load_boundary_module()

        routes = module.PromptServer.instance.routes

        self.assertIn(("POST", "/cutlery/run"), routes.handlers)
        self.assertIn(("POST", "/cutlery/wf3/run"), routes.handlers)

    def test_run_endpoints_are_disabled_without_explicit_opt_in(self):
        module = _load_boundary_module()
        routes = module.PromptServer.instance.routes

        for path in ("/cutlery/run", "/cutlery/wf3/run"):
            with self.subTest(path=path), mock.patch.object(module, "_workflow_run_enabled", return_value=False), mock.patch.object(
                module, "_request_json", side_effect=AssertionError("disabled route must not read the request")
            ):
                response = asyncio.run(routes.handlers[("POST", path)](object()))

            self.assertEqual(response["status"], 403)
            self.assertEqual(response["payload"]["code"], "workflow_run_disabled")

    def test_new_and_legacy_run_endpoints_share_enabled_handler(self):
        module = _load_boundary_module()
        routes = module.PromptServer.instance.routes

        for path in ("/cutlery/run", "/cutlery/wf3/run"):
            with self.subTest(path=path), mock.patch.object(module, "_workflow_run_enabled", return_value=True), mock.patch.object(
                module, "_request_json", new=mock.AsyncMock(return_value={"workflow": {}, "values": {}})
            ), mock.patch.object(module, "_run_workflow", new=mock.AsyncMock(return_value=({"ok": True}, 200))):
                response = asyncio.run(routes.handlers[("POST", path)](object()))

            self.assertEqual(response["status"], 200)
            self.assertTrue(response["payload"]["ok"])


if __name__ == "__main__":
    unittest.main()
