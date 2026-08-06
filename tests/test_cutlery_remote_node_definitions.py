import json
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from cutlery_remote.node_definitions import (  # noqa: E402
    BROWSER_OWNED_INPUT_REGISTRIES,
    NodeDefinitionRequestError,
    SHARED_EXTENSION_MODULE,
    _install_shared_browser_input_registries,
    build_node_definitions_payload,
)


class RemoteNodeDefinitionsTests(unittest.TestCase):
    def test_private_browser_contracts_can_be_published_before_public_module_loads(self):
        extension = types.ModuleType(SHARED_EXTENSION_MODULE)
        extension.browser_input_registries = {
            "PrivateNode": {"model": "private.registry.v1"},
            "InvalidNode": {"model": object()},
        }
        with mock.patch.dict(sys.modules, {SHARED_EXTENSION_MODULE: extension}):
            installed = _install_shared_browser_input_registries()
        try:
            self.assertEqual(installed, {"PrivateNode"})
            self.assertEqual(BROWSER_OWNED_INPUT_REGISTRIES["PrivateNode"], {"model": "private.registry.v1"})
            self.assertNotIn("InvalidNode", BROWSER_OWNED_INPUT_REGISTRIES)
        finally:
            BROWSER_OWNED_INPUT_REGISTRIES.pop("PrivateNode", None)

    def test_legacy_definitions_preserve_scalar_combo_types_and_structural_signature(self):
        class LegacyClipLoader:
            calls = 0
            RETURN_TYPES = ("CLIP", "STRING")
            RETURN_NAMES = ("clip", "label")
            OUTPUT_IS_LIST = (False, True)

            @classmethod
            def INPUT_TYPES(cls):
                cls.calls += 1
                return {
                    "required": {
                        "clip_name": (["clip.safetensors", 2, False, None, 1.5],),
                        "strength": ("FLOAT", {"default": 1.0}),
                    },
                    "optional": {
                        "device": (["default", "cpu"],),
                        "image": (["remote.png"], {"image_upload": True}),
                    },
                    "hidden": {"prompt": "PROMPT"},
                }

        payload = build_node_definitions_payload(
            ["CLIPLoader"],
            node_class_mappings={"CLIPLoader": LegacyClipLoader},
        )
        definition = payload["definitions"]["CLIPLoader"]

        self.assertTrue(payload["ok"])
        self.assertEqual(LegacyClipLoader.calls, 1)
        self.assertEqual(definition["source"], "INPUT_TYPES")
        self.assertEqual(
            definition["inputs"]["required"]["clip_name"]["options"],
            ["clip.safetensors", 2, False, None, 1.5],
        )
        self.assertTrue(definition["inputs"]["required"]["clip_name"]["materializable"])
        self.assertFalse(definition["inputs"]["required"]["clip_name"]["upload_backed"])
        self.assertTrue(definition["inputs"]["optional"]["image"]["upload_backed"])
        self.assertEqual(definition["inputs"]["required"]["strength"]["kind"], "noncombo")
        self.assertEqual(definition["inputs"]["hidden"]["prompt"]["type"], "PROMPT")
        self.assertEqual(
            definition["cache"],
            {
                "declared_inputs_only": True,
                "has_change_fingerprint": False,
                "not_idempotent": False,
                "output_node": False,
            },
        )
        self.assertEqual(
            definition["signature"]["inputs"]["required"],
            [
                {"name": "clip_name", "kind": "combo", "type": "COMBO"},
                {"name": "strength", "kind": "noncombo", "type": "FLOAT"},
            ],
        )
        self.assertEqual(
            definition["outputs"],
            [
                {"index": 0, "type": "CLIP", "name": "clip", "is_list": False},
                {"index": 1, "type": "STRING", "name": "label", "is_list": True},
            ],
        )
        json.dumps(payload, allow_nan=False)

    def test_change_tracking_and_not_idempotent_nodes_are_not_sender_cacheable(self):
        class ChangeTrackedNode:
            RETURN_TYPES = ("STRING",)

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"value": ("STRING",)}}

            @classmethod
            def IS_CHANGED(cls, value):
                return value

        class NonIdempotentNode:
            RETURN_TYPES = ("STRING",)
            NOT_IDEMPOTENT = True

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"value": ("STRING",)}}

        definitions = build_node_definitions_payload(
            ["ChangeTrackedNode", "NonIdempotentNode"],
            node_class_mappings={
                "ChangeTrackedNode": ChangeTrackedNode,
                "NonIdempotentNode": NonIdempotentNode,
            },
        )["definitions"]

        self.assertEqual(
            definitions["ChangeTrackedNode"]["cache"],
            {
                "declared_inputs_only": False,
                "has_change_fingerprint": True,
                "not_idempotent": False,
                "output_node": False,
            },
        )
        self.assertEqual(
            definitions["NonIdempotentNode"]["cache"],
            {
                "declared_inputs_only": False,
                "has_change_fingerprint": False,
                "not_idempotent": True,
                "output_node": False,
            },
        )
        self.assertEqual(
            definitions["ChangeTrackedNode"]["signature"]["cache"],
            definitions["ChangeTrackedNode"]["cache"],
        )

    def test_v3_definition_is_preferred_and_dynamic_inputs_are_explicit(self):
        class V3Node:
            calls = 0

            @classmethod
            def GET_NODE_INFO_V1(cls):
                cls.calls += 1
                return {
                    "input": {
                        "required": {
                            "model_name": ["COMBO", {"options": ["remote.pth", 7]}],
                            "mode": [
                                "COMFY_DYNAMICCOMBO_V3",
                                {"options": [{"key": "one", "inputs": {}}]},
                            ],
                            "caption": ["STRING", {"multiline": True}],
                        }
                    },
                    "is_input_list": True,
                    "output": ["UPSCALE_MODEL"],
                    "output_name": ["model"],
                    "output_is_list": [False],
                }

            @classmethod
            def INPUT_TYPES(cls):
                raise AssertionError("V3 nodes must not fall back to INPUT_TYPES")

        payload = build_node_definitions_payload(
            ["UpscaleModelLoader"],
            node_class_mappings={"UpscaleModelLoader": V3Node},
        )
        definition = payload["definitions"]["UpscaleModelLoader"]

        self.assertTrue(definition["ok"])
        self.assertEqual(V3Node.calls, 1)
        self.assertEqual(definition["source"], "GET_NODE_INFO_V1")
        self.assertEqual(definition["inputs"]["required"]["model_name"]["options"], ["remote.pth", 7])
        self.assertTrue(definition["inputs"]["required"]["model_name"]["materializable"])
        self.assertEqual(definition["inputs"]["required"]["mode"]["kind"], "dynamic")
        self.assertEqual(definition["inputs"]["required"]["caption"]["kind"], "noncombo")
        self.assertTrue(definition["signature"]["input_is_list"])

    def test_remote_clip_placeholder_combos_are_registry_owned(self):
        class RemoteClipNode:
            RETURN_TYPES = ("CONDITIONING",)

            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {
                        "prompt": ("STRING", {"default": ""}),
                        "clip_name1": (["Loading remote CLIP choices..."],),
                        "clip_name2": (["Loading remote CLIP choices..."],),
                        "clip_type": (["ltxv"],),
                    }
                }

        definition = build_node_definitions_payload(
            ["CutleryRemoteDualClipTextEncode"],
            node_class_mappings={
                "CutleryRemoteDualClipTextEncode": RemoteClipNode,
            },
        )["definitions"]["CutleryRemoteDualClipTextEncode"]

        for input_name in ("clip_name1", "clip_name2", "clip_type"):
            input_definition = definition["inputs"]["required"][input_name]
            self.assertEqual(input_definition["kind"], "dynamic")
            self.assertEqual(
                input_definition["registry"],
                "cutlery.remote_clip.v1",
            )
            self.assertNotIn("options", input_definition)

    def test_missing_and_raising_classes_report_independent_per_class_errors(self):
        class BrokenNode:
            calls = 0

            @classmethod
            def INPUT_TYPES(cls):
                cls.calls += 1
                raise RuntimeError("registry offline")

        payload = build_node_definitions_payload(
            ["MissingNode", "BrokenNode"],
            node_class_mappings={"BrokenNode": BrokenNode},
        )

        self.assertFalse(payload["ok"])
        self.assertTrue(payload["definitions"]["MissingNode"]["missing"])
        self.assertEqual(
            payload["definitions"]["MissingNode"]["errors"][0]["code"],
            "node_class_missing",
        )
        self.assertFalse(payload["definitions"]["BrokenNode"]["missing"])
        self.assertEqual(
            payload["definitions"]["BrokenNode"]["errors"][0]["code"],
            "definition_error",
        )
        self.assertEqual(BrokenNode.calls, 1)

    def test_duplicate_class_names_are_inspected_once(self):
        class CountedNode:
            calls = 0
            RETURN_TYPES = ()

            @classmethod
            def INPUT_TYPES(cls):
                cls.calls += 1
                return {"required": {"choice": (["a"],)}}

        payload = build_node_definitions_payload(
            ["Counted", "Counted"],
            node_class_mappings={"Counted": CountedNode},
        )

        self.assertEqual(payload["requested_count"], 2)
        self.assertEqual(payload["definition_count"], 1)
        self.assertEqual(CountedNode.calls, 1)

    def test_option_limits_and_non_json_options_are_explicit_and_json_safe(self):
        class UnsafeNode:
            RETURN_TYPES = ()

            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {
                        "too_many": (["a", "b", "c"],),
                        "object_value": (["safe", object()],),
                        "infinite": ([float("inf")],),
                    }
                }

        payload = build_node_definitions_payload(
            ["UnsafeNode"],
            node_class_mappings={"UnsafeNode": UnsafeNode},
            max_options_per_input=2,
        )
        definition = payload["definitions"]["UnsafeNode"]

        self.assertFalse(definition["ok"])
        self.assertEqual(definition["inputs"]["required"]["too_many"]["kind"], "error")
        self.assertEqual(
            definition["inputs"]["required"]["too_many"]["error"]["code"],
            "option_limit_exceeded",
        )
        self.assertEqual(
            definition["inputs"]["required"]["object_value"]["error"]["code"],
            "invalid_option",
        )
        self.assertEqual(
            definition["inputs"]["required"]["infinite"]["error"]["code"],
            "invalid_option",
        )
        self.assertNotIn("options", definition["inputs"]["required"]["object_value"])
        json.dumps(payload, allow_nan=False)

    def test_malformed_noncombo_and_dynamic_combo_shapes_are_distinguished(self):
        class ShapeNode:
            RETURN_TYPES = ()

            @classmethod
            def INPUT_TYPES(cls):
                return {
                    "required": {
                        "malformed": (),
                        "dynamic_combo": ("COMBO", {"options": None}),
                        "bad_combo": ("COMBO", {"options": {"not": "a list"}}),
                        "plain": ("INT", {"default": 1}),
                    }
                }

        definition = build_node_definitions_payload(
            ["ShapeNode"],
            node_class_mappings={"ShapeNode": ShapeNode},
        )["definitions"]["ShapeNode"]

        self.assertEqual(definition["inputs"]["required"]["malformed"]["kind"], "error")
        self.assertEqual(definition["inputs"]["required"]["dynamic_combo"]["kind"], "dynamic")
        self.assertEqual(definition["inputs"]["required"]["bad_combo"]["kind"], "error")
        self.assertEqual(definition["inputs"]["required"]["plain"]["kind"], "noncombo")

    def test_batch_validation_enforces_class_and_option_bounds(self):
        with self.assertRaises(NodeDefinitionRequestError) as class_limit:
            build_node_definitions_payload(
                ["A", "B"],
                node_class_mappings={},
                max_class_types=1,
            )
        self.assertEqual(class_limit.exception.code, "class_limit_exceeded")

        with self.assertRaises(NodeDefinitionRequestError) as invalid_class:
            build_node_definitions_payload([123], node_class_mappings={})
        self.assertEqual(invalid_class.exception.code, "invalid_class_type")

        with self.assertRaises(NodeDefinitionRequestError) as invalid_limit:
            build_node_definitions_payload([], node_class_mappings={}, max_options_per_input=0)
        self.assertEqual(invalid_limit.exception.code, "invalid_limit")

    def test_default_mapping_is_loaded_lazily_from_nodes(self):
        class DefaultNode:
            RETURN_TYPES = ("STRING",)

            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"choice": (["remote"],)}}

        fake_nodes = types.SimpleNamespace(NODE_CLASS_MAPPINGS={"DefaultNode": DefaultNode})
        with mock.patch.dict(sys.modules, {"nodes": fake_nodes}):
            payload = build_node_definitions_payload(["DefaultNode"])

        self.assertTrue(payload["definitions"]["DefaultNode"]["ok"])
        self.assertEqual(
            payload["definitions"]["DefaultNode"]["inputs"]["required"]["choice"]["options"],
            ["remote"],
        )


if __name__ == "__main__":
    unittest.main()
