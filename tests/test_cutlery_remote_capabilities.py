import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cutlery_remote.capabilities as capabilities
from cutlery_remote.capabilities import (
    REMOTE_PROTOCOL_VERSION,
    WORKFLOW_BLOB_FEATURE,
    build_capabilities_payload,
    required_features_for_workflow,
    required_serializers_for_boundary_ports,
    validate_remote_group_capabilities,
)


class RemoteCapabilitiesTests(unittest.TestCase):
    def test_capabilities_payload_exposes_protocol_and_serializers(self):
        payload = build_capabilities_payload()

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["protocol_version"], REMOTE_PROTOCOL_VERSION)
        self.assertIn("primitive", payload["serializers"])
        self.assertIn("tensor", payload["serializers"])
        self.assertIn("latent", payload["serializers"])
        self.assertIn("conditioning", payload["serializers"])
        self.assertIn("cutlery_lora_chain", payload["serializers"])
        self.assertEqual(payload["auth"]["type"], "shared_token")
        self.assertTrue(payload["features"]["remote_node_definitions_v1"])
        self.assertTrue(payload["features"]["prompt_specific_interrupt"])
        self.assertTrue(payload["features"]["remote_runtime_object_relocation_v1"])
        self.assertTrue(payload["features"]["remote_model_preload_v1"])
        self.assertTrue(payload["features"]["remote_progress_v1"])
        self.assertTrue(payload["features"][WORKFLOW_BLOB_FEATURE])
        self.assertTrue(payload["features"]["remote_lora_chain_boundary_v1"])

    def test_workflow_blob_feature_is_required_for_runtime_structure_adapters(self):
        plain_workflow = {"1": {"class_type": "KSampler", "inputs": {}}}
        self.assertEqual(required_features_for_workflow(plain_workflow), set())

        for class_type in (
            "WF3LatentToBlob",
            "WF3LatentFromBlob",
            "WF3MaskToBlob",
            "WF3MaskFromBlob",
            "WF3ConditioningToBlob",
            "WF3ConditioningFromBlob",
        ):
            with self.subTest(class_type=class_type):
                workflow = {"1": {"class_type": class_type, "inputs": {}}}
                self.assertEqual(required_features_for_workflow(workflow), {WORKFLOW_BLOB_FEATURE})

    def test_remote_group_capabilities_require_matching_protocol_and_features(self):
        payload = build_capabilities_payload()
        self.assertIs(validate_remote_group_capabilities(payload), payload)

        incompatible = build_capabilities_payload()
        incompatible["protocol_version"] = REMOTE_PROTOCOL_VERSION + 1
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            validate_remote_group_capabilities(incompatible)

        missing_feature = build_capabilities_payload()
        missing_feature["features"]["remote_node_definitions_v1"] = False
        with self.assertRaisesRegex(RuntimeError, "remote_node_definitions_v1"):
            validate_remote_group_capabilities(missing_feature)

        missing_interrupt = build_capabilities_payload()
        missing_interrupt["features"]["prompt_specific_interrupt"] = False
        with self.assertRaisesRegex(RuntimeError, "prompt_specific_interrupt"):
            validate_remote_group_capabilities(missing_interrupt)

    def test_remote_group_capabilities_require_serializers_for_actual_boundary_ports(self):
        required = required_serializers_for_boundary_ports(
            [
                {"name": "image", "type": "image"},
                {"name": "prompt", "type": "string"},
                {"name": "loras", "type": "cutlery_lora_chain"},
            ],
            [{"name": "latent", "type": "latent"}],
        )
        self.assertEqual(required, {"primitive", "tensor", "latent", "cutlery_lora_chain"})

        payload = build_capabilities_payload()
        payload["serializers"].remove("tensor")
        with self.assertRaisesRegex(RuntimeError, "tensor"):
            validate_remote_group_capabilities(
                payload,
                required_serializers=required,
            )

    def test_remote_group_capabilities_require_lora_chain_boundary_feature_when_used(self):
        required_features_for_boundary_ports = getattr(
            capabilities,
            "required_features_for_boundary_ports",
            None,
        )
        self.assertTrue(callable(required_features_for_boundary_ports))
        required_features = required_features_for_boundary_ports(
            [{"name": "loras", "type": "cutlery_lora_chain"}],
            [],
        )
        self.assertEqual(required_features, {"remote_lora_chain_boundary_v1"})

        payload = build_capabilities_payload()
        payload["features"]["remote_lora_chain_boundary_v1"] = False
        with self.assertRaisesRegex(RuntimeError, "remote_lora_chain_boundary_v1"):
            validate_remote_group_capabilities(
                payload,
                required_features=required_features,
            )


if __name__ == "__main__":
    unittest.main()
