import sys
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutlery_remote.inventory import CANONICAL_MODEL_TYPES
from cutlery_remote.registry_proxy import REGISTRY_OPERATIONS


class RemoteOpenApiTests(unittest.TestCase):
    def test_contract_documents_runtime_routes_and_canonical_model_types(self):
        document = yaml.safe_load((REPO_ROOT / "docs" / "cutlery_remote_openapi.yaml").read_text(encoding="utf-8"))

        self.assertEqual(document["openapi"], "3.1.0")
        self.assertEqual(
            document["components"]["schemas"]["ModelType"]["enum"],
            list(CANONICAL_MODEL_TYPES),
        )
        self.assertEqual(
            document["components"]["schemas"]["RemoteRegistryId"]["enum"],
            list(REGISTRY_OPERATIONS),
        )
        self.assertTrue(
            {
                "/cutlery/remote/capabilities",
                "/cutlery/remote/node-definitions",
                "/cutlery/remote/proxy/node-definitions",
                "/cutlery/remote/proxy/registry",
                "/cutlery/remote/models",
                "/cutlery/remote/models/resolve",
                "/cutlery/remote/models/resolve-batch",
                "/cutlery/remote/blobs/exists",
                "/cutlery/remote/blobs",
                "/cutlery/remote/compile",
                "/cutlery/remote/group/preload",
                "/cutlery/remote/group/run-stream",
                "/cutlery/remote/group/run",
                "/cutlery/remote/group/{remote_prompt_id}/interrupt",
            }.issubset(document["paths"])
        )
        workflow_notes = document["paths"]["/cutlery/remote/group/run"]["post"]["x-agent-notes"]["workflow"]
        runtime_structure_note = next(note for note in workflow_notes if "remote-to-local MASK" in note)
        self.assertIn("LATENT", runtime_structure_note)
        self.assertIn("CONDITIONING", runtime_structure_note)
        self.assertIn("tensor-tree blob adapters", runtime_structure_note)
        do_not_use_notes = document["paths"]["/cutlery/remote/group/run"]["post"]["x-agent-notes"]["doNotUseWhen"]
        self.assertTrue(any("declare raw remote-to-local MASK" in note for note in do_not_use_notes))
        compile_route = document["paths"]["/cutlery/remote/compile"]["post"]
        self.assertIn("CutleryRemoteGroupValueExecutor", compile_route["description"])
        self.assertIn("CutleryRemoteGroupExecutor", compile_route["description"])
        schemas = document["components"]["schemas"]
        self.assertIn("cache", schemas["RemoteNodeDefinition"]["required"])
        self.assertIn("cache", schemas["RemoteNodeSignature"]["required"])
        self.assertEqual(
            set(schemas["RemoteNodeCacheContract"]["required"]),
            {"declared_inputs_only", "has_change_fingerprint", "not_idempotent", "output_node"},
        )


if __name__ == "__main__":
    unittest.main()
