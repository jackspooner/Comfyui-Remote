from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ROUTE_SOURCES = (
    "cutlery_features.py",
    "cutlery_runtime_routes.py",
    "nodes.py",
    "nodes_3d.py",
    "nodes_lora.py",
    "nodes_remote.py",
    "nodes_remote_clip.py",
    "nodes_wf3_boundary.py",
)
ROUTE_PATTERN = re.compile(
    r"@(?:PromptServer\.instance\.)?routes\.(get|post|put|delete|patch)"
    r"\(\s*[\"'](/cutlery/[^\"']*)[\"']"
)


def runtime_operations() -> set[tuple[str, str]]:
    operations: set[tuple[str, str]] = set()
    for relative_path in ROUTE_SOURCES:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        operations.update((path, method.lower()) for method, path in ROUTE_PATTERN.findall(source))
    return operations


def documented_operations() -> tuple[set[tuple[str, str]], list[dict]]:
    operations: set[tuple[str, str]] = set()
    documents = []
    for path in sorted((ROOT / "docs").glob("*_openapi.yaml")):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        documents.append(document)
        for route_path, path_item in document.get("paths", {}).items():
            for method, operation in path_item.items():
                if method.lower() in {"get", "post", "put", "delete", "patch"}:
                    operations.add((route_path, method.lower()))
                    operation["_source"] = path.name
    return operations, documents


class CutleryOpenApiParityTests(unittest.TestCase):
    def test_runtime_cutlery_routes_match_split_openapi_documents(self):
        documented, _documents = documented_operations()
        self.assertEqual(documented, runtime_operations())

    def test_every_operation_has_stable_metadata_and_success_response(self):
        _documented, documents = documented_operations()
        operation_ids = []
        failures = []
        for document in documents:
            for route_path, path_item in document["paths"].items():
                for method, operation in path_item.items():
                    if method.lower() not in {"get", "post", "put", "delete", "patch"}:
                        continue
                    label = f"{operation['_source']}:{method.upper()} {route_path}"
                    if not operation.get("operationId"):
                        failures.append(f"{label} has no operationId")
                    else:
                        operation_ids.append(operation["operationId"])
                    if not operation.get("tags"):
                        failures.append(f"{label} has no tags")
                    if "200" not in {str(status) for status in operation.get("responses", {})}:
                        failures.append(f"{label} has no 200 response")
        self.assertEqual(failures, [])
        self.assertEqual(len(operation_ids), len(set(operation_ids)))

    def test_default_disabled_and_bounded_routes_document_their_errors(self):
        _documented, documents = documented_operations()
        operations = {}
        for document in documents:
            for route_path, path_item in document["paths"].items():
                for method, operation in path_item.items():
                    if method.lower() in {"get", "post"}:
                        operations[(route_path, method.lower())] = operation

        gated = {
            ("/cutlery/runtime/status", "get"),
            ("/cutlery/run", "post"),
            ("/cutlery/wf3/run", "post"),
            ("/cutlery/loaded-models", "get"),
            ("/cutlery/clear-vram", "post"),
            ("/cutlery/restart", "post"),
            ("/cutlery/loras/info", "get"),
            ("/cutlery/loras/info", "post"),
            ("/cutlery/3d/model-files", "get"),
            ("/cutlery/3d/model-files", "post"),
            ("/cutlery/remote/capabilities", "get"),
            ("/cutlery/remote/node-definitions", "post"),
            ("/cutlery/remote/models", "get"),
            ("/cutlery/remote/models/resolve", "post"),
            ("/cutlery/remote/blobs/exists", "post"),
            ("/cutlery/remote/blobs", "post"),
            ("/cutlery/remote/group/run", "post"),
            ("/cutlery/remote/group/{remote_prompt_id}/interrupt", "post"),
            ("/cutlery/remote/clip/inventory", "get"),
            ("/cutlery/remote/clip/text-encode", "post"),
            ("/cutlery/remote/clip/dual-text-encode", "post"),
            ("/cutlery/remote/clip/qwen-image-edit-plus", "post"),
            ("/cutlery/remote/clip/clips/materialize", "post"),
            ("/cutlery/remote/clip/loras/materialize", "post"),
            ("/cutlery/remote/clip/images/materialize", "post"),
            ("/cutlery/remote/clip/clips/clear", "post"),
            ("/cutlery/remote/clip/loras/clear", "post"),
            ("/cutlery/remote/clip/images/clear", "post"),
            ("/cutlery/remote/clip/unload", "post"),
        }
        bounded = {
            ("/cutlery/3d/model-files", "post"),
            ("/cutlery/remote/clip/qwen-image-edit-plus", "post"),
            ("/cutlery/remote/clip/clips/materialize", "post"),
            ("/cutlery/remote/clip/loras/materialize", "post"),
            ("/cutlery/remote/clip/images/materialize", "post"),
        }
        for operation_key in gated:
            with self.subTest(operation=operation_key, status=403):
                self.assertIn("403", {str(status) for status in operations[operation_key]["responses"]})
        for operation_key in bounded:
            with self.subTest(operation=operation_key, status=413):
                self.assertIn("413", {str(status) for status in operations[operation_key]["responses"]})

    def test_runtime_catalog_documents_manual_command_semantics(self):
        document = yaml.safe_load((ROOT / "docs" / "cutlery_runtime_openapi.yaml").read_text(encoding="utf-8"))
        runtime_schema = document["components"]["schemas"]["RuntimeCatalogEntry"]
        commands = runtime_schema["properties"]["commands"]
        self.assertTrue(commands["uniqueItems"])
        self.assertIn("Manual integrations expose an empty array", commands["description"])
        availability_rule = runtime_schema["allOf"][0]
        self.assertEqual(availability_rule["if"]["properties"]["availability"]["const"], "manual")
        self.assertEqual(availability_rule["then"]["properties"]["commands"]["maxItems"], 0)
        self.assertEqual(availability_rule["else"]["properties"]["commands"], {"minItems": 7, "maxItems": 7})

    def test_clear_vram_contract_is_synchronous(self):
        document = yaml.safe_load((ROOT / "docs" / "cutlery_ui_actions_openapi.yaml").read_text(encoding="utf-8"))

        self.assertFalse(document["components"]["schemas"]["ClearVramResponse"]["properties"]["queued"]["const"])

    def test_workflow_run_documents_partial_remote_execution_targets(self):
        document = yaml.safe_load((ROOT / "docs" / "cutlery_workflow_openapi.yaml").read_text(encoding="utf-8"))
        properties = document["paths"]["/cutlery/run"]["post"]["requestBody"]["content"]["application/json"]["schema"]["properties"]

        targets = properties["partial_execution_targets"]
        self.assertEqual(targets["type"], "array")
        self.assertEqual(targets["items"], {"type": "string"})
        self.assertIn("promoted to terminal", targets["description"])


if __name__ == "__main__":
    unittest.main()
