from __future__ import annotations

import json
import shutil
import subprocess
import textwrap
import unittest
from pathlib import Path

from cutlery_remote.node_definitions import BROWSER_OWNED_INPUT_REGISTRIES


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_FILES = (
    REPO_ROOT / "web" / "remote_clip_text_encode.js",
)


class RemoteRegistryCrossContractTests(unittest.TestCase):
    def test_frontend_adapter_ownership_matches_backend_node_definitions(self):
        if shutil.which("node") is None:
            self.skipTest("node is not available")
        script = textwrap.dedent(
            r"""
            const fs = require("node:fs");
            const vm = require("node:vm");
            const paths = process.argv.slice(2);
            const adapters = [];
            globalThis.window = {
              location: { origin: "http://127.0.0.1:8188" },
              addEventListener() {},
              clearTimeout() {},
              setTimeout(callback) {
                callback();
                return 1;
              },
            };
            globalThis.document = { activeElement: null };
            globalThis.prompt = () => null;
            globalThis.alert = () => {};
            globalThis.api = {
              addEventListener() {},
              async fetchApi() {
                throw new Error("Registry ownership inspection must not make HTTP requests.");
              },
            };
            globalThis.app = {
              graph: { _nodes: [], _groups: [], setDirtyCanvas() {} },
              canvas: null,
              registerExtension() {},
            };
            globalThis.cutleryRemoteGroups = {
              registerRegistryAdapter(adapter) {
                adapters.push(adapter);
              },
              nodeRemoteTarget() {
                return null;
              },
              scheduleRemoteWidgetRefresh() {},
              applyRemoteWidgetOptions() {
                return { applied: true, selectedAvailable: true };
              },
            };

            for (const path of paths) {
              const source = fs
                .readFileSync(path, "utf8")
                .replace('import { app } from "../../scripts/app.js";', "const app = globalThis.app;")
                .replace('import { api } from "../../scripts/api.js";', "const api = globalThis.api;");
              vm.runInThisContext(`(() => {\n${source}\n})();`, { filename: path });
            }

            const result = {};
            for (const adapter of adapters) {
              result[adapter.id] = {};
              for (const classType of adapter.classTypes ?? []) {
                const node = { comfyClass: classType, type: classType, constructor: { comfyClass: classType } };
                const inputs =
                  typeof adapter.managedInputs === "function"
                    ? adapter.managedInputs(node)
                    : adapter.managedInputs;
                result[adapter.id][classType] = [...(inputs ?? [])].sort();
              }
            }
            process.stdout.write(JSON.stringify(result));
            """
        )
        result = subprocess.run(
            ["node", "-", *(str(path) for path in ADAPTER_FILES)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        actual = json.loads(result.stdout)

        expected: dict[str, dict[str, list[str]]] = {}
        for class_type, inputs in BROWSER_OWNED_INPUT_REGISTRIES.items():
            registries = set(inputs.values())
            self.assertEqual(
                len(registries),
                1,
                f"{class_type} must have one browser registry owner.",
            )
            registry_id = registries.pop()
            expected.setdefault(registry_id, {})[class_type] = sorted(inputs)

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
