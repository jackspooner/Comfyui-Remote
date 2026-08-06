from __future__ import annotations

import shutil
import subprocess
import textwrap
import unittest
from pathlib import Path


REMOTE_CLIP_JS = Path(__file__).resolve().parents[1] / "web" / "remote_clip_text_encode.js"


class RemoteClipFrontendBehaviorTests(unittest.TestCase):
    def test_remote_clip_registers_target_aware_choice_adapter(self):
        source = REMOTE_CLIP_JS.read_text(encoding="utf-8")

        self.assertIn('const REMOTE_REGISTRY_ENDPOINT = "/cutlery/remote/proxy/registry";', source)
        self.assertIn('const REMOTE_REGISTRY_ID = "remote_clip.choices";', source)
        self.assertIn('id: "cutlery.remote_clip.v1"', source)
        self.assertIn("registerRegistryAdapter", source)
        self.assertIn("applyRemoteWidgetOptions", source)
        self.assertNotIn("deferToRemoteRegistry", source)

    def test_remote_clip_removes_legacy_manual_lora_inputs(self):
        if shutil.which("node") is None:
            self.skipTest("node is not available")
        script = textwrap.dedent(
            r"""
            (() => {
            const fs = require("node:fs");
            const vm = require("node:vm");
            const sourcePath = process.argv.at(-1);
            let source = fs
              .readFileSync(sourcePath, "utf8")
              .replace('import { app } from "../../scripts/app.js";', "const app = globalThis.app;")
              .replace('import { api } from "../../scripts/api.js";', "const api = globalThis.api;");

            const extensions = [];
            globalThis.window = {
              addEventListener() {},
              setTimeout(callback) {
                callback();
                return 1;
              },
              clearTimeout() {},
            };
            globalThis.app = {
              graph: { _nodes: [], setDirtyCanvas() {} },
              canvas: null,
              registerExtension(extension) {
                extensions.push(extension);
              },
            };
            globalThis.api = { async fetchApi() {} };

            vm.runInThisContext(source, { filename: sourcePath });
            const extension = extensions.find((item) => item.name === "Cutlery.RemoteClipTextEncode");
            if (!extension) {
              throw new Error("Cutlery.RemoteClipTextEncode extension was not registered.");
            }

            const nodeType = function FakeRemoteClipTextEncode() {};
            extension.beforeRegisterNodeDef(nodeType, { name: "CutleryRemoteClipTextEncode" });
            const node = new nodeType();
            node.comfyClass = "CutleryRemoteClipTextEncode";
            node.inputs = [
              { name: "lora_chain", type: "CUTLERY_LORA_CHAIN" },
              ...Array.from({ length: 8 }, (_, index) => ({ name: `lora_name_${index + 1}`, type: "STRING" })),
              { name: "prompt", type: "STRING" },
              { name: "text_encoder", type: "COMBO" },
              { name: "clip_type", type: "COMBO" },
              ...Array.from({ length: 8 }, (_, index) => ({ name: `lora_${index + 1}`, type: "COMBO" })),
              ...Array.from({ length: 8 }, (_, index) => ({ name: `strength_clip_${index + 1}`, type: "FLOAT" })),
            ];
            node.widgets = [
              { name: "prompt" },
              { name: "text_encoder" },
              { name: "clip_type" },
            ];
            node.computeSize = () => [320, 180];
            node.setSize = function setSize(size) {
              this.size = size;
            };
            node.graph = globalThis.app.graph;
            globalThis.app.graph._nodes = [node];

            if (typeof node.configure !== "function") {
              throw new Error("Expected Remote CLIP to install a configure cleanup hook.");
            }
            node.configure({});

            const inputNames = node.inputs.map((input) => input.name);
            const legacy = inputNames.filter((name) => /^(?:lora_name|lora|strength_clip)_[1-8]$/.test(name));
            if (legacy.length) {
              throw new Error(`Expected legacy LoRA inputs to be removed, still found ${legacy.join(",")}`);
            }
            if (inputNames.join(",") !== "lora_chain,prompt,text_encoder,clip_type") {
              throw new Error(`Expected real inputs to remain in order, got ${inputNames.join(",")}`);
            }
            if (node.size?.join(",") !== "320,180") {
              throw new Error("Expected node size to be recomputed after removing inputs.");
            }
            })();
            """
        )
        result = subprocess.run(
            ["node", "-", str(REMOTE_CLIP_JS)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_remote_clip_node_creation_and_r_key_refresh_remote_device_model_choices(self):
        if shutil.which("node") is None:
            self.skipTest("node is not available")
        script = textwrap.dedent(
            r"""
            (async () => {
            const fs = require("node:fs");
            const vm = require("node:vm");
            const sourcePath = process.argv.at(-1);
            let source = fs
              .readFileSync(sourcePath, "utf8")
              .replace('import { app } from "../../scripts/app.js";', "const app = globalThis.app;")
              .replace('import { api } from "../../scripts/api.js";', "const api = globalThis.api;");

            const listeners = {};
            const extensions = [];
            const fetches = [];
            const timers = [];
            globalThis.window = {
              addEventListener(type, callback) {
                listeners[type] = listeners[type] ?? [];
                listeners[type].push(callback);
              },
              setTimeout(callback) {
                timers.push(callback);
                return timers.length;
              },
              clearTimeout() {},
            };
            globalThis.document = { activeElement: null };
            globalThis.app = {
              graph: { _nodes: [], setDirtyCanvas() {} },
              canvas: null,
              registerExtension(extension) {
                extensions.push(extension);
              },
            };
            globalThis.api = {
              async fetchApi(path) {
                fetches.push(path);
                if (path !== "/cutlery/remote/clip/choices") {
                  throw new Error(`Unexpected fetch path ${path}`);
                }
                return {
                  ok: true,
                  async json() {
                    return {
                      ok: true,
                      text_encoders: ["fresh-t5.gguf", "fresh-clip.safetensors"],
                      clip_types: ["stable_diffusion", "flux"],
                      vaes: ["remote-vae.safetensors"],
                    };
                  },
                };
              },
            };

            vm.runInThisContext(source, { filename: sourcePath });
            const extension = extensions.find((item) => item.name === "Cutlery.RemoteClipTextEncode");
            if (!extension) {
              throw new Error("Cutlery.RemoteClipTextEncode extension was not registered.");
            }

            const nodeType = function FakeRemoteClipTextEncode() {};
            extension.beforeRegisterNodeDef(nodeType, { name: "CutleryRemoteClipTextEncode" });
            const node = new nodeType();
            node.comfyClass = "CutleryRemoteClipTextEncode";
            node.widgets = [
              { name: "prompt", value: "", type: "string" },
              { name: "text_encoder", value: "stale-t5.safetensors", type: "combo", options: { values: ["stale-t5.safetensors"] } },
              { name: "clip_type", value: "stable_diffusion", type: "combo", options: { values: ["stable_diffusion"] } },
            ];
            node.computeSize = () => [300, 160];
            node.setSize = function setSize(size) {
              this.size = size;
            };
            node.graph = globalThis.app.graph;
            const qwenType = function FakeRemoteQwenImageEditPlus() {};
            extension.beforeRegisterNodeDef(qwenType, { name: "CutleryRemoteTextEncodeQwenImageEditPlus" });
            const qwenNode = new qwenType();
            qwenNode.comfyClass = "CutleryRemoteTextEncodeQwenImageEditPlus";
            qwenNode.widgets = [
              { name: "prompt", value: "", type: "string" },
              { name: "text_encoder", value: "stale-qwen.gguf", type: "combo", options: { values: ["stale-qwen.gguf"] } },
              { name: "vae_name", value: "stale-vae.safetensors", type: "combo", options: { values: ["stale-vae.safetensors"] } },
            ];
            qwenNode.graph = globalThis.app.graph;
            globalThis.app.graph._nodes = [node, qwenNode];

            if (!listeners.keyup?.length) {
              throw new Error("Remote CLIP should install an r-key refresh handler.");
            }
            extension.nodeCreated(node);
            while (timers.length) {
              await timers.shift()();
            }

            const textEncoderValues = node.widgets.find((widget) => widget.name === "text_encoder").options.values;
            if (textEncoderValues.join(",") !== "fresh-t5.gguf,fresh-clip.safetensors") {
              throw new Error(`Expected refreshed text encoders, got ${textEncoderValues.join(",")}`);
            }
            if (node.widgets.find((widget) => widget.name === "text_encoder").value !== "fresh-t5.gguf") {
              throw new Error("Stale text encoder value should be replaced by the first fresh remote choice.");
            }
            if (fetches.length !== 1) {
              throw new Error(`Expected one remote choices refresh, got ${fetches.length}`);
            }

            listeners.keyup[0]({ key: "r", target: null });
            while (timers.length) {
              await timers.shift()();
            }
            const qwenTextEncoderValues = qwenNode.widgets.find((widget) => widget.name === "text_encoder").options.values;
            if (qwenTextEncoderValues.join(",") !== "fresh-t5.gguf,fresh-clip.safetensors") {
              throw new Error(`Expected refreshed Qwen text encoders, got ${qwenTextEncoderValues.join(",")}`);
            }
            const qwenVaeValues = qwenNode.widgets.find((widget) => widget.name === "vae_name").options.values;
            if (qwenVaeValues.join(",") !== "None,remote-vae.safetensors") {
              throw new Error(`Expected refreshed Qwen VAEs, got ${qwenVaeValues.join(",")}`);
            }
            if (qwenNode.widgets.find((widget) => widget.name === "vae_name").value !== "None") {
              throw new Error("Stale Qwen VAE value should be replaced by the first fresh VAE choice.");
            }
            if (fetches.length !== 2) {
              throw new Error(`Expected r-key to perform a second remote choices refresh, got ${fetches.length}`);
            }
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
            """
        )
        result = subprocess.run(
            ["node", "-", str(REMOTE_CLIP_JS)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_remote_dual_clip_string_backed_model_fields_render_as_dropdowns_immediately(self):
        if shutil.which("node") is None:
            self.skipTest("node is not available")
        script = textwrap.dedent(
            r"""
            (() => {
            const fs = require("node:fs");
            const vm = require("node:vm");
            const sourcePath = process.argv.at(-1);
            let source = fs
              .readFileSync(sourcePath, "utf8")
              .replace('import { app } from "../../scripts/app.js";', "const app = globalThis.app;")
              .replace('import { api } from "../../scripts/api.js";', "const api = globalThis.api;");

            const extensions = [];
            const timers = [];
            globalThis.window = {
              addEventListener() {},
              setTimeout(callback) {
                timers.push(callback);
                return timers.length;
              },
              clearTimeout() {},
            };
            globalThis.document = { activeElement: null };
            globalThis.app = {
              graph: { _nodes: [], setDirtyCanvas() {} },
              canvas: null,
              registerExtension(extension) {
                extensions.push(extension);
              },
            };
            globalThis.api = {
              async fetchApi() {
                throw new Error("Remote choices should not be needed to promote the dual widgets.");
              },
            };

            vm.runInThisContext(source, { filename: sourcePath });
            const extension = extensions.find((item) => item.name === "Cutlery.RemoteClipTextEncode");
            if (!extension) {
              throw new Error("Cutlery.RemoteClipTextEncode extension was not registered.");
            }

            const nodeType = function FakeRemoteDualClipTextEncode() {};
            extension.beforeRegisterNodeDef(nodeType, { name: "CutleryRemoteDualClipTextEncode" });
            const node = new nodeType();
            node.comfyClass = "CutleryRemoteDualClipTextEncode";
            node.widgets = [
              { name: "prompt", value: "", type: "string" },
              { name: "clip_name1", value: "Loading remote CLIP choices...", type: "string" },
              { name: "clip_name2", value: "Loading remote CLIP choices...", type: "string" },
              { name: "clip_type", value: "ltxv", type: "combo", options: { values: ["ltxv"] } },
            ];
            node.graph = globalThis.app.graph;
            globalThis.app.graph._nodes = [node];

            extension.nodeCreated(node);

            for (const name of ["clip_name1", "clip_name2"]) {
              const widget = node.widgets.find((item) => item.name === name);
              if (widget.type !== "combo") {
                throw new Error(`${name} should render as a combo widget, got ${widget.type}`);
              }
              if (widget.options?.values?.join(",") !== "Loading remote CLIP choices...") {
                throw new Error(`${name} should keep its current placeholder as the initial dropdown choice.`);
              }
            }
            if (timers.length !== 1) {
              throw new Error(`Expected one scheduled remote refresh, got ${timers.length}`);
            }
            })();
            """
        )
        result = subprocess.run(
            ["node", "-", str(REMOTE_CLIP_JS)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

if __name__ == "__main__":
    unittest.main()
