import shutil
import subprocess
import textwrap
import unittest
from pathlib import Path


REMOTE_MODELS_JS = Path(__file__).resolve().parents[1] / "web" / "remote_models.js"


class RemoteModelFrontendBehaviorTests(unittest.TestCase):
    def test_remote_model_node_refreshes_choices_from_enclosing_group_target(self):
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

            const timers = [];
            const extensions = [];
            const fetches = [];
            globalThis.window = {
              setTimeout(callback) {
                timers.push(callback);
                return timers.length;
              },
              clearTimeout() {},
              addEventListener() {},
            };
            globalThis.document = { activeElement: null };
            globalThis.app = {
              graph: {
                _nodes: [],
                _groups: [{ title: "192.0.2.247:8188 // Remote renderer", boundingRect: new Float64Array([0, 0, 400, 400]) }],
                setDirtyCanvas() {},
              },
              canvas: null,
              registerExtension(extension) {
                extensions.push(extension);
              },
            };
            globalThis.api = {
              async fetchApi(path) {
                fetches.push(path);
                if (String(path) === "/cutlery/remote/proxy/node-definitions") {
                  return {
                    ok: true,
                    async json() {
                      return {
                        ok: true,
                        nodes: {
                          CutleryRemoteModelName: {
                            available: true,
                            inputs: {
                              required: {
                                model_type: {
                                  kind: "combo",
                                  options: ["checkpoints", "text_encoders"],
                                  materializable: false,
                                },
                              },
                            },
                          },
                        },
                      };
                    },
                  };
                }
                if (!String(path).includes("/cutlery/remote/models?")) {
                  throw new Error(`Unexpected fetch path ${path}`);
                }
                return {
                  ok: true,
                  async json() {
                    return { ok: true, model_type: "checkpoints", models: ["remote-a.safetensors", "remote-b.safetensors"] };
                  },
                };
              },
            };

            vm.runInThisContext(source, { filename: sourcePath });
            const extension = extensions.find((item) => item.name === "Cutlery.RemoteModels");
            if (!extension) {
              throw new Error("Cutlery.RemoteModels extension was not registered.");
            }

            const nodeType = function FakeRemoteModelName() {};
            extension.beforeRegisterNodeDef(nodeType, { name: "CutleryRemoteModelName" });
            const node = new nodeType();
            node.comfyClass = "CutleryRemoteModelName";
            node.pos = [100, 100];
            node.size = [160, 80];
            node.widgets = [
              { name: "model_type", value: "checkpoints", type: "combo", options: { values: ["checkpoints"] } },
              { name: "model_name", value: "loading", type: "string", options: {} },
              { name: "remote_target", value: "", type: "string", options: {} },
            ];
            node.graph = globalThis.app.graph;
            globalThis.app.graph._nodes = [node];

            extension.nodeCreated(node);
            while (timers.length) {
              await timers.shift()();
            }

            const modelWidget = node.widgets.find((widget) => widget.name === "model_name");
            const targetWidget = node.widgets.find((widget) => widget.name === "remote_target");
            if (modelWidget.options.values.join(",") !== "remote-a.safetensors,remote-b.safetensors") {
              throw new Error(`Expected remote choices, got ${modelWidget.options.values.join(",")}`);
            }
            if (targetWidget.value !== "192.0.2.247:8188") {
              throw new Error(`Expected hidden remote target to be serialized, got ${targetWidget.value}`);
            }
            const modelFetch = fetches.find((path) => String(path).includes("/cutlery/remote/models?"));
            if (!modelFetch?.includes("target=192.0.2.247%3A8188") || !modelFetch.includes("model_type=checkpoints")) {
              throw new Error(`Expected group target and model type in fetch path, got ${modelFetch}`);
            }
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
            """
        )
        result = subprocess.run(
            ["node", "-", str(REMOTE_MODELS_JS)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_generic_remote_combo_overlay_preserves_typed_values_and_restores_local_choices(self):
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

            let nextTimerId = 1;
            const timers = [];
            const cancelledTimers = new Set();
            const windowListeners = [];
            const apiListeners = {};
            const extensions = [];
            let definitionFetches = 0;
            globalThis.window = {
              setTimeout(callback) {
                const id = nextTimerId++;
                timers.push({ id, callback });
                return id;
              },
              clearTimeout(id) {
                cancelledTimers.add(id);
              },
              addEventListener(name) {
                windowListeners.push(name);
              },
            };
            globalThis.document = { activeElement: null };
            globalThis.app = {
              graph: {
                _nodes: [],
                _groups: [{ title: "127.0.0.1:8189", boundingRect: new Float64Array([0, 0, 400, 400]) }],
                setDirtyCanvas() {},
              },
              canvas: null,
              registerExtension(extension) {
                extensions.push(extension);
              },
            };
            globalThis.api = {
              addEventListener(name, callback) {
                apiListeners[name] = callback;
              },
              async fetchApi(path, options) {
                if (String(path) !== "/cutlery/remote/proxy/node-definitions") {
                  throw new Error(`Unexpected fetch path ${path}`);
                }
                definitionFetches += 1;
                const request = JSON.parse(options.body);
                return {
                  ok: true,
                  async json() {
                    return {
                      ok: true,
                      nodes: Object.fromEntries(
                        request.class_types.map((classType) => [
                          classType,
                          {
                            available: true,
                            inputs: {
                              required: {
                                choice: {
                                  kind: "combo",
                                  options: [false, 2, "remote"],
                                  materializable: false,
                                },
                                empty_choice: {
                                  kind: "combo",
                                  options: [],
                                  materializable: false,
                                  upload_backed: true,
                                },
                                model_choice: {
                                  kind: "combo",
                                  options: ["remote-model.safetensors"],
                                  materializable: true,
                                },
                              },
                              optional: {},
                              hidden: {},
                            },
                          },
                        ]),
                      ),
                    };
                  },
                };
              },
            };

            vm.runInThisContext(source, { filename: sourcePath });
            const extension = extensions.find((item) => item.name === "Cutlery.RemoteModels");
            if (!extension) {
              throw new Error("Cutlery.RemoteModels extension was not registered.");
            }
            const nodeData = {
              name: "TypedRegistryLoader",
              input: {
                required: {
                  choice: [[1, 2], {}],
                  empty_choice: [["local-empty"], {}],
                  model_choice: [["local-model.safetensors"], {}],
                },
              },
            };
            const nodeType = function TypedRegistryLoader() {};
            await extension.beforeRegisterNodeDef(nodeType, nodeData);

            function makeNode(id, y) {
              const node = new nodeType();
              node.id = id;
              node.comfyClass = "TypedRegistryLoader";
              node.pos = [1000, 1000];
              node.size = [120, 80];
              node.boundingRect = new Float64Array([100, y, 120, 80]);
              node.__originalUpload = function originalUpload() {
                return "uploaded-locally";
              };
              node.widgets = [
                { name: "choice", value: 2, type: "combo", options: { values: [1, 2] } },
                { name: "empty_choice", value: "local-empty", type: "combo", options: { values: ["local-empty"] } },
                {
                  name: "model_choice",
                  value: "local-model.safetensors",
                  type: "combo",
                  options: { values: ["local-model.safetensors"] },
                },
                {
                  name: "choose file to upload",
                  type: "button",
                  disabled: false,
                  options: {},
                  callback: node.__originalUpload,
                },
              ];
              node.graph = globalThis.app.graph;
              return node;
            }

            const first = makeNode(1, 80);
            const second = makeNode(2, 200);
            globalThis.app.graph._nodes = [first, second];
            extension.setup();
            extension.nodeCreated(first);
            extension.nodeCreated(second);
            while (timers.length) {
              const timer = timers.shift();
              if (!cancelledTimers.has(timer.id)) {
                await timer.callback();
              }
            }

            const choice = first.widgets.find((widget) => widget.name === "choice");
            const emptyChoice = first.widgets.find((widget) => widget.name === "empty_choice");
            const modelChoice = first.widgets.find((widget) => widget.name === "model_choice");
            const uploadButton = first.widgets.find((widget) => widget.name === "choose file to upload");
            if (
              choice.options.values.length !== 3 ||
              choice.options.values[0] !== false ||
              choice.options.values[1] !== 2 ||
              choice.options.values[2] !== "remote"
            ) {
              throw new Error(`Expected typed remote choices, got ${JSON.stringify(choice.options.values)}`);
            }
            if (choice.value !== 2) {
              throw new Error(`Remote overlay changed the selected typed value to ${JSON.stringify(choice.value)}`);
            }
            if (emptyChoice.options.values.length !== 0 || emptyChoice.value !== "local-empty") {
              throw new Error(
                `Empty remote choices must clear the menu without changing the value: ${JSON.stringify(emptyChoice)}`,
              );
            }
            if (modelChoice.value !== "local-model.safetensors" || !first.__cutleryRemoteOptionsWarning) {
              throw new Error("Materializable model selection should be preserved as a warning.");
            }
            if (!first.__cutleryRemoteOptionsError?.includes("has no valid choices")) {
              throw new Error(`Expected a visible empty-choice error, got ${first.__cutleryRemoteOptionsError}`);
            }
            if (!uploadButton.disabled || uploadButton.callback() !== undefined) {
              throw new Error("Upload-backed local action should be disabled while the node runs remotely.");
            }
            if (definitionFetches !== 1) {
              throw new Error(`Expected one batched/coalesced definition fetch, got ${definitionFetches}`);
            }
            if (globalThis.cutleryRemoteGroups.nodeRemoteTarget(first) !== "127.0.0.1:8189") {
              throw new Error("Shared remote-group helper did not use boundingRect membership.");
            }
            if (!apiListeners.graphChanged) {
              throw new Error("Expected graphChanged refresh listener to be installed.");
            }
            if (windowListeners.includes("keyup")) {
              throw new Error("Cutlery should rely on refreshComboInNodes, not install a duplicate R-key listener.");
            }

            first.boundingRect = new Float64Array([500, 80, 120, 80]);
            second.boundingRect = new Float64Array([500, 200, 120, 80]);
            await extension.refreshComboInNodes();
            if (JSON.stringify(choice.options.values) !== JSON.stringify([1, 2])) {
              throw new Error(`Expected local choices to be restored, got ${JSON.stringify(choice.options.values)}`);
            }
            if (JSON.stringify(emptyChoice.options.values) !== JSON.stringify(["local-empty"])) {
              throw new Error(`Expected local empty choices to be restored, got ${JSON.stringify(emptyChoice.options.values)}`);
            }
            if (uploadButton.disabled || uploadButton.callback !== first.__originalUpload) {
              throw new Error("Local upload action was not restored after leaving the remote group.");
            }
            if (first.__cutleryRemoteOptionsError) {
              throw new Error(`Remote error should clear after leaving the group: ${first.__cutleryRemoteOptionsError}`);
            }
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
            """
        )
        result = subprocess.run(
            ["node", "-", str(REMOTE_MODELS_JS)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_registered_frontend_registry_adapter_owns_dynamic_remote_inputs(self):
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

            const extensions = [];
            let restored = 0;
            globalThis.window = {
              setTimeout(callback) {
                callback();
                return 1;
              },
              clearTimeout() {},
              addEventListener() {},
            };
            globalThis.app = {
              graph: {
                _nodes: [],
                _groups: [{ title: "127.0.0.1:8189", boundingRect: [0, 0, 400, 400] }],
                setDirtyCanvas() {},
              },
              canvas: null,
              registerExtension(extension) {
                extensions.push(extension);
              },
            };
            globalThis.api = {
              addEventListener() {},
              async fetchApi(path) {
                if (String(path) !== "/cutlery/remote/proxy/node-definitions") {
                  throw new Error(`Unexpected fetch path ${path}`);
                }
                return {
                  ok: true,
                  async json() {
                    return {
                      ok: true,
                      nodes: {
                        DynamicRouter: {
                          available: true,
                          compatible: true,
                          inputs: {
                            required: {
                              provider: {
                                kind: "dynamic",
                                type: "COMBO",
                                registry: "test.router.v1",
                              },
                            },
                            optional: {},
                            hidden: {},
                          },
                        },
                      },
                    };
                  },
                };
              },
            };

            vm.runInThisContext(source, { filename: sourcePath });
            const extension = extensions.find((item) => item.name === "Cutlery.RemoteModels");
            const nodeType = function DynamicRouter() {};
            await extension.beforeRegisterNodeDef(nodeType, {
              name: "DynamicRouter",
              input: { required: { provider: [["local-provider"], {}] } },
            });
            const node = new nodeType();
            node.id = 5;
            node.comfyClass = "DynamicRouter";
            node.boundingRect = [100, 100, 100, 80];
            node.widgets = [
              {
                name: "provider",
                value: "remote-provider",
                type: "combo",
                options: { values: ["local-provider"] },
              },
            ];
            node.graph = globalThis.app.graph;
            globalThis.app.graph._nodes = [node];

            await extension.refreshComboInNodes();
            if (!node.__cutleryRemoteOptionsError?.includes("test.router.v1")) {
              throw new Error(`Missing dynamic registry adapter should be visible: ${node.__cutleryRemoteOptionsError}`);
            }
            if (node.widgets[0].options.values.length !== 0) {
              throw new Error("Missing dynamic registry adapter must not leave local placeholder choices visible.");
            }

            node.boundingRect = [500, 500, 100, 80];
            await extension.refreshComboInNodes();
            if (JSON.stringify(node.widgets[0].options.values) !== JSON.stringify(["local-provider"])) {
              throw new Error("Leaving the group should restore local choices before adapter failure testing.");
            }

            let adapterShouldFail = true;
            globalThis.cutleryRemoteGroups.registerRegistryAdapter({
              id: "test.router.v1",
              classTypes: ["DynamicRouter"],
              managedInputs: ["provider"],
              async refresh(adapterNode, target) {
                if (adapterShouldFail) {
                  throw new Error("registry offline");
                }
                const result = globalThis.cutleryRemoteGroups.applyRemoteWidgetOptions(
                  adapterNode,
                  "provider",
                  ["remote-provider", "other-remote-provider"],
                  { target },
                );
                return result.selectedAvailable ? {} : { errors: ["selection missing"] };
              },
              restore() {
                restored += 1;
              },
            });

            node.boundingRect = [100, 100, 100, 80];
            await extension.refreshComboInNodes();
            if (!node.__cutleryRemoteOptionsError?.includes("registry offline")) {
              throw new Error(`Adapter failure should be visible: ${node.__cutleryRemoteOptionsError}`);
            }
            if (node.widgets[0].options.values.length !== 0) {
              throw new Error("A failed registry adapter must clear local placeholder choices.");
            }

            adapterShouldFail = false;
            await extension.refreshComboInNodes();
            const provider = node.widgets[0];
            if (JSON.stringify(provider.options.values) !== JSON.stringify(["remote-provider", "other-remote-provider"])) {
              throw new Error(`Adapter did not own the remote choices: ${JSON.stringify(provider.options.values)}`);
            }
            if (provider.value !== "remote-provider" || node.__cutleryRemoteOptionsError) {
              throw new Error("Adapter refresh should preserve a valid selected value without a generic dynamic-input error.");
            }

            node.boundingRect = [500, 500, 100, 80];
            await extension.refreshComboInNodes();
            if (JSON.stringify(provider.options.values) !== JSON.stringify(["local-provider"])) {
              throw new Error(`Local choices were not restored: ${JSON.stringify(provider.options.values)}`);
            }
            if (restored !== 1) {
              throw new Error(`Expected one adapter restore callback, got ${restored}`);
            }
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
            """
        )
        result = subprocess.run(
            ["node", "-", str(REMOTE_MODELS_JS)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_queue_preflight_blocks_if_registry_adapter_changes_serialized_widget_values(self):
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

            const extensions = [];
            const forwarded = [];
            globalThis.window = {
              location: { origin: "http://127.0.0.1:8188" },
              setTimeout() {
                return 1;
              },
              clearTimeout() {},
              addEventListener() {},
            };
            globalThis.app = {
              graph: {
                _nodes: [],
                _groups: [{ title: "127.0.0.1:8189", boundingRect: [0, 0, 400, 400] }],
                links: {},
                setDirtyCanvas() {},
                serialize() {
                  return {
                    nodes: [{ id: 1, type: "MutatingRegistryNode", pos: [100, 100], size: [100, 80] }],
                    groups: [{ title: "127.0.0.1:8189", bounding: [0, 0, 400, 400] }],
                    links: [],
                  };
                },
              },
              canvas: null,
              registerExtension(extension) {
                extensions.push(extension);
              },
            };
            globalThis.api = {
              addEventListener() {},
              async fetchApi(path, options) {
                if (String(path) === "/cutlery/remote/proxy/node-definitions") {
                  return {
                    ok: true,
                    async json() {
                      return {
                        ok: true,
                        nodes: {
                          MutatingRegistryNode: {
                            available: true,
                            compatible: true,
                            inputs: {
                              required: {
                                provider: {
                                  kind: "dynamic",
                                  type: "COMBO",
                                  registry: "test.mutating.v1",
                                },
                              },
                              optional: {},
                              hidden: {},
                            },
                          },
                        },
                      };
                    },
                  };
                }
                if (String(path) === "/prompt") {
                  forwarded.push(JSON.parse(options.body));
                  return { ok: true, async json() { return { ok: true }; } };
                }
                if (String(path) === "/cutlery/remote/compile") {
                  const request = JSON.parse(options.body);
                  const prompt = {
                    cutlery_remote_group_1: {
                      class_type: "CutleryRemoteGroupExecutor",
                      inputs: {
                        remote_workflow_json: JSON.stringify({ "1": request.prompt["1"] }),
                      },
                    },
                  };
                  return {
                    ok: true,
                    async json() {
                      return { ok: true, prompt, remaps: { "1": "cutlery_remote_group_1" } };
                    },
                  };
                }
                throw new Error(`Unexpected fetch path ${path}`);
              },
            };

            vm.runInThisContext(source, { filename: sourcePath });
            const extension = extensions.find((item) => item.name === "Cutlery.RemoteModels");
            const nodeType = function MutatingRegistryNode() {};
            await extension.beforeRegisterNodeDef(nodeType, {
              name: "MutatingRegistryNode",
              input: { required: { provider: [["local-provider"], {}], prompt_text: ["STRING", {}] } },
            });
            const node = new nodeType();
            node.id = 1;
            node.comfyClass = "MutatingRegistryNode";
            node.boundingRect = [100, 100, 100, 80];
            node.widgets = [
              { name: "provider", value: "remote-provider", type: "combo", options: { values: ["local-provider"] } },
              { name: "prompt_text", value: "before-preflight", type: "string", options: {} },
            ];
            node.graph = globalThis.app.graph;
            globalThis.app.graph._nodes = [node];

            globalThis.cutleryRemoteGroups.registerRegistryAdapter({
              id: "test.mutating.v1",
              classTypes: ["MutatingRegistryNode"],
              managedInputs: ["provider"],
              async refresh(adapterNode, target, options) {
                globalThis.cutleryRemoteGroups.applyRemoteWidgetOptions(
                  adapterNode,
                  "provider",
                  ["remote-provider"],
                  { target },
                );
                if (options.preflight && adapterNode.widgets[1].value === "before-preflight") {
                  adapterNode.widgets[1].value = "after-preflight";
                }
                return {};
              },
            });
            extension.setup();

            const firstBody = {
              prompt: {
                "1": {
                  class_type: "MutatingRegistryNode",
                  inputs: { provider: "remote-provider", prompt_text: "before-preflight" },
                },
              },
            };
            let blocked = false;
            try {
              await globalThis.api.fetchApi("/prompt", { method: "POST", body: JSON.stringify(firstBody) });
            } catch (error) {
              blocked = String(error).includes("updated widget values during remote preflight");
            }
            if (!blocked || forwarded.length !== 0 || node.widgets[1].value !== "after-preflight") {
              throw new Error(
                `Mutating preflight must update the UI but block the stale prompt: ${JSON.stringify({
                  blocked,
                  forwarded: forwarded.length,
                  value: node.widgets[1].value,
                })}`,
              );
            }

            const secondBody = {
              prompt: {
                "1": {
                  class_type: "MutatingRegistryNode",
                  inputs: { provider: "remote-provider", prompt_text: "after-preflight" },
                },
              },
            };
            await globalThis.api.fetchApi("/prompt", { method: "POST", body: JSON.stringify(secondBody) });
            if (forwarded.length !== 1) {
              throw new Error(`The reviewed second queue should be forwarded once, got ${forwarded.length}.`);
            }
            const wrapper = Object.values(forwarded[0].prompt).find(
              (promptNode) => promptNode.class_type === "CutleryRemoteGroupExecutor",
            );
            const remoteWorkflow = JSON.parse(wrapper.inputs.remote_workflow_json);
            if (remoteWorkflow["1"].inputs.prompt_text !== "after-preflight") {
              throw new Error(`Expected reviewed widget value in remote workflow: ${wrapper.inputs.remote_workflow_json}`);
            }
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
            """
        )
        result = subprocess.run(
            ["node", "-", str(REMOTE_MODELS_JS)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_remote_combo_overlay_discards_a_stale_target_response(self):
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

            const extensions = [];
            let resolveOldTarget;
            globalThis.window = {
              setTimeout() {
                return 1;
              },
              clearTimeout() {},
              addEventListener() {},
            };
            globalThis.app = {
              graph: {
                _nodes: [],
                _groups: [{ title: "127.0.0.1:8189", boundingRect: [0, 0, 400, 400] }],
                setDirtyCanvas() {},
              },
              canvas: null,
              registerExtension(extension) {
                extensions.push(extension);
              },
            };
            function definitionResponse(value) {
              return {
                ok: true,
                async json() {
                  return {
                    ok: true,
                    nodes: {
                      TargetAwareLoader: {
                        available: true,
                        compatible: true,
                        input_options: {
                          model_name: { kind: "combo", options: [value], materializable: false },
                        },
                      },
                    },
                  };
                },
              };
            }
            globalThis.api = {
              async fetchApi(path, options) {
                const request = JSON.parse(options.body);
                if (request.target === "127.0.0.1:8189") {
                  return new Promise((resolve) => {
                    resolveOldTarget = () => resolve(definitionResponse("old-target"));
                  });
                }
                if (request.target === "127.0.0.1:8190") {
                  return definitionResponse("new-target");
                }
                throw new Error(`Unexpected target ${request.target} for ${path}`);
              },
            };

            vm.runInThisContext(source, { filename: sourcePath });
            const extension = extensions.find((item) => item.name === "Cutlery.RemoteModels");
            const nodeType = function TargetAwareLoader() {};
            await extension.beforeRegisterNodeDef(nodeType, {
              name: "TargetAwareLoader",
              input: { required: { model_name: [["local"], {}] } },
            });
            const node = new nodeType();
            node.id = 1;
            node.comfyClass = "TargetAwareLoader";
            node.boundingRect = [100, 100, 120, 80];
            node.widgets = [{ name: "model_name", value: "new-target", type: "combo", options: { values: ["local"] } }];
            node.graph = globalThis.app.graph;
            globalThis.app.graph._nodes = [node];

            const oldRefresh = extension.refreshComboInNodes();
            if (!resolveOldTarget) {
              throw new Error("Old target request did not start.");
            }
            globalThis.app.graph._groups[0].title = "127.0.0.1:8190";
            await extension.refreshComboInNodes();
            resolveOldTarget();
            await oldRefresh;

            const values = node.widgets[0].options.values;
            if (JSON.stringify(values) !== JSON.stringify(["new-target"])) {
              throw new Error(`Stale old-target response overwrote current choices: ${JSON.stringify(values)}`);
            }
            if (globalThis.cutleryRemoteGroups.nodeRemoteTarget(node) !== "127.0.0.1:8190") {
              throw new Error("Shared target helper returned stale membership.");
            }
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
            """
        )
        result = subprocess.run(
            ["node", "-", str(REMOTE_MODELS_JS)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_overlapping_groups_are_visible_and_block_only_the_exact_prompt_post(self):
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

            const extensions = [];
            const forwarded = [];
            globalThis.window = {
              setTimeout() {
                return 1;
              },
              clearTimeout() {},
              addEventListener() {},
            };
            const node = {
              id: 9,
              comfyClass: "OverlapNode",
              boundingRect: [100, 100, 100, 60],
              widgets: [],
            };
            globalThis.app = {
              graph: {
                _nodes: [node],
                _groups: [
                  { title: "127.0.0.1:8189", boundingRect: [0, 0, 300, 300] },
                  { title: "127.0.0.1:8190", boundingRect: [50, 50, 300, 300] },
                ],
                links: {},
                setDirtyCanvas() {},
              },
              canvas: null,
              registerExtension(extension) {
                extensions.push(extension);
              },
            };
            node.graph = globalThis.app.graph;
            globalThis.api = {
              async fetchApi(path, options) {
                forwarded.push({ path, options });
                return { ok: true, async json() { return { ok: true }; } };
              },
            };

            vm.runInThisContext(source, { filename: sourcePath });
            const extension = extensions.find((item) => item.name === "Cutlery.RemoteModels");
            extension.setup();

            await globalThis.api.fetchApi("/debug/prompt-proxy", {
              method: "POST",
              body: JSON.stringify({ prompt: { "9": { class_type: "OverlapNode", inputs: {} } } }),
            });
            if (forwarded.length !== 1 || forwarded[0].path !== "/debug/prompt-proxy") {
              throw new Error("Non-prompt POST should pass through unchanged.");
            }

            let blocked = false;
            try {
              await globalThis.api.fetchApi("/prompt", {
                method: "POST",
                body: JSON.stringify({ prompt: { "9": { class_type: "OverlapNode", inputs: {} } } }),
              });
            } catch (error) {
              blocked = String(error).includes("multiple Cutlery remote groups");
            }
            if (!blocked) {
              throw new Error("Overlapping remote groups should block queueing with the shared membership error.");
            }
            if (forwarded.length !== 1) {
              throw new Error("Blocked /prompt request should not reach the original API fetch.");
            }
            if (!node.__cutleryRemoteOptionsError?.includes("multiple Cutlery remote groups")) {
              throw new Error(`Expected visible overlap status, got ${node.__cutleryRemoteOptionsError}`);
            }
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
            """
        )
        result = subprocess.run(
            ["node", "-", str(REMOTE_MODELS_JS)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_prompt_hook_passes_unrelated_and_class_mismatched_prompts_through_untouched(self):
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

            const extensions = [];
            const forwarded = [];
            let definitionRequests = 0;
            const remoteNode = {
              id: 1,
              comfyClass: "CLIPLoader",
              boundingRect: [100, 100, 100, 60],
              widgets: [],
            };
            const secondRemoteNode = {
              id: 2,
              comfyClass: "VAELoader",
              boundingRect: [100, 180, 100, 60],
              widgets: [],
            };
            const graph = {
              _nodes: [remoteNode, secondRemoteNode],
              _groups: [{ title: "127.0.0.1:8189", boundingRect: [0, 0, 300, 300] }],
              links: {},
              setDirtyCanvas() {},
            };
            remoteNode.graph = graph;
            secondRemoteNode.graph = graph;
            globalThis.window = {
              location: { origin: "http://127.0.0.1:8188" },
              setTimeout() {
                return 1;
              },
              clearTimeout() {},
              addEventListener() {},
            };
            globalThis.app = {
              graph,
              canvas: null,
              registerExtension(extension) {
                extensions.push(extension);
              },
            };
            globalThis.api = {
              addEventListener() {},
              async fetchApi(path, options) {
                if (String(path) === "/cutlery/remote/proxy/node-definitions") {
                  definitionRequests += 1;
                  return { ok: true, async json() { return { ok: true, nodes: {} }; } };
                }
                forwarded.push({ path, options });
                return { ok: true, async json() { return { ok: true }; } };
              },
            };

            vm.runInThisContext(source, { filename: sourcePath });
            const extension = extensions.find((item) => item.name === "Cutlery.RemoteModels");
            extension.setup();

            const unrelatedBody = {
              prompt: {
                "99": { class_type: "UnrelatedNode", inputs: { text: "unchanged" } },
              },
            };
            const unrelatedOptions = {
              method: "POST",
              body: JSON.stringify(unrelatedBody),
            };
            await globalThis.api.fetchApi("/prompt", unrelatedOptions);

            const mismatchedBody = {
              prompt: {
                "1": { class_type: "KSampler", inputs: { seed: 7 } },
              },
            };
            const mismatchedOptions = {
              method: "POST",
              body: JSON.stringify(mismatchedBody),
            };
            await globalThis.api.fetchApi("/prompt", mismatchedOptions);

            const mixedBody = {
              prompt: {
                "1": { class_type: "CLIPLoader", inputs: { clip_name: "local.safetensors" } },
                "2": { class_type: "KSampler", inputs: { seed: 8 } },
              },
            };
            const mixedOptions = {
              method: "POST",
              body: JSON.stringify(mixedBody),
            };
            await globalThis.api.fetchApi("/prompt", mixedOptions);

            if (definitionRequests !== 0) {
              throw new Error(`Unrelated prompts triggered ${definitionRequests} remote definition request(s).`);
            }
            if (forwarded.length !== 3) {
              throw new Error(`Expected three untouched prompt forwards, got ${forwarded.length}.`);
            }
            if (
              forwarded[0].options !== unrelatedOptions ||
              forwarded[1].options !== mismatchedOptions ||
              forwarded[2].options !== mixedOptions
            ) {
              throw new Error("Prompt pass-through replaced the original request options.");
            }
            if (
              JSON.stringify(JSON.parse(forwarded[0].options.body)) !== JSON.stringify(unrelatedBody) ||
              JSON.stringify(JSON.parse(forwarded[1].options.body)) !== JSON.stringify(mismatchedBody) ||
              JSON.stringify(JSON.parse(forwarded[2].options.body)) !== JSON.stringify(mixedBody)
            ) {
              throw new Error(`Prompt pass-through mutated a body: ${JSON.stringify(forwarded)}`);
            }
            for (const record of forwarded) {
              const prompt = JSON.parse(record.options.body).prompt;
              if (Object.values(prompt).some((node) => ["CutleryRemoteGroupExecutor", "CutleryRemoteGroupValueExecutor"].includes(node.class_type))) {
                throw new Error(`Unrelated prompt received a remote executor: ${record.options.body}`);
              }
            }
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
            """
        )
        result = subprocess.run(
            ["node", "-", str(REMOTE_MODELS_JS)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_api_export_command_compiles_remote_groups_and_skips_local_workflows(self):
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

            const extensions = [];
            const blobs = new Map();
            const downloads = [];
            let compileRequest = null;
            let compileCalls = 0;
            let graphToPromptCalls = 0;
            let nextBlobId = 0;
            globalThis.Blob = class Blob {
              constructor(parts, options) {
                this.parts = parts;
                this.options = options;
              }
            };
            globalThis.URL = {
              createObjectURL(blob) {
                const url = `blob:cutlery-${++nextBlobId}`;
                blobs.set(url, blob);
                return url;
              },
              revokeObjectURL(url) {
                blobs.delete(url);
              },
            };
            globalThis.document = {
              activeElement: null,
              createElement() {
                return {
                  click() {
                    downloads.push({ download: this.download, content: blobs.get(this.href).parts.join("") });
                  },
                };
              },
            };
            globalThis.window = {
              location: { origin: "http://127.0.0.1:8188" },
              setTimeout() { return 1; },
              clearTimeout() {},
              addEventListener() {},
            };

            const remoteNode = {
              id: 2,
              comfyClass: "RemoteWork",
              boundingRect: [100, 100, 80, 60],
              widgets: [],
            };
            const remoteGraph = {
              _nodes: [remoteNode],
              _groups: [{ title: "192.0.2.247:8188", boundingRect: [0, 0, 300, 300] }],
              links: {},
              setDirtyCanvas() {},
            };
            remoteNode.graph = remoteGraph;
            const localNode = {
              id: 5,
              comfyClass: "LocalWork",
              boundingRect: [100, 100, 80, 60],
              widgets: [],
            };
            const localGraph = {
              _nodes: [localNode],
              _groups: [],
              links: {},
              setDirtyCanvas() {},
            };
            localNode.graph = localGraph;
            const initialWorkflow = { nodes: [{ id: 2, type: "RemoteWork" }], groups: [{ title: "192.0.2.247:8188" }] };
            const preparedWorkflow = { nodes: [{ id: 2, type: "RemoteWork", prepared: true }], groups: [{ title: "192.0.2.247:8188" }] };
            const initialPrompt = { "2": { class_type: "RemoteWork", inputs: { seed: 1 } } };
            const preparedPrompt = { "2": { class_type: "RemoteWork", inputs: { seed: 2 } } };
            const compiledPrompt = {
              cutlery_remote_group_1: {
                class_type: "CutleryRemoteGroupExecutor",
                inputs: { remote_base_url: "192.0.2.247:8188" },
              },
            };
            const localPrompt = { "5": { class_type: "LocalWork", inputs: { value: "local" } } };

            globalThis.app = {
              graph: remoteGraph,
              canvas: null,
              registerExtension(extension) {
                extensions.push(extension);
              },
              async graphToPrompt() {
                graphToPromptCalls += 1;
                if (this.graph === remoteGraph) {
                  return graphToPromptCalls === 1
                    ? { workflow: initialWorkflow, output: initialPrompt }
                    : { workflow: preparedWorkflow, output: preparedPrompt };
                }
                return { workflow: { nodes: [{ id: 5, type: "LocalWork" }], groups: [] }, output: localPrompt };
              },
            };
            globalThis.api = {
              addEventListener() {},
              async fetchApi(path, options) {
                if (String(path) === "/cutlery/remote/proxy/node-definitions") {
                  const request = JSON.parse(options.body);
                  return {
                    ok: true,
                    async json() {
                      return {
                        ok: true,
                        nodes: Object.fromEntries(
                          request.class_types.map((classType) => [
                            classType,
                            {
                              available: true,
                              compatible: true,
                              cache: { declared_inputs_only: true },
                              inputs: { required: {}, optional: {}, hidden: {} },
                            },
                          ]),
                        ),
                      };
                    },
                  };
                }
                if (String(path) === "/cutlery/remote/compile") {
                  compileCalls += 1;
                  compileRequest = JSON.parse(options.body);
                  return { ok: true, async json() { return { ok: true, prompt: compiledPrompt }; } };
                }
                throw new Error(`Unexpected fetch path ${path}`);
              },
            };

            vm.runInThisContext(source, { filename: sourcePath });
            const extension = extensions.find((item) => item.name === "Cutlery.RemoteModels");
            const exportCommand = extension?.commands?.find((command) => command.id === "Comfy.ExportWorkflowAPI");
            if (!exportCommand) {
              throw new Error("Remote-aware API export command was not registered.");
            }
            if (
              exportCommand.icon !== "pi pi-download" ||
              exportCommand.label !== "Export Workflow (API Format)" ||
              exportCommand.menubarLabel !== "Export (API)"
            ) {
              throw new Error("Remote-aware API export command did not preserve the core command presentation.");
            }

            const exportedRemotePrompt = await exportCommand.function();
            if (exportedRemotePrompt !== compiledPrompt) {
              throw new Error("Remote API export did not return the compiled prompt.");
            }
            if (JSON.stringify(compileRequest) !== JSON.stringify({ workflow: preparedWorkflow, prompt: preparedPrompt })) {
              throw new Error(`Compiler did not receive the prepared graphToPrompt result: ${JSON.stringify(compileRequest)}`);
            }
            if (graphToPromptCalls !== 2) {
              throw new Error(`Remote API export should serialize before and after preflight, got ${graphToPromptCalls} calls.`);
            }
            if (
              downloads.length !== 1 ||
              downloads[0].download !== "workflow_api.json" ||
              JSON.stringify(JSON.parse(downloads[0].content)) !== JSON.stringify(compiledPrompt)
            ) {
              throw new Error(`Remote API export did not download the compiled prompt: ${JSON.stringify(downloads)}`);
            }

            globalThis.app.graph = localGraph;
            const exportedLocalPrompt = await exportCommand.function();
            if (exportedLocalPrompt !== localPrompt) {
              throw new Error("Local API export did not return the original prompt.");
            }
            if (graphToPromptCalls !== 3 || compileCalls !== 1) {
              throw new Error(`Local API export should serialize once without compiling: graphToPrompt=${graphToPromptCalls}, compile=${compileCalls}`);
            }
            if (
              downloads.length !== 2 ||
              downloads[1].download !== "workflow_api.json" ||
              JSON.stringify(JSON.parse(downloads[1].content)) !== JSON.stringify(localPrompt)
            ) {
              throw new Error(`Local API export did not download the original prompt: ${JSON.stringify(downloads)}`);
            }
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
            """
        )
        result = subprocess.run(
            ["node", "-", str(REMOTE_MODELS_JS)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_partially_overlapping_node_is_not_silently_compiled_as_remote(self):
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

            const extensions = [];
            let forwarded = 0;
            const apiListeners = {};
            const timers = [];
            const node = {
              id: 12,
              comfyClass: "PartialNode",
              boundingRect: [250, 100, 100, 80],
              widgets: [],
            };
            globalThis.window = {
              location: { origin: "http://127.0.0.1:8188" },
              setTimeout(callback) {
                timers.push(callback);
                return timers.length;
              },
              clearTimeout() {},
              addEventListener() {},
            };
            globalThis.app = {
              graph: {
                _nodes: [node],
                _groups: [{ title: "127.0.0.1:8189", boundingRect: [0, 0, 300, 300] }],
                links: {},
                setDirtyCanvas() {},
              },
              canvas: null,
              registerExtension(extension) {
                extensions.push(extension);
              },
            };
            node.graph = globalThis.app.graph;
            globalThis.api = {
              addEventListener(name, listener) {
                apiListeners[name] = listener;
              },
              async fetchApi() {
                forwarded += 1;
                return { ok: true, async json() { return { ok: true }; } };
              },
            };

            vm.runInThisContext(source, { filename: sourcePath });
            const extension = extensions.find((item) => item.name === "Cutlery.RemoteModels");
            extension.setup();
            let blocked = false;
            try {
              await globalThis.api.fetchApi("/prompt", {
                method: "POST",
                body: JSON.stringify({ prompt: { "12": { class_type: "PartialNode", inputs: {} } } }),
              });
            } catch (error) {
              blocked = String(error).includes("partially overlaps");
            }
            if (!blocked || forwarded !== 0) {
              throw new Error(`Partial overlap should block before dispatch; blocked=${blocked} forwarded=${forwarded}`);
            }
            if (!node.__cutleryRemoteOptionsError?.includes("partially overlaps")) {
              throw new Error(`Expected visible partial-overlap error, got ${node.__cutleryRemoteOptionsError}`);
            }

            node.boundingRect = [400, 100, 100, 80];
            apiListeners.graphChanged();
            while (timers.length) {
              await timers.shift()();
            }
            if (node.__cutleryRemoteOptionsError || node.__cutleryRemoteOptionsStatus) {
              throw new Error(`Partial-overlap error should clear after moving the node out of the group: ${node.__cutleryRemoteOptionsError}`);
            }
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
            """
        )
        result = subprocess.run(
            ["node", "-", str(REMOTE_MODELS_JS)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_remote_group_executor_frontend_labels_ports_from_serialized_contract(self):
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

            const timers = [];
            const extensions = [];
            globalThis.window = {
              setTimeout(callback) {
                timers.push(callback);
                callback();
                return timers.length;
              },
              clearTimeout() {},
              addEventListener() {},
            };
            globalThis.app = {
              graph: { _nodes: [], _groups: [], setDirtyCanvas() {} },
              canvas: null,
              registerExtension(extension) {
                extensions.push(extension);
              },
            };
            globalThis.api = { async fetchApi() {} };

            vm.runInThisContext(source, { filename: sourcePath });
            const extension = extensions.find((item) => item.name === "Cutlery.RemoteModels");
            for (const className of ["CutleryRemoteGroupExecutor", "CutleryRemoteGroupValueExecutor"]) {
              const nodeType = function FakeRemoteGroupExecutor() {};
              const nodeData = {
                name: className,
                input: {
                  required: {
                    remote_base_url: ["STRING", {}],
                    remote_workflow_json: ["STRING", {}],
                    input_ports_json: ["STRING", {}],
                    output_ports_json: ["STRING", {}],
                    timeout_seconds: ["FLOAT", {}],
                  },
                  optional: { value_1: ["*", {}], value_2: ["*", {}], value_3: ["*", {}] },
                },
                input_order: { optional: ["value_1", "value_2", "value_3"] },
                output: ["*", "*", "*"],
                output_name: ["value_1", "value_2", "value_3"],
              };
              extension.beforeRegisterNodeDef(nodeType, nodeData);
              const node = new nodeType();
              node.widgets = [
                { name: "remote_base_url", value: "192.0.2.247:8188" },
                { name: "remote_workflow_json", value: "{}" },
                { name: "input_ports_json", value: '[{"name":"image","type":"IMAGE"},{"name":"latent","type":"LATENT"}]' },
                { name: "output_ports_json", value: '[{"name":"result","type":"IMAGE"}]' },
              ];
              node.inputs = [];
              node.outputs = [];
              node.addInput = function addInput(name, type) {
                this.inputs.push({ name, type });
              };
              node.addOutput = function addOutput(name, type) {
                this.outputs.push({ name, type });
              };
              node.removeInput = function removeInput(index) {
                this.inputs.splice(index, 1);
              };
              node.removeOutput = function removeOutput(index) {
                this.outputs.splice(index, 1);
              };
              node.computeSize = () => [260, 120];
              node.setSize = function setSize(size) {
                this.size = size;
              };

              node.onNodeCreated();
              if (node.inputs.map((input) => `${input.name}:${input.type}`).join(",") !== "image:IMAGE,latent:LATENT") {
                throw new Error(`Expected labelled inputs, got ${JSON.stringify(node.inputs)}`);
              }
              if (node.outputs.map((output) => `${output.name}:${output.type}`).join(",") !== "result:IMAGE") {
                throw new Error(`Expected labelled outputs, got ${JSON.stringify(node.outputs)}`);
              }
              if (!node.widgets.every((widget) => widget.hidden || widget.name === "timeout_seconds")) {
                throw new Error("Executor transport widgets should be hidden.");
              }
            }
            })();
            """
        )
        result = subprocess.run(
            ["node", "-", str(REMOTE_MODELS_JS)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_prompt_fetch_hook_compiles_remote_group_nested_in_subgraphs(self):
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

            const extensions = [];
            let capturedBody = null;
            let capturedCompileBody = null;
            const sourceNode = {
              id: 1,
              type: "Source",
              pos: [0, 130],
              size: [80, 60],
              outputs: [{ name: "IMAGE", type: "IMAGE" }],
            };
            const remoteNode = {
              id: 2,
              type: "RemoteWork",
              comfyClass: "RemoteWork",
              boundingRect: [0, 0, 0, 0],
              pos: [130, 130],
              size: [80, 300],
              flags: { collapsed: true },
              inputs: [{ name: "image", type: "IMAGE" }],
              outputs: [{ name: "IMAGE", type: "IMAGE" }],
              widgets: [],
            };
            const sinkNode = {
              id: 3,
              type: "Sink",
              pos: [360, 130],
              size: [80, 60],
              inputs: [{ name: "image", type: "IMAGE" }],
            };
            const inner = {
              id: "inner-subgraph",
              _groups: [{ title: "192.0.2.247:8188", boundingRect: [100, 100, 200, 200] }],
              _nodes: [sourceNode, remoteNode, sinkNode],
              _subgraphs: new Map(),
              links: {
                10: { id: 10, origin_id: 1, origin_slot: 0, target_id: 2, target_slot: 0, type: "IMAGE" },
                11: { id: 11, origin_id: 2, origin_slot: 0, target_id: 3, target_slot: 0, type: "IMAGE" },
              },
            };
            for (const node of inner._nodes) {
              node.graph = inner;
            }
            const innerInstance = { id: 40, type: inner.id, subgraph: inner, pos: [0, 0], size: [100, 100] };
            const outer = {
              id: "outer-subgraph",
              _groups: [],
              _nodes: [innerInstance],
              _subgraphs: new Map([[inner.id, inner]]),
              links: {},
            };
            innerInstance.graph = outer;
            const outerInstance = { id: 30, type: outer.id, subgraph: outer, pos: [0, 0], size: [100, 100] };
            const root = {
              _groups: [],
              _nodes: [outerInstance],
              _subgraphs: new Map([[outer.id, outer], [inner.id, inner]]),
              links: {},
              setDirtyCanvas() {},
            };
            outerInstance.graph = root;

            globalThis.window = {
              location: { origin: "http://127.0.0.1:8188" },
              setTimeout() { return 1; },
              clearTimeout() {},
              addEventListener() {},
            };
            globalThis.document = { activeElement: null };
            globalThis.app = {
              graph: root,
              canvas: null,
              registerExtension(extension) { extensions.push(extension); },
            };
            globalThis.api = {
              async fetchApi(path, options) {
                if (String(path) === "/cutlery/remote/proxy/node-definitions") {
                  const request = JSON.parse(options.body);
                  return {
                    ok: true,
                    async json() {
                      return {
                        ok: true,
                        nodes: Object.fromEntries(
                          request.class_types.map((classType) => [
                            classType,
                            {
                              available: true,
                              compatible: true,
                              cache: { declared_inputs_only: true },
                              inputs: { required: {}, optional: {}, hidden: {} },
                            },
                          ]),
                        ),
                      };
                    },
                  };
                }
                if (String(path) === "/cutlery/remote/compile") {
                  capturedCompileBody = JSON.parse(options.body);
                  const remoteId = "30:40:2";
                  const wrapperId = "cutlery_remote_group_1";
                  const prompt = JSON.parse(JSON.stringify(capturedCompileBody.prompt));
                  delete prompt[remoteId];
                  prompt["30:40:3"].inputs.image = [wrapperId, 0];
                  prompt[wrapperId] = {
                    class_type: "CutleryRemoteGroupExecutor",
                    inputs: {
                      remote_base_url: "192.0.2.247:8188",
                      remote_workflow_json: JSON.stringify({
                        [remoteId]: { class_type: "RemoteWork", inputs: { image: ["cutlery_remote_input_1_1", 0] } },
                      }),
                      value_1: ["30:40:1", 0],
                    },
                  };
                  return {
                    ok: true,
                    async json() {
                      return { ok: true, prompt, remaps: { [remoteId]: wrapperId } };
                    },
                  };
                }
                capturedBody = JSON.parse(options.body);
                return { ok: true, async json() { return { ok: true }; } };
              },
            };

            vm.runInThisContext(source, { filename: sourcePath });
            const extension = extensions.find((item) => item.name === "Cutlery.RemoteModels");
            extension.setup();
            if (globalThis.cutleryRemoteGroups.nodeRemoteTarget(remoteNode) !== "192.0.2.247:8188") {
              throw new Error("Nested remote node did not inherit its enclosing target.");
            }

            const remoteId = "30:40:2";
            const serializedWorkflow = {
              nodes: [{ id: 30, type: "outer-subgraph" }],
              groups: [],
              links: [],
              definitions: {
                subgraphs: [
                  { id: "outer-subgraph", nodes: [{ id: 40, type: "inner-subgraph" }], groups: [], links: [] },
                  {
                    id: "inner-subgraph",
                    nodes: [
                      { id: 1, type: "Source", pos: [0, 130], size: [80, 60] },
                      { id: 2, type: "RemoteWork", pos: [130, 130], size: [80, 300] },
                      { id: 3, type: "Sink", pos: [360, 130], size: [80, 60] },
                    ],
                    groups: [{ title: "192.0.2.247:8188", bounding: [100, 100, 200, 200] }],
                    links: [],
                  },
                ],
              },
            };
            const requestBody = {
              prompt: {
                "30:40:1": { class_type: "Source", inputs: {} },
                [remoteId]: { class_type: "RemoteWork", inputs: { image: ["30:40:1", 0] } },
                "30:40:3": { class_type: "Sink", inputs: { image: [remoteId, 0] } },
              },
              partial_execution_targets: [remoteId],
              extra_data: { extra_pnginfo: { workflow: serializedWorkflow } },
            };
            await globalThis.api.fetchApi("/prompt", { method: "POST", body: JSON.stringify(requestBody) });

            if (!capturedCompileBody || capturedCompileBody.workflow !== serializedWorkflow) {
              if (JSON.stringify(capturedCompileBody?.workflow) !== JSON.stringify(serializedWorkflow)) {
                throw new Error(`Serialized workflow was not sent to the canonical compiler: ${JSON.stringify(capturedCompileBody)}`);
              }
            }
            if (JSON.stringify(capturedCompileBody.partial_execution_targets) !== JSON.stringify([remoteId])) {
              throw new Error(`Original partial target was not sent to canonical compilation: ${JSON.stringify(capturedCompileBody)}`);
            }

            const prompt = capturedBody.prompt;
            const wrapperId = Object.keys(prompt).find((id) => prompt[id].class_type === "CutleryRemoteGroupExecutor");
            if (!wrapperId || prompt[remoteId]) {
              throw new Error(`Nested remote node was not replaced: ${JSON.stringify(prompt)}`);
            }
            if (JSON.stringify(prompt["30:40:3"].inputs.image) !== JSON.stringify([wrapperId, 0])) {
              throw new Error(`Nested downstream link was not remapped: ${JSON.stringify(prompt)}`);
            }
            if (JSON.stringify(prompt[wrapperId].inputs.value_1) !== JSON.stringify(["30:40:1", 0])) {
              throw new Error(`Nested inbound link was not preserved: ${JSON.stringify(prompt[wrapperId])}`);
            }
            if (JSON.stringify(capturedBody.partial_execution_targets) !== JSON.stringify([wrapperId])) {
              throw new Error(`Nested partial target was not remapped: ${JSON.stringify(capturedBody)}`);
            }
            const remoteWorkflow = JSON.parse(prompt[wrapperId].inputs.remote_workflow_json);
            if (!remoteWorkflow[remoteId] || remoteWorkflow[remoteId].class_type !== "RemoteWork") {
              throw new Error(`Nested remote workflow lost its prompt identity: ${JSON.stringify(remoteWorkflow)}`);
            }
            })().catch((error) => {
              console.error(error);
              process.exit(1);
            });
            """
        )
        result = subprocess.run(
            ["node", "-", str(REMOTE_MODELS_JS)],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
