from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
for path in (REPO_ROOT, COMFY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import execution
import nodes
from cutlery_remote.group_compiler import compile_editor_remote_groups
from comfy_execution.graph import DynamicPrompt, TopologicalSort
from comfy_extras.nodes_logic import SwitchNode


class _Source:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


class _Preparation:
    RETURN_TYPES = ("PREPARATION",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


class _ValueExecutor:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"preparation": ("PREPARATION", {})}}


class _TerminalExecutor:
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}


class _Sink:
    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE", {})}}


class RemoteLazySchedulerTests(unittest.TestCase):
    def test_lazy_switch_adds_remote_value_branch_only_when_selected(self):
        prompt = {
            "source": {"class_type": "TestSource", "inputs": {}},
            "preparation": {"class_type": "TestPreparation", "inputs": {}},
            "value": {"class_type": "TestValueExecutor", "inputs": {"preparation": ["preparation", 0]}},
            "switch": {
                "class_type": "ComfySwitchNode",
                "inputs": {"switch": False, "on_false": ["source", 0], "on_true": ["value", 0]},
            },
            "sink": {"class_type": "TestSink", "inputs": {"image": ["switch", 0]}},
        }
        patched = {
            "TestSource": _Source,
            "TestPreparation": _Preparation,
            "TestValueExecutor": _ValueExecutor,
            "ComfySwitchNode": SwitchNode,
            "TestSink": _Sink,
        }
        previous = {name: nodes.NODE_CLASS_MAPPINGS.get(name) for name in patched}
        nodes.NODE_CLASS_MAPPINGS.update(patched)
        try:
            false_execution = TopologicalSort(DynamicPrompt(prompt))
            false_execution.add_node("sink")
            self.assertEqual(set(false_execution.pendingNodes), {"sink", "switch"})
            false_execution.make_input_strong_link("switch", "on_false")
            self.assertEqual(set(false_execution.pendingNodes), {"sink", "switch", "source"})

            true_execution = TopologicalSort(DynamicPrompt(prompt))
            true_execution.add_node("sink")
            self.assertEqual(set(true_execution.pendingNodes), {"sink", "switch"})
            true_execution.make_input_strong_link("switch", "on_true")
            self.assertEqual(set(true_execution.pendingNodes), {"sink", "switch", "value", "preparation"})

            valid = asyncio.run(execution.validate_prompt("remote-lazy-test", prompt, ["sink"]))
            self.assertTrue(valid[0], valid[1])
        finally:
            for name, original in previous.items():
                if original is None:
                    del nodes.NODE_CLASS_MAPPINGS[name]
                else:
                    nodes.NODE_CLASS_MAPPINGS[name] = original

    def test_partial_remote_group_promotes_a_validated_terminal_target(self):
        workflow = {
            "nodes": [
                {"id": 1, "type": "Source", "pos": [-200, 100], "size": [100, 60], "outputs": [{"name": "image", "type": "IMAGE", "links": [1]}]},
                {"id": 2, "type": "RemoteNode", "pos": [100, 100], "size": [160, 100], "inputs": [{"name": "image", "type": "IMAGE", "link": 1}], "outputs": [{"name": "image", "type": "IMAGE", "links": [2]}]},
                {"id": 3, "type": "Consumer", "pos": [500, 100], "size": [160, 100], "inputs": [{"name": "image", "type": "IMAGE", "link": 2}]},
            ],
            "links": [[1, 1, 0, 2, 0, "IMAGE"], [2, 2, 0, 3, 0, "IMAGE"]],
            "groups": [{"title": "127.0.0.1:8889", "bounding": [50, 50, 260, 200]}],
        }
        prompt = {
            "1": {"class_type": "Source", "inputs": {}},
            "2": {"class_type": "RemoteNode", "inputs": {"image": ["1", 0]}},
            "3": {"class_type": "Consumer", "inputs": {"image": ["2", 0]}},
        }

        compiled, remaps, _targets = compile_editor_remote_groups(
            workflow,
            prompt,
            partial_execution_targets=["2"],
        )
        wrapper_id = remaps["2"]
        self.assertEqual(compiled[wrapper_id]["class_type"], "CutleryRemoteGroupExecutor")

        original = nodes.NODE_CLASS_MAPPINGS.get("CutleryRemoteGroupExecutor")
        nodes.NODE_CLASS_MAPPINGS["CutleryRemoteGroupExecutor"] = _TerminalExecutor
        try:
            validation_prompt = {wrapper_id: compiled[wrapper_id] | {"inputs": {}}}
            valid = asyncio.run(execution.validate_prompt("partial-remote-test", validation_prompt, [wrapper_id]))
            self.assertTrue(valid[0], valid[1])
        finally:
            if original is None:
                del nodes.NODE_CLASS_MAPPINGS["CutleryRemoteGroupExecutor"]
            else:
                nodes.NODE_CLASS_MAPPINGS["CutleryRemoteGroupExecutor"] = original


if __name__ == "__main__":
    unittest.main()
