from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutlery_remote.group_compiler import compile_editor_remote_groups


def _editor_workflow():
    return {
        "nodes": [
            {
                "id": 1,
                "type": "Source",
                "pos": [-200, 100],
                "size": [100, 60],
                "outputs": [{"name": "text", "type": "STRING", "links": [1]}],
            },
            {
                "id": 2,
                "type": "RemoteNode",
                "pos": [100, 100],
                "size": [160, 100],
                "inputs": [{"name": "text", "type": "STRING", "link": 1}],
                "outputs": [{"name": "conditioning", "type": "CONDITIONING", "links": [2]}],
            },
            {
                "id": 3,
                "type": "Consumer",
                "pos": [500, 100],
                "size": [160, 100],
                "inputs": [{"name": "conditioning", "type": "CONDITIONING", "link": 2}],
            },
        ],
        "links": [
            [1, 1, 0, 2, 0, "STRING"],
            [2, 2, 0, 3, 0, "CONDITIONING"],
        ],
        "groups": [
            {
                "title": "127.0.0.1:8889",
                "bounding": [50, 50, 260, 200],
            }
        ],
    }


def _api_prompt():
    return {
        "1": {"class_type": "Source", "inputs": {}},
        "2": {"class_type": "RemoteNode", "inputs": {"text": ["1", 0]}},
        "3": {"class_type": "Consumer", "inputs": {"conditioning": ["2", 0]}},
    }


def _nested_workflow(*, depth=1, instance_ids=(30,)):
    inner = {
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
    definitions = [inner]
    definition_id = "inner"
    nested_suffix = "2"
    if depth == 2:
        definitions.append(
            {
                "id": "outer",
                "nodes": [{"id": 40, "type": "inner", "pos": [0, 0], "size": [100, 100]}],
                "links": [],
                "groups": [],
            }
        )
        definition_id = "outer"
        nested_suffix = "40:2"

    root_nodes = [
        {
            "id": 1,
            "type": "Source",
            "pos": [-200, 100],
            "size": [100, 60],
            "outputs": [{"name": "image", "type": "IMAGE"}],
        }
    ]
    prompt = {"1": {"class_type": "Source", "inputs": {}}}
    remote_ids = []
    sink_ids = []
    for index, instance_id in enumerate(instance_ids):
        root_nodes.append(
            {"id": instance_id, "type": definition_id, "pos": [0, 0], "size": [100, 100]}
        )
        sink_id = 100 + index
        root_nodes.append(
            {
                "id": sink_id,
                "type": "Consumer",
                "pos": [500, 100 + index * 100],
                "size": [160, 100],
                "inputs": [{"name": "image", "type": "IMAGE"}],
            }
        )
        remote_id = f"{instance_id}:{nested_suffix}"
        remote_ids.append(remote_id)
        sink_ids.append(str(sink_id))
        prompt[remote_id] = {"class_type": "RemoteNode", "inputs": {"image": ["1", 0]}}
        prompt[str(sink_id)] = {"class_type": "Consumer", "inputs": {"image": [remote_id, 0]}}

    workflow = {
        "nodes": root_nodes,
        "links": [],
        "groups": [],
        "definitions": {"subgraphs": definitions},
    }
    return workflow, prompt, remote_ids, sink_ids


class WF3RemoteGroupCompilerTests(unittest.TestCase):
    def test_compiles_editor_group_into_remote_executor(self):
        prompt, remaps, targets = compile_editor_remote_groups(_editor_workflow(), _api_prompt())

        self.assertNotIn("2", prompt)
        self.assertEqual(remaps, {"2": "cutlery_remote_group_1"})
        self.assertEqual(targets, ["127.0.0.1:8889"])
        wrapper = prompt["cutlery_remote_group_1"]
        self.assertEqual(wrapper["class_type"], "CutleryRemoteGroupValueExecutor")
        self.assertEqual(wrapper["inputs"]["remote_base_url"], "127.0.0.1:8889")
        self.assertEqual(json.loads(wrapper["inputs"]["input_ports_json"]), [{"name": "input_1", "type": "string"}])
        self.assertEqual(json.loads(wrapper["inputs"]["output_ports_json"]), [{"name": "output_1", "type": "json"}])
        self.assertEqual(wrapper["inputs"]["value_1"], ["1", 0])

        remote_prompt = json.loads(wrapper["inputs"]["remote_workflow_json"])
        self.assertEqual(remote_prompt["2"]["inputs"]["text"], ["cutlery_remote_input_1_1", 0])
        self.assertEqual(
            remote_prompt["cutlery_remote_encode_1_1"]["class_type"],
            "WF3ConditioningToBlob",
        )
        self.assertEqual(prompt["cutlery_remote_decode_1_1"]["class_type"], "WF3ConditioningFromBlob")
        self.assertEqual(prompt["3"]["inputs"]["conditioning"], ["cutlery_remote_decode_1_1", 0])

    def test_compiles_labelled_group_using_only_its_endpoint(self):
        workflow = _editor_workflow()
        workflow["groups"][0]["title"] = "127.0.0.1:8889 // Local GPU"

        prompt, _remaps, targets = compile_editor_remote_groups(workflow, _api_prompt())

        self.assertEqual(targets, ["127.0.0.1:8889"])
        self.assertEqual(prompt["cutlery_remote_group_1"]["inputs"]["remote_base_url"], "127.0.0.1:8889")

    def test_compiles_group_without_outbound_ports_as_terminal_executor(self):
        workflow = _editor_workflow()
        prompt = _api_prompt()
        workflow["nodes"] = workflow["nodes"][:2]
        workflow["links"] = workflow["links"][:1]
        prompt.pop("3")

        compiled, remaps, _targets = compile_editor_remote_groups(workflow, prompt)

        self.assertEqual(compiled[remaps["2"]]["class_type"], "CutleryRemoteGroupExecutor")

    def test_compiles_group_containing_output_node_as_terminal_executor(self):
        resolver = {
            "RemoteNode": {
                "local": {"output_node": True},
                "remote": {"output_node": True},
            }
        }

        compiled, remaps, _targets = compile_editor_remote_groups(
            _editor_workflow(),
            _api_prompt(),
            definition_resolver=resolver,
        )

        self.assertEqual(compiled[remaps["2"]]["class_type"], "CutleryRemoteGroupExecutor")

    def test_partial_execution_target_promotes_group_to_terminal_executor(self):
        compiled, remaps, _targets = compile_editor_remote_groups(
            _editor_workflow(),
            _api_prompt(),
            partial_execution_targets=["2"],
        )

        self.assertEqual(compiled[remaps["2"]]["class_type"], "CutleryRemoteGroupExecutor")

    def test_switch_keeps_local_false_branch_and_uses_dependency_remote_true_branch(self):
        workflow = {
            "nodes": [
                {"id": 1, "type": "Source", "pos": [-300, 100], "size": [100, 60], "outputs": [{"name": "image", "type": "IMAGE"}]},
                {"id": 2, "type": "LocalFalse", "pos": [-100, 100], "size": [100, 60], "inputs": [{"name": "image", "type": "IMAGE"}], "outputs": [{"name": "image", "type": "IMAGE"}]},
                {"id": 3, "type": "RemoteTrue", "pos": [150, 100], "size": [100, 60], "inputs": [{"name": "image", "type": "IMAGE"}], "outputs": [{"name": "image", "type": "IMAGE"}]},
                {"id": 4, "type": "ComfySwitchNode", "pos": [400, 100], "size": [100, 60], "inputs": [{"name": "on_false", "type": "IMAGE"}, {"name": "on_true", "type": "IMAGE"}, {"name": "switch", "type": "BOOLEAN"}], "outputs": [{"name": "image", "type": "IMAGE"}]},
                {"id": 5, "type": "SaveImage", "pos": [650, 100], "size": [100, 60], "inputs": [{"name": "images", "type": "IMAGE"}]},
                {"id": 6, "type": "PrimitiveBoolean", "pos": [150, 250], "size": [100, 60], "outputs": [{"name": "boolean", "type": "BOOLEAN"}]},
            ],
            "links": [
                [1, 1, 0, 2, 0, "IMAGE"],
                [2, 1, 0, 3, 0, "IMAGE"],
                [3, 2, 0, 4, 0, "IMAGE"],
                [4, 3, 0, 4, 1, "IMAGE"],
                [5, 4, 0, 5, 0, "IMAGE"],
                [6, 6, 0, 4, 2, "BOOLEAN"],
            ],
            "groups": [{"title": "127.0.0.1:8889", "bounding": [100, 50, 200, 160]}],
        }
        prompt = {
            "1": {"class_type": "Source", "inputs": {}},
            "2": {"class_type": "LocalFalse", "inputs": {"image": ["1", 0]}},
            "3": {"class_type": "RemoteTrue", "inputs": {"image": ["1", 0]}},
            "4": {"class_type": "ComfySwitchNode", "inputs": {"on_false": ["2", 0], "on_true": ["3", 0], "switch": ["6", 0]}},
            "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}},
            "6": {"class_type": "PrimitiveBoolean", "inputs": {"value": True}},
        }

        compiled, remaps, _targets = compile_editor_remote_groups(workflow, prompt)

        wrapper_id = remaps["3"]
        self.assertEqual(compiled[wrapper_id]["class_type"], "CutleryRemoteGroupValueExecutor")
        self.assertEqual(compiled["2"]["inputs"]["image"], ["1", 0])
        self.assertEqual(compiled["4"]["inputs"]["on_false"], ["2", 0])
        self.assertEqual(compiled["4"]["inputs"]["on_true"], [wrapper_id, 0])
        self.assertEqual(compiled["4"]["inputs"]["switch"], ["6", 0])
        self.assertEqual(compiled["5"]["inputs"]["images"], ["4", 0])

    def test_rejects_partial_node_overlap(self):
        workflow = _editor_workflow()
        workflow["nodes"][1]["pos"] = [250, 100]

        with self.assertRaisesRegex(ValueError, "partially overlaps"):
            compile_editor_remote_groups(workflow, _api_prompt())

    def test_leaves_api_prompt_unchanged_when_editor_has_no_remote_groups(self):
        workflow = _editor_workflow()
        workflow["groups"] = [{"title": "Size", "bounding": [50, 50, 260, 200]}]
        prompt = _api_prompt()

        compiled, remaps, targets = compile_editor_remote_groups(workflow, prompt)

        self.assertIs(compiled, prompt)
        self.assertEqual(remaps, {})
        self.assertEqual(targets, [])

    def test_compiles_remote_group_inside_subgraph_with_qualified_ids(self):
        workflow, api_prompt, remote_ids, sink_ids = _nested_workflow()

        prompt, remaps, targets = compile_editor_remote_groups(workflow, api_prompt)

        remote_id = remote_ids[0]
        wrapper_id = remaps[remote_id]
        self.assertEqual(targets, ["127.0.0.1:8889"])
        self.assertNotIn(remote_id, prompt)
        self.assertEqual(prompt[sink_ids[0]]["inputs"]["image"], [wrapper_id, 0])
        remote_prompt = json.loads(prompt[wrapper_id]["inputs"]["remote_workflow_json"])
        self.assertEqual(remote_prompt[remote_id]["class_type"], "RemoteNode")
        progress = json.loads(prompt[wrapper_id]["inputs"]["progress_map_json"])
        self.assertEqual(progress[remote_id]["display_node_id"], "2")
        self.assertEqual(progress[remote_id]["parent_node_id"], "30")
        self.assertEqual(progress[remote_id]["real_node_id"], "2")
        self.assertEqual(progress[remote_id]["subgraph_instance"], "30")

    def test_compiles_two_nested_subgraph_levels(self):
        workflow, api_prompt, remote_ids, sink_ids = _nested_workflow(depth=2)

        prompt, remaps, targets = compile_editor_remote_groups(workflow, api_prompt)

        remote_id = remote_ids[0]
        wrapper_id = remaps[remote_id]
        self.assertEqual(remote_id, "30:40:2")
        self.assertEqual(targets, ["127.0.0.1:8889"])
        self.assertEqual(prompt[sink_ids[0]]["inputs"]["image"], [wrapper_id, 0])
        progress = json.loads(prompt[wrapper_id]["inputs"]["progress_map_json"])
        self.assertEqual(progress[remote_id]["parent_node_id"], "30:40")
        self.assertEqual(progress[remote_id]["subgraph_instance"], "30:40")

    def test_repeated_subgraph_instances_compile_independently(self):
        workflow, api_prompt, remote_ids, sink_ids = _nested_workflow(instance_ids=(30, 31))

        prompt, remaps, targets = compile_editor_remote_groups(workflow, api_prompt)

        self.assertEqual(targets, ["127.0.0.1:8889", "127.0.0.1:8889"])
        self.assertEqual(set(remaps), set(remote_ids))
        self.assertNotEqual(remaps[remote_ids[0]], remaps[remote_ids[1]])
        for remote_id, sink_id in zip(remote_ids, sink_ids):
            self.assertEqual(prompt[sink_id]["inputs"]["image"], [remaps[remote_id], 0])

    def test_inactive_nested_remote_branch_does_not_compile(self):
        workflow, api_prompt, remote_ids, _ = _nested_workflow()
        api_prompt.pop(remote_ids[0])
        api_prompt["100"]["inputs"] = {"image": ["1", 0]}

        prompt, remaps, targets = compile_editor_remote_groups(workflow, api_prompt)

        self.assertIs(prompt, api_prompt)
        self.assertEqual(remaps, {})
        self.assertEqual(targets, [])

    def test_nested_switch_branch_compiles_only_selected_remote_encoders(self):
        workflow = {
            "nodes": [{"id": 1190, "type": "qwen", "pos": [0, 0], "size": [100, 100]}],
            "links": [],
            "groups": [],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "qwen",
                        "nodes": [
                            {"id": 4, "type": "TextEncode", "pos": [-400, 0], "size": [100, 80]},
                            {"id": 7, "type": "TextEncode", "pos": [-400, 100], "size": [100, 80]},
                            {"id": 1179, "type": "TextEncode", "pos": [100, 100], "size": [100, 80]},
                            {"id": 1180, "type": "TextEncode", "pos": [100, 200], "size": [100, 80]},
                        ],
                        "links": [],
                        "groups": [{"title": "127.0.0.1:8889", "bounding": [50, 50, 200, 260]}],
                    }
                ]
            },
        }
        remote_prompt = {
            "1190:1179": {"class_type": "TextEncode", "inputs": {}},
            "1190:1180": {"class_type": "TextEncode", "inputs": {}},
        }
        local_prompt = {
            "1190:4": {"class_type": "TextEncode", "inputs": {}},
            "1190:7": {"class_type": "TextEncode", "inputs": {}},
        }

        compiled, remaps, targets = compile_editor_remote_groups(workflow, remote_prompt)
        unchanged, local_remaps, local_targets = compile_editor_remote_groups(workflow, local_prompt)

        self.assertEqual(set(remaps), {"1190:1179", "1190:1180"})
        self.assertEqual(targets, ["127.0.0.1:8889"])
        self.assertEqual(compiled["cutlery_remote_group_1"]["class_type"], "CutleryRemoteGroupExecutor")
        self.assertIs(unchanged, local_prompt)
        self.assertEqual(local_remaps, {})
        self.assertEqual(local_targets, [])

    def test_nested_clip_join_relocation_does_not_pull_in_model_branch(self):
        workflow = {
            "nodes": [{"id": 1190, "type": "qwen", "pos": [0, 0], "size": [100, 100]}],
            "links": [],
            "groups": [],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "qwen",
                        "nodes": [
                            {"id": 1, "type": "DiffusionLoader", "pos": [-700, 0], "size": [120, 60], "outputs": [{"name": "model", "type": "MODEL"}]},
                            {"id": 2, "type": "CutleryJoinModel", "pos": [-450, 0], "size": [120, 60], "inputs": [{"name": "model", "type": "MODEL"}], "outputs": [{"name": "model", "type": "MODEL"}]},
                            {"id": 3, "type": "CLIPLoader", "pos": [-700, 300], "size": [120, 60], "outputs": [{"name": "clip", "type": "CLIP"}]},
                            {"id": 4, "type": "CutleryJoinClip", "pos": [-200, 300], "size": [120, 60], "inputs": [{"name": "clip", "type": "CLIP"}], "outputs": [{"name": "clip", "type": "CLIP"}]},
                            {"id": 5, "type": "TextEncode", "pos": [100, 300], "size": [120, 60], "inputs": [{"name": "clip", "type": "CLIP"}], "outputs": [{"name": "conditioning", "type": "CONDITIONING"}]},
                            {"id": 6, "type": "LocalSampler", "pos": [400, 0], "size": [120, 60], "inputs": [{"name": "model", "type": "MODEL"}]},
                            {"id": 7, "type": "ConditioningSink", "pos": [400, 300], "size": [120, 60], "inputs": [{"name": "conditioning", "type": "CONDITIONING"}]},
                        ],
                        "links": [],
                        "groups": [{"title": "127.0.0.1:8889", "bounding": [50, 250, 220, 160]}],
                    }
                ]
            },
        }
        prompt = {
            "1190:1": {"class_type": "DiffusionLoader", "inputs": {}},
            "1190:2": {"class_type": "CutleryJoinModel", "inputs": {"model": ["1190:1", 0]}},
            "1190:3": {"class_type": "CLIPLoader", "inputs": {}},
            "1190:4": {"class_type": "CutleryJoinClip", "inputs": {"clip": ["1190:3", 0]}},
            "1190:5": {"class_type": "TextEncode", "inputs": {"clip": ["1190:4", 0]}},
            "1190:6": {"class_type": "LocalSampler", "inputs": {"model": ["1190:2", 0]}},
            "1190:7": {"class_type": "ConditioningSink", "inputs": {"conditioning": ["1190:5", 0]}},
        }
        clip_loader = {"inputs": {"required": {}}, "outputs": [{"type": "CLIP"}]}
        clip_join = {"inputs": {"required": {"clip": {"type": "CLIP"}}}, "outputs": [{"type": "CLIP"}]}
        resolver = {
            "CLIPLoader": {"local": clip_loader, "remote": clip_loader},
            "CutleryJoinClip": {"local": clip_join, "remote": clip_join},
        }

        compiled, remaps, targets = compile_editor_remote_groups(workflow, prompt, definition_resolver=resolver)

        wrapper = compiled[remaps["1190:5"]]
        remote_prompt = json.loads(wrapper["inputs"]["remote_workflow_json"])
        self.assertEqual(targets, ["127.0.0.1:8889"])
        self.assertTrue({"1190:3", "1190:4", "1190:5"}.issubset(remote_prompt))
        self.assertNotIn("1190:1", remote_prompt)
        self.assertNotIn("1190:2", remote_prompt)
        self.assertIn("1190:1", compiled)
        self.assertIn("1190:2", compiled)
        self.assertEqual(compiled["1190:6"]["inputs"]["model"], ["1190:2", 0])

    def test_recursive_definition_guard_stops_self_reference(self):
        workflow, api_prompt, remote_ids, _ = _nested_workflow()
        workflow["definitions"]["subgraphs"][0]["nodes"].append(
            {"id": 9, "type": "inner", "pos": [400, 400], "size": [100, 100]}
        )

        prompt, remaps, targets = compile_editor_remote_groups(workflow, api_prompt)

        self.assertEqual(targets, ["127.0.0.1:8889"])
        self.assertEqual(set(remaps), {remote_ids[0]})
        self.assertIn(remaps[remote_ids[0]], prompt)


if __name__ == "__main__":
    unittest.main()
