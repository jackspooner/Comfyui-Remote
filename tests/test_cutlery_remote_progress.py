import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutlery_remote.progress import (
    ProgressEventError,
    ProgressMappingError,
    ProgressMirror,
    parse_progress_mapping,
)


def _identity(api_node_id="original-1", visible=True):
    return {
        "api_node_id": api_node_id,
        "display_node_id": api_node_id,
        "parent_node_id": "",
        "real_node_id": "",
        "subgraph_instance": "",
        "group": "group-1",
        "target": "renderhost",
        "visible": visible,
    }


class ProgressMirrorTests(unittest.TestCase):
    def test_mapping_is_immutable_and_requires_canonical_identity(self):
        mapping = parse_progress_mapping({"nodes": {"peer-1": _identity()}})

        self.assertEqual(mapping["peer-1"].api_node_id, "original-1")
        with self.assertRaises(TypeError):
            mapping["peer-2"] = mapping["peer-1"]
        invalid = _identity()
        invalid.pop("target")
        with self.assertRaises(ProgressMappingError):
            parse_progress_mapping({"peer-1": invalid})

    def test_invisible_helpers_may_omit_presentation_ids(self):
        helper = _identity(visible=False)
        helper["api_node_id"] = ""
        helper["display_node_id"] = ""

        mapping = parse_progress_mapping({"peer-helper": helper})

        self.assertFalse(mapping["peer-helper"].visible)

    def test_aggregates_duplicate_reconstructions_with_local_contribution(self):
        updates = []
        clock = [0.0]
        mirror = ProgressMirror(
            local_prompt_id="local-prompt",
            remote_prompt_id="remote-prompt",
            mapping={"peer-a": _identity(), "peer-b": _identity()},
            emitter=updates.append,
            clock=lambda: clock[0],
        )

        first = mirror.ingest({"prompt_id": "remote-prompt", "node": "peer-a", "value": 2, "max": 10})
        self.assertTrue(first.first)
        self.assertEqual((first.value, first.max_value), (2, 10))
        self.assertIsNone(mirror.ingest({"prompt_id": "remote-prompt", "node": "peer-b", "value": 3, "max": 10}))
        mirror.set_local_progress("original-1", 1, 2)
        clock[0] = 0.1
        flushed = mirror.flush()

        self.assertEqual(len(flushed), 1)
        self.assertEqual((flushed[0].value, flushed[0].max_value), (6, 22))
        self.assertEqual(flushed[0].as_progress_data()["node"], "original-1")

    def test_first_and_terminal_events_bypass_coalescing_and_helpers_are_hidden(self):
        updates = []
        mirror = ProgressMirror(
            local_prompt_id="local-prompt",
            remote_prompt_id="remote-prompt",
            mapping={"peer-main": _identity(), "peer-helper": _identity(visible=False)},
            emitter=updates.append,
            clock=lambda: 0.0,
        )

        self.assertIsNone(mirror.ingest({"prompt_id": "remote-prompt", "node": "peer-helper", "value": 1, "max": 1}))
        first = mirror.ingest({"prompt_id": "remote-prompt", "node": "peer-main", "value": 1, "max": 2})
        terminal = mirror.ingest({"prompt_id": "remote-prompt", "node": "peer-main", "value": 2, "max": 2})

        self.assertTrue(first.first)
        self.assertTrue(terminal.terminal)
        self.assertEqual(len(updates), 2)

    def test_rejects_unknown_prompt_node_and_out_of_order_events_then_clears(self):
        mirror = ProgressMirror(
            local_prompt_id="local-prompt",
            remote_prompt_id="remote-prompt",
            mapping={"peer-1": _identity()},
            emitter=lambda _: None,
        )

        with self.assertRaises(ProgressEventError):
            mirror.ingest({"prompt_id": "other", "node": "peer-1", "value": 1, "max": 2})
        with self.assertRaises(ProgressEventError):
            mirror.ingest({"prompt_id": "remote-prompt", "node": "unknown", "value": 1, "max": 2})
        mirror.ingest({"prompt_id": "remote-prompt", "node": "peer-1", "value": 1, "max": 2, "sequence": 2})
        with self.assertRaises(ProgressEventError):
            mirror.ingest({"prompt_id": "remote-prompt", "node": "peer-1", "value": 1, "max": 2, "sequence": 1})
        mirror.cancel()
        self.assertTrue(mirror.closed)
        with self.assertRaises(ProgressEventError):
            mirror.ingest({"prompt_id": "remote-prompt", "node": "peer-1", "value": 2, "max": 2})


if __name__ == "__main__":
    unittest.main()
