import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nodes_remote_proxy  # noqa: E402
from cutlery_remote.target import TrustedRemoteTarget  # noqa: E402


class RemoteProxyNodeTests(unittest.TestCase):
    def test_cached_remote_schema_registers_non_executable_local_proxy(self):
        definition = {
            "input": {
                "required": {
                    "steps": ["INT", {"default": 12, "min": 1, "max": 100}],
                    "pipeline": ["TRELLIS2PIPELINE"],
                }
            },
            "output": ["SHAPE_SLAT", "TRELLIS2PIPELINE"],
            "output_name": ["shape_slat", "pipeline"],
            "output_is_list": [False, False],
            "category": "Trellis2Wrapper",
            "display_name": "Trellis2 - Shape Generator",
        }
        target = TrustedRemoteTarget(
            name="trellis2",
            base_url="http://127.0.0.1:8890",
            canonical="cutlery://trellis2",
            display_label="TRELLIS.2",
            expose_node_prefixes=("Trellis2",),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = Path(temp_dir) / "trellis2.json"
            catalog_path.write_text(
                json.dumps({"schema_version": 1, "nodes": {"Trellis2ShapeGenerator": definition}}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(nodes_remote_proxy, "configured_remote_targets", return_value={"trellis2": target}),
                mock.patch.object(nodes_remote_proxy, "_catalog_path", return_value=catalog_path),
            ):
                classes, names = nodes_remote_proxy.load_remote_proxy_nodes()

        proxy = classes["Trellis2ShapeGenerator"]
        self.assertEqual(proxy.INPUT_TYPES(), definition["input"])
        self.assertEqual(proxy.RETURN_TYPES, ("SHAPE_SLAT", "TRELLIS2PIPELINE"))
        self.assertEqual(proxy.CATEGORY, "Remote/trellis2/Trellis2Wrapper")
        self.assertIn("[Remote: trellis2]", names["Trellis2ShapeGenerator"])
        with self.assertRaisesRegex(RuntimeError, "inside that remote group"):
            proxy()._remote_proxy_error()


if __name__ == "__main__":
    unittest.main()
