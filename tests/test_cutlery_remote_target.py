import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutlery_remote.target import (
    RemoteTarget,
    configured_remote_targets,
    parse_remote_target,
    remote_target_endpoint,
    resolve_trusted_remote_target,
)


class RemoteTargetTests(unittest.TestCase):
    def test_parse_plain_host_port_group_title(self):
        target = parse_remote_target("192.0.2.247:8188")

        self.assertEqual(target.host, "192.0.2.247")
        self.assertEqual(target.port, 8188)
        self.assertEqual(target.canonical, "cutlery://192.0.2.247:8188")
        self.assertEqual(target.base_url, "http://192.0.2.247:8188")
        self.assertEqual(target.display_label, "192.0.2.247:8188")

    def test_parse_curly_port_group_title(self):
        target = parse_remote_target("192.0.2.247:{8188}")

        self.assertEqual(
            target,
            RemoteTarget(
                scheme="http",
                host="192.0.2.247",
                port=8188,
                canonical="cutlery://192.0.2.247:8188",
                base_url="http://192.0.2.247:8188",
                display_label="192.0.2.247:8188",
            ),
        )

    def test_parse_canonical_cutlery_url(self):
        target = parse_remote_target("cutlery://studio-gpu:8190")

        self.assertEqual(target.host, "studio-gpu")
        self.assertEqual(target.port, 8190)
        self.assertEqual(target.canonical, "cutlery://studio-gpu:8190")
        self.assertEqual(target.base_url, "http://studio-gpu:8190")

    def test_labelled_group_title_uses_only_the_endpoint(self):
        target = parse_remote_target("127.0.0.1:8889 // Name of group")

        self.assertEqual(remote_target_endpoint("127.0.0.1:8889 // Name of group"), "127.0.0.1:8889")
        self.assertEqual(target.base_url, "http://127.0.0.1:8889")
        self.assertEqual(target.display_label, "127.0.0.1:8889")

    def test_labelled_alias_uses_only_the_alias_for_trust_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "remote_targets": {
                            "renderhost": {"base_url": "http://127.0.0.1:8189"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            target = resolve_trusted_remote_target("cutlery://renderhost // Production renderer", config_path=config_path)
            by_origin = resolve_trusted_remote_target("127.0.0.1:8189 // Render peer", config_path=config_path)

        self.assertEqual(target.name, "renderhost")
        self.assertEqual(target.base_url, "http://127.0.0.1:8189")
        self.assertEqual(by_origin, target)

    def test_legacy_single_slash_alias_resolves_only_when_configured(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "remote_targets": {
                            "trellis2": {"base_url": "http://127.0.0.1:8890"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            target = resolve_trusted_remote_target("cutlery/trellis2 // Trellis worker", config_path=config_path)

        self.assertEqual(target.name, "trellis2")
        self.assertEqual(target.base_url, "http://127.0.0.1:8890")

    def test_legacy_single_slash_alias_is_rejected_when_unconfigured(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "remote_targets": {
                            "secondpc": {"base_url": "http://127.0.0.1:8889"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "not trusted"):
                resolve_trusted_remote_target("cutlery/trellis2", config_path=config_path)

    def test_group_label_cannot_make_an_untrusted_endpoint_trusted(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "remote_targets": {
                            "renderhost": {"base_url": "http://127.0.0.1:8189"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "not trusted"):
                resolve_trusted_remote_target("192.0.2.50:8889 // renderhost", config_path=config_path)

    def test_reject_non_remote_group_titles(self):
        for title in ["Group", "192.0.2.247", "cutlery://", "http://192.0.2.247:8188"]:
            with self.subTest(title=title):
                self.assertIsNone(parse_remote_target(title))

    def test_resolve_trusted_target_from_local_alias_or_exact_origin(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "cutlery.local.json"
            config_path.write_text(
                json.dumps(
                    {
                        "remote_targets": {
                            "renderhost": {
                                "base_url": "http://127.0.0.1:8189",
                                "copy_host": "renderhost",
                                "copy_root": "D:\\ComfyUI\\models",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            targets = configured_remote_targets(config_path)
            by_alias = resolve_trusted_remote_target("cutlery://renderhost", config_path=config_path)
            by_origin = resolve_trusted_remote_target("127.0.0.1:8189", config_path=config_path)

        self.assertEqual(targets["renderhost"].base_url, "http://127.0.0.1:8189")
        self.assertEqual(by_alias, by_origin)
        self.assertEqual(by_alias.copy_host, "renderhost")
        self.assertEqual(by_alias.copy_root, "D:/ComfyUI/models")

    def test_unconfigured_loopback_and_network_origins_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "missing.json"
            with self.assertRaisesRegex(ValueError, "not trusted"):
                resolve_trusted_remote_target("127.0.0.1:8189", config_path=config_path)
            with self.assertRaisesRegex(ValueError, "not trusted"):
                resolve_trusted_remote_target("https://attacker.example:443", config_path=config_path)

    def test_configured_origin_rejects_userinfo_paths_queries_and_fragments(self):
        invalid_urls = [
            "https://user:pass@example.com:443",
            "https://example.com:443/path",
            "https://example.com:443?token=x",
            "https://example.com:443#fragment",
        ]
        for base_url in invalid_urls:
            with self.subTest(base_url=base_url), tempfile.TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "cutlery.local.json"
                config_path.write_text(
                    json.dumps({"remote_targets": {"bad": {"base_url": base_url}}}),
                    encoding="utf-8",
                )
                with self.assertRaises(ValueError):
                    configured_remote_targets(config_path)

    def test_loopback_target_can_define_lazy_worker_and_proxy_prefixes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "remote_targets": {
                            "trellis2": {
                                "base_url": "http://127.0.0.1:8890",
                                "worker_python": "C:/Comfy/trellis/Scripts/python.exe",
                                "worker_comfy_root": "C:/ComfyUI",
                                "worker_idle_seconds": 600,
                                "expose_node_prefixes": ["Trellis2"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            target = configured_remote_targets(config_path)["trellis2"]

        self.assertEqual(target.worker_idle_seconds, 600)
        self.assertEqual(target.expose_node_prefixes, ("Trellis2",))
        self.assertEqual(target.worker_comfy_root, "C:/ComfyUI")

    def test_worker_launch_is_rejected_for_non_loopback_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "remote_targets": {
                            "unsafe": {
                                "base_url": "http://192.0.2.10:8890",
                                "worker_python": "C:/Comfy/python.exe",
                                "worker_comfy_root": "C:/ComfyUI",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "only on loopback"):
                configured_remote_targets(config_path)


if __name__ == "__main__":
    unittest.main()
