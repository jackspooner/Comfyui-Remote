from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT_FILES = {
    ".comfyignore",
    ".env.example",
    ".gitignore",
    "LICENSE",
    "README.md",
    "SECURITY.md",
    "__init__.py",
    "cutlery_clip_gguf.py",
    "cutlery_config.py",
    "cutlery_features.py",
    "cutlery_gpu_memory.py",
    "cutlery_interrupt.py",
    "cutlery_lora_chain.py",
    "cutlery_vram.py",
    "nodes_remote.py",
    "nodes_remote_clip.py",
    "nodes_remote_proxy.py",
    "nodes_wf3_boundary.py",
    "pyproject.toml",
    "requirements-dev.txt",
}
ALLOWED_DIRECTORIES = {"cutlery_remote", "docs", "web"}


def registry_paths() -> set[str]:
    ignored_prefixes = {".github/", "tests/", ".git/", "build/", "dist/", "wheelhouse/", "__pycache__/"}
    ignored_names = {"PROJECT_TRACKER.md"}
    paths = set()
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in ignored_names or relative.endswith(".egg-info/PKG-INFO") or ".egg-info/" in relative or any(relative.startswith(prefix) for prefix in ignored_prefixes):
            continue
        if path.suffix in {".pyc", ".pyo"} or "__pycache__/" in relative:
            continue
        paths.add(relative)
    return paths


class ReleaseArchiveTests(unittest.TestCase):
    def test_archive_has_an_exact_public_allowlist(self):
        archive = registry_paths()
        unexpected = []
        for relative in archive:
            parts = relative.split("/")
            if len(parts) == 1:
                if relative not in ALLOWED_ROOT_FILES:
                    unexpected.append(relative)
            elif parts[0] not in ALLOWED_DIRECTORIES:
                unexpected.append(relative)
            elif parts[0] == "cutlery_remote" and not relative.endswith(".py"):
                unexpected.append(relative)
            elif parts[0] == "docs" and Path(relative).suffix not in {".md", ".yaml"}:
                unexpected.append(relative)
            elif parts[0] == "web" and Path(relative).suffix != ".js":
                unexpected.append(relative)
        self.assertEqual(sorted(unexpected), [])

    def test_archive_excludes_machine_state_and_credentials(self):
        archive = registry_paths()
        forbidden_names = {".env", "cutlery.local.json", "router.local.json", "cookies.txt", "cookies.sqlite", "id_rsa", "id_ed25519"}
        self.assertEqual(sorted(Path(path).name for path in archive if Path(path).name in forbidden_names), [])
        self.assertEqual(sorted(path for path in archive if Path(path).suffix in {".pem", ".key", ".p12", ".pfx"}), [])

    def test_public_package_has_the_contract_dependency_and_openapi_documents(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(
            '"cutlery-workflow-contracts @ git+https://github.com/jackspooner/cutlery-workflow-contracts.git@7b3db7218c8a781255a1dca695110f3df1e6c59f"',
            pyproject,
        )
        for name in ("cutlery_remote_openapi.yaml", "cutlery_remote_clip_openapi.yaml"):
            self.assertTrue((ROOT / "docs" / name).is_file())


if __name__ == "__main__":
    unittest.main()
