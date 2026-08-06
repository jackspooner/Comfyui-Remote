import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class RemoteModelPreparationTests(unittest.TestCase):
    def _local_model(self, directory: Path, name: str, contents: bytes):
        from cutlery_remote.model_preparation import LocalModelDigestCache, local_model_file

        source = directory / name
        source.write_bytes(contents)
        return local_model_file(
            source,
            category="checkpoints",
            canonical_name=name,
            digest_cache=LocalModelDigestCache(directory / "digests.json"),
        )

    def test_resolution_request_and_response_accept_category_or_model_type(self):
        from cutlery_remote.model_preparation import (
            ModelIdentity,
            build_model_resolution_request,
            validate_model_resolution_response,
        )

        model = ModelIdentity("checkpoint", "sdxl/base.safetensors", 3, "a" * 64)
        request = build_model_resolution_request([model])
        response = {
            "ok": True,
            "models": [
                {
                    "model_type": "checkpoints",
                    "name": "sdxl/base.safetensors",
                    "size": 3,
                    "hash": "a" * 64,
                    "present": True,
                }
            ],
        }

        resolved = validate_model_resolution_response(request, response)

        self.assertTrue(resolved[0].present)
        self.assertEqual(resolved[0].identity.category, "checkpoints")
        self.assertEqual(request["models"][0]["canonical_name"], "sdxl/base.safetensors")

    def test_resolution_response_rejects_unverified_same_name_content(self):
        from cutlery_remote.model_preparation import (
            ModelIdentity,
            build_model_resolution_request,
            validate_model_resolution_response,
        )

        request = build_model_resolution_request([ModelIdentity("vae", "vae.safetensors", 5, "a" * 64)])
        with self.assertRaisesRegex(RuntimeError, "conflicts by SHA-256"):
            validate_model_resolution_response(
                request,
                {
                    "ok": True,
                    "models": [
                        {
                            "category": "vae",
                            "canonical_name": "vae.safetensors",
                            "size": 5,
                            "sha256": "b" * 64,
                            "present": True,
                        }
                    ],
                },
            )

    def test_digest_cache_is_stat_keyed_and_persisted_atomically(self):
        from cutlery_remote.model_preparation import LocalModelDigestCache

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "model.safetensors"
            source.write_bytes(b"first")
            cache = LocalModelDigestCache(root / "cache" / "digests.json")
            size, first = cache.digest_for(source)
            self.assertEqual(size, 5)
            self.assertTrue(cache.path.exists())
            self.assertEqual(cache.digest_for(source), (5, first))

            time.sleep(0.002)
            source.write_bytes(b"other")
            self.assertNotEqual(cache.digest_for(source)[1], first)
            payload = json.loads(cache.path.read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 1)
            self.assertEqual(len(payload["entries"]), 2)

    def test_digest_hashing_checks_cancellation(self):
        from cutlery_remote.model_preparation import LocalModelDigestCache

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "model.safetensors"
            source.write_bytes(b"x" * 32)
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                LocalModelDigestCache(Path(temp_dir) / "digests.json").digest_for(
                    source,
                    check_cancelled=lambda: (_ for _ in ()).throw(RuntimeError("cancelled")),
                )

    def test_manifest_identity_is_order_independent_and_rejects_content_conflicts(self):
        from cutlery_remote.model_preparation import ModelIdentity, build_model_manifest

        first = ModelIdentity("vae", "a.safetensors", 1, "a" * 64)
        second = ModelIdentity("checkpoints", "b.safetensors", 2, "b" * 64)
        self.assertEqual(build_model_manifest([first, second])["identity"], build_model_manifest([second, first])["identity"])
        with self.assertRaisesRegex(ValueError, "conflicting local content"):
            build_model_manifest([first, ModelIdentity("vae", "a.safetensors", 1, "b" * 64)])

    def test_inference_requires_a_single_category_and_regular_file(self):
        from cutlery_remote.model_preparation import LocalModelDigestCache, infer_unregistered_file_model_input

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "model.safetensors"
            source.write_bytes(b"model")
            cache = LocalModelDigestCache(root / "digests.json")
            inferred = infer_unregistered_file_model_input(
                source,
                resolve_unique_category=lambda _path: "vae",
                digest_cache=cache,
            )
            self.assertEqual(inferred.identity.category, "vae")
            self.assertEqual(inferred.identity.canonical_name, "model.safetensors")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                infer_unregistered_file_model_input(
                    source,
                    resolve_unique_category=lambda _path: ["vae", "checkpoints"],
                    digest_cache=cache,
                )
            with self.assertRaisesRegex(ValueError, "directory"):
                infer_unregistered_file_model_input(
                    root,
                    resolve_unique_category=lambda _path: "vae",
                    digest_cache=cache,
                )

    def test_transfer_coordinator_serializes_target_and_shares_follower_result(self):
        from cutlery_remote.model_preparation import ModelTransferCoordinator

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = self._local_model(root, "model.safetensors", b"model")
            coordinator = ModelTransferCoordinator()
            started = threading.Event()
            release = threading.Event()
            calls = []
            results = []

            def transfer(path, category, name):
                calls.append((path, category, name))
                started.set()
                release.wait(timeout=2)
                return {"ok": True, "name": name}

            def run():
                results.append(coordinator.transfer("render-a", model, transfer=transfer))

            leader = threading.Thread(target=run)
            follower = threading.Thread(target=run)
            leader.start()
            self.assertTrue(started.wait(timeout=1))
            follower.start()
            release.set()
            leader.join(timeout=2)
            follower.join(timeout=2)

            self.assertEqual(len(calls), 1)
            self.assertEqual(results, [{"ok": True, "name": "model.safetensors"}] * 2)

    def test_transfer_coordinator_propagates_failure_and_releases_target(self):
        from cutlery_remote.model_preparation import ModelTransferCoordinator

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = self._local_model(root, "model.safetensors", b"model")
            coordinator = ModelTransferCoordinator()
            errors = []

            def failing_transfer(_path, _category, _name):
                raise RuntimeError("staging failed")

            def run():
                try:
                    coordinator.transfer("render-a", model, transfer=failing_transfer)
                except RuntimeError as error:
                    errors.append(str(error))

            first = threading.Thread(target=run)
            second = threading.Thread(target=run)
            first.start()
            second.start()
            first.join(timeout=2)
            second.join(timeout=2)
            self.assertEqual(errors, ["staging failed", "staging failed"])

            self.assertEqual(
                coordinator.transfer("render-a", model, transfer=lambda *_args: {"ok": True}),
                {"ok": True},
            )

    def test_prepare_transfers_only_missing_models_with_injected_staged_callback(self):
        from cutlery_remote.model_preparation import ModelTransferCoordinator, prepare_models_for_target

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            present = self._local_model(root, "present.safetensors", b"yes")
            missing = self._local_model(root, "missing.safetensors", b"no")
            copied = []

            def resolve(request):
                records = []
                for record in request["models"]:
                    records.append(
                        {
                            **record,
                            "present": record["canonical_name"] == "present.safetensors",
                        }
                    )
                return {"ok": True, "models": records}

            result = prepare_models_for_target(
                "render-a",
                [present, missing],
                resolve_batch=resolve,
                transfer_coordinator=ModelTransferCoordinator(),
                transfer=lambda path, category, name: copied.append((path, category, name)) or {"ok": True},
            )

            self.assertEqual([entry[2] for entry in copied], ["missing.safetensors"])
            self.assertEqual([entry["transferred"] for entry in result["models"]], [True, False])

    def test_sequential_missing_preparations_transfer_again_after_fresh_resolution(self):
        from cutlery_remote.model_preparation import ModelTransferCoordinator, prepare_models_for_target

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = self._local_model(root, "missing.safetensors", b"model")
            copied = []

            def resolve(request):
                return {
                    "ok": True,
                    "models": [{**record, "present": False} for record in request["models"]],
                }

            coordinator = ModelTransferCoordinator()
            for _ in range(2):
                prepare_models_for_target(
                    "render-a",
                    [model],
                    resolve_batch=resolve,
                    transfer_coordinator=coordinator,
                    transfer=lambda path, category, name: copied.append((path, category, name)) or {"ok": True},
                )

            self.assertEqual([entry[2] for entry in copied], ["missing.safetensors", "missing.safetensors"])


if __name__ == "__main__":
    unittest.main()
