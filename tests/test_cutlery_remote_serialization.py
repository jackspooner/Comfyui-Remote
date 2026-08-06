import sys
import tempfile
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutlery_remote.blobs import BlobStore, sha256_bytes
from cutlery_remote.serialization import decode_value, decode_value_bundle, encode_value, encode_value_bundle


class RemoteSerializationTests(unittest.TestCase):
    def test_blob_store_writes_content_addressed_bytes_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = BlobStore(Path(temp_dir))
            payload = b"hello remote comfy"

            descriptor = store.put_bytes(payload)
            second = store.put_bytes(payload)

            self.assertEqual(descriptor, second)
            self.assertEqual(descriptor["hash"], sha256_bytes(payload))
            self.assertTrue(store.has_blob(descriptor["hash"]))
            self.assertEqual(store.get_bytes(descriptor["hash"]), payload)

    def test_round_trips_primitive_values_inline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = BlobStore(Path(temp_dir))

            for value in ["text", 42, 3.5, True, None]:
                with self.subTest(value=value):
                    manifest = encode_value(value, store)
                    self.assertEqual(decode_value(manifest, store), value)

    def test_round_trips_tensor_values_through_blob_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = BlobStore(Path(temp_dir))
            tensor = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

            manifest = encode_value(tensor, store)
            restored = decode_value(manifest, store)

            self.assertEqual(manifest["kind"], "tensor")
            self.assertEqual(tuple(restored.shape), (2, 3, 4))
            self.assertTrue(torch.equal(restored, tensor))

    def test_round_trips_bfloat16_tensor_values_through_bundle(self):
        tensor = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4).to(torch.bfloat16)

        restored = decode_value_bundle(encode_value_bundle(tensor))

        self.assertEqual(restored.dtype, torch.bfloat16)
        self.assertEqual(tuple(restored.shape), (2, 3, 4))
        self.assertTrue(torch.equal(restored, tensor))

    def test_round_trips_bytes_through_blob_store(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = BlobStore(Path(temp_dir))
            payload = b"remote media bytes"

            manifest = encode_value(payload, store)
            restored = decode_value(manifest, store)

        self.assertEqual(manifest["kind"], "bytes")
        self.assertEqual(restored, payload)

    def test_round_trips_latent_and_conditioning_shaped_structures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = BlobStore(Path(temp_dir))
            value = {
                "latent": {"samples": torch.ones((1, 4, 8, 8), dtype=torch.float16), "batch_index": 0},
                "conditioning": [
                    [
                        torch.zeros((1, 77, 4), dtype=torch.float32),
                        {"pooled_output": torch.ones((1, 4), dtype=torch.float32), "width": 1024},
                    ]
                ],
            }

            manifest = encode_value(value, store)
            restored = decode_value(manifest, store)

            self.assertEqual(restored["latent"]["batch_index"], 0)
            self.assertTrue(torch.equal(restored["latent"]["samples"], value["latent"]["samples"]))
            self.assertTrue(torch.equal(restored["conditioning"][0][0], value["conditioning"][0][0]))
            self.assertTrue(torch.equal(restored["conditioning"][0][1]["pooled_output"], value["conditioning"][0][1]["pooled_output"]))

    def test_round_trips_cutlery_lora_chain_recipe(self):
        value = {
            "loras": [
                {
                    "lora_name": "styles/character.safetensors",
                    "strength_model": 0.75,
                    "strength_clip": 0.4,
                },
                {
                    "lora_name": "lighting.safetensors",
                    "strength_model": -0.25,
                    "strength_clip": 1.0,
                },
            ]
        }

        restored = decode_value_bundle(encode_value_bundle(value))

        self.assertEqual(restored, value)


if __name__ == "__main__":
    unittest.main()
