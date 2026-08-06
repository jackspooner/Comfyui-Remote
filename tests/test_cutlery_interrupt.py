from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cutlery_interrupt


class _Response:
    def __init__(self, chunks: list[bytes], *, content_length: str | None = None) -> None:
        self.chunks = list(chunks)
        self.headers = {} if content_length is None else {"Content-Length": content_length}
        self.read_sizes: list[int] = []

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self.chunks.pop(0) if self.chunks else b""


class CutleryInterruptTests(unittest.TestCase):
    def test_response_limit_rejects_declared_content_length_before_reading(self):
        response = _Response([b"never read"], content_length="5")

        with self.assertRaisesRegex(RuntimeError, r"declares 5 bytes.*4-byte limit"):
            cutlery_interrupt.read_response_bytes(response, max_response_bytes=4)

        self.assertEqual(response.read_sizes, [])

    def test_response_limit_rejects_chunked_body_while_streaming(self):
        response = _Response([b"abc", b"def", b""])

        with self.assertRaisesRegex(RuntimeError, r"exceeds the 4-byte limit"):
            cutlery_interrupt.read_response_bytes(response, max_response_bytes=4)

        self.assertEqual(response.read_sizes, [cutlery_interrupt.HTTP_RESPONSE_READ_CHUNK_SIZE] * 2)

    def test_response_limit_accepts_chunked_body_at_limit(self):
        response = _Response([b"ab", b"cd", b""])

        self.assertEqual(cutlery_interrupt.read_response_bytes(response, max_response_bytes=4), b"abcd")


if __name__ == "__main__":
    unittest.main()
