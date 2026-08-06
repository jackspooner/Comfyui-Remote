from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class BlobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for_hash(self, blob_hash: str) -> Path:
        safe_hash = str(blob_hash or "").strip().lower()
        if len(safe_hash) != 64 or any(char not in "0123456789abcdef" for char in safe_hash):
            raise ValueError("Blob hash must be a SHA-256 hex digest.")
        return self.root / safe_hash[:2] / safe_hash

    def put_bytes(self, data: bytes) -> dict[str, object]:
        payload = bytes(data)
        blob_hash = sha256_bytes(payload)
        path = self._path_for_hash(blob_hash)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(prefix=".blob-", suffix=".tmp", dir=str(path.parent))
            temp_path = Path(temp_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, path)
            finally:
                temp_path.unlink(missing_ok=True)
        return {"hash": blob_hash, "size": len(payload)}

    def has_blob(self, blob_hash: str) -> bool:
        return self._path_for_hash(blob_hash).exists()

    def get_bytes(self, blob_hash: str) -> bytes:
        path = self._path_for_hash(blob_hash)
        if not path.exists():
            raise FileNotFoundError(f"Remote blob {blob_hash} is missing.")
        return path.read_bytes()
