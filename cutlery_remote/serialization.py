"""Content-addressed tensor-tree manifests with no tensor-runtime dependency.

Tensor frameworks convert their objects to and from :class:`TensorPayload` at
their integration boundary.  This module owns only the portable bytes and JSON
shapes sent between peers.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

try:  # PyTorch is supplied by ComfyUI; primitive-only consumers need not install it.
    import torch
except ImportError:  # pragma: no cover - exercised by primitive-only installations.
    torch = None

VALUE_SCHEMA = "cutlery.remote.value.v1"
VALUE_BUNDLE_SCHEMA = "cutlery.remote.value_bundle.v1"
_PRIMITIVE_TYPES = (str, int, float, bool)


class BlobStore(Protocol):
    def put_bytes(self, data: bytes) -> dict[str, object]: ...
    def get_bytes(self, blob_hash: str) -> bytes: ...


@dataclass(frozen=True)
class TensorPayload:
    """Raw contiguous tensor bytes plus portable dtype and shape metadata."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class MemoryBlobStore:
    """In-memory content-addressed store for bundles and compatibility tests."""

    def __init__(self, blobs: dict[str, bytes] | None = None) -> None:
        self.blobs = {str(key).lower(): bytes(value) for key, value in (blobs or {}).items()}

    def put_bytes(self, data: bytes) -> dict[str, object]:
        payload = bytes(data)
        blob_hash = sha256_bytes(payload)
        self.blobs[blob_hash] = payload
        return {"hash": blob_hash, "size": len(payload)}

    def get_bytes(self, blob_hash: str) -> bytes:
        safe_hash = str(blob_hash or "").strip().lower()
        try:
            return self.blobs[safe_hash]
        except KeyError as exc:
            raise FileNotFoundError(f"Remote blob {safe_hash} is missing from the value bundle.") from exc


def _blob_descriptor(data: bytes, blob_store: BlobStore) -> dict[str, object]:
    descriptor = blob_store.put_bytes(data)
    blob_hash = str(descriptor.get("hash") or "").lower()
    size = descriptor.get("size")
    if blob_hash != sha256_bytes(data) or size != len(data):
        raise ValueError("Blob store returned an invalid content-addressed descriptor.")
    return {"hash": blob_hash, "size": size}


def _validate_max_blob_bytes(max_blob_bytes: int | None) -> None:
    if max_blob_bytes is not None and (not isinstance(max_blob_bytes, int) or isinstance(max_blob_bytes, bool) or max_blob_bytes < 0):
        raise ValueError("Maximum blob size must be a non-negative integer or None.")


def _base64_payload_size(encoded: str) -> int:
    if len(encoded) % 4:
        raise ValueError("Remote value bundle has invalid base64 data.")
    padding = len(encoded) - len(encoded.rstrip("="))
    if padding > 2 or (padding and "=" in encoded[:-padding]):
        raise ValueError("Remote value bundle has invalid base64 data.")
    return (len(encoded) // 4) * 3 - padding


def _read_blob(descriptor: Any, blob_store: BlobStore, *, max_blob_bytes: int | None) -> bytes:
    if not isinstance(descriptor, dict):
        raise ValueError("Value manifest is missing blob descriptor.")
    blob_hash = str(descriptor.get("hash") or "").strip().lower()
    size = descriptor.get("size")
    if len(blob_hash) != 64 or any(character not in "0123456789abcdef" for character in blob_hash):
        raise ValueError("Value manifest has an invalid blob hash.")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError("Value manifest has an invalid blob size.")
    if max_blob_bytes is not None and size > max_blob_bytes:
        raise ValueError(f"Value blob exceeds the configured limit of {max_blob_bytes} bytes.")
    payload = blob_store.get_bytes(blob_hash)
    if len(payload) != size or sha256_bytes(payload) != blob_hash:
        raise ValueError("Value blob does not match its content-addressed descriptor.")
    return payload


def _validated_shape(shape: Any) -> tuple[int, ...]:
    if not isinstance(shape, list) or any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0 for dimension in shape):
        raise ValueError("Tensor manifest has an invalid shape.")
    return tuple(shape)


def _tensor_to_bytes(tensor: Any) -> bytes:
    cpu_tensor = tensor.detach().cpu().contiguous()
    if str(cpu_tensor.dtype) == "torch.bfloat16":
        return cpu_tensor.view(torch.uint16).numpy().tobytes()
    return cpu_tensor.numpy().tobytes()


def _tensor_from_bytes(payload: bytes, dtype_name: str, shape: tuple[int, ...]) -> Any:
    if torch is None:
        raise RuntimeError("Decoding tensor manifests requires PyTorch.")
    dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
    if dtype is None:
        raise ValueError(f"Unsupported tensor dtype {dtype_name!r}.")
    if dtype == torch.bfloat16:
        return torch.frombuffer(bytearray(payload), dtype=torch.uint16).view(torch.bfloat16).reshape(shape).clone()
    return torch.frombuffer(bytearray(payload), dtype=dtype).reshape(shape).clone()


def encode_tensor_payload(value: TensorPayload, blob_store: BlobStore) -> dict[str, Any]:
    if not isinstance(value.dtype, str) or not value.dtype:
        raise ValueError("Tensor payload dtype must be a non-empty string.")
    return {"schema": VALUE_SCHEMA, "kind": "tensor", "dtype": value.dtype, "shape": list(value.shape), "blob": _blob_descriptor(value.data, blob_store)}


def decode_tensor_payload(manifest: dict[str, Any], blob_store: BlobStore, *, max_blob_bytes: int | None = None) -> TensorPayload:
    if manifest.get("schema") != VALUE_SCHEMA or manifest.get("kind") != "tensor":
        raise ValueError("Unsupported tensor manifest.")
    dtype = manifest.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("Tensor manifest is missing dtype.")
    return TensorPayload(dtype=dtype, shape=_validated_shape(manifest.get("shape")), data=_read_blob(manifest.get("blob"), blob_store, max_blob_bytes=max_blob_bytes))


def encode_value(value: Any, blob_store: BlobStore) -> dict[str, Any]:
    """Encode a supported value tree using the stable v1 manifest format."""

    if value is None or isinstance(value, _PRIMITIVE_TYPES):
        return {"schema": VALUE_SCHEMA, "kind": "primitive", "value": value}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"schema": VALUE_SCHEMA, "kind": "bytes", "blob": _blob_descriptor(bytes(value), blob_store)}
    if isinstance(value, TensorPayload):
        return encode_tensor_payload(value, blob_store)
    if torch is not None and isinstance(value, torch.Tensor):
        return encode_tensor_payload(TensorPayload(str(value.dtype), tuple(value.shape), _tensor_to_bytes(value)), blob_store)
    if isinstance(value, list):
        return {"schema": VALUE_SCHEMA, "kind": "list", "items": [encode_value(item, blob_store) for item in value]}
    if isinstance(value, tuple):
        return {"schema": VALUE_SCHEMA, "kind": "tuple", "items": [encode_value(item, blob_store) for item in value]}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Remote serialization only supports string keys in dictionaries.")
        return {"schema": VALUE_SCHEMA, "kind": "dict", "items": {key: encode_value(item, blob_store) for key, item in value.items()}}
    raise TypeError(f"Remote serialization does not support values of type {type(value).__name__}.")


def decode_value(manifest: dict[str, Any], blob_store: BlobStore, *, max_blob_bytes: int | None = None) -> Any:
    """Decode a v1 manifest; tensor leaves are returned as ``TensorPayload``."""

    if not isinstance(manifest, dict) or manifest.get("schema") != VALUE_SCHEMA:
        raise ValueError("Unsupported remote value manifest schema.")
    kind = manifest.get("kind")
    if kind == "primitive":
        value = manifest.get("value")
        if value is not None and not isinstance(value, _PRIMITIVE_TYPES):
            raise ValueError("Primitive manifest contains an unsupported value.")
        return value
    if kind == "tensor":
        tensor_payload = decode_tensor_payload(manifest, blob_store, max_blob_bytes=max_blob_bytes)
        if torch is None:
            return tensor_payload
        return _tensor_from_bytes(tensor_payload.data, tensor_payload.dtype, tensor_payload.shape)
    if kind == "bytes":
        return _read_blob(manifest.get("blob"), blob_store, max_blob_bytes=max_blob_bytes)
    items = manifest.get("items")
    if kind in {"list", "tuple"}:
        if not isinstance(items, list):
            raise ValueError(f"{kind.title()} manifest is missing items.")
        decoded = [decode_value(item, blob_store, max_blob_bytes=max_blob_bytes) for item in items]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "dict":
        if not isinstance(items, dict) or not all(isinstance(key, str) for key in items):
            raise ValueError("Dict manifest is missing items.")
        return {key: decode_value(item, blob_store, max_blob_bytes=max_blob_bytes) for key, item in items.items()}
    raise ValueError(f"Unsupported remote value kind {kind!r}.")


def encode_value_bundle(value: Any) -> dict[str, Any]:
    store = MemoryBlobStore()
    return {"schema": VALUE_BUNDLE_SCHEMA, "manifest": encode_value(value, store), "blobs": {blob_hash: base64.b64encode(payload).decode("ascii") for blob_hash, payload in sorted(store.blobs.items())}}


def decode_value_bundle(bundle: dict[str, Any], *, max_blob_bytes: int | None = None) -> Any:
    _validate_max_blob_bytes(max_blob_bytes)
    if not isinstance(bundle, dict) or bundle.get("schema") != VALUE_BUNDLE_SCHEMA:
        raise ValueError("Unsupported remote value bundle schema.")
    manifest, blobs = bundle.get("manifest"), bundle.get("blobs")
    if not isinstance(manifest, dict):
        raise ValueError("Remote value bundle is missing a manifest.")
    if not isinstance(blobs, dict):
        raise ValueError("Remote value bundle is missing blobs.")
    decoded_blobs: dict[str, bytes] = {}
    for blob_hash, encoded in blobs.items():
        if not isinstance(blob_hash, str) or not isinstance(encoded, str):
            raise ValueError("Remote value bundle has an invalid blob entry.")
        decoded_size = _base64_payload_size(encoded)
        if max_blob_bytes is not None and decoded_size > max_blob_bytes:
            raise ValueError(f"Value blob exceeds the configured limit of {max_blob_bytes} bytes.")
        try:
            decoded_blobs[blob_hash.strip().lower()] = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError("Remote value bundle has invalid base64 data.") from exc
    return decode_value(manifest, MemoryBlobStore(decoded_blobs), max_blob_bytes=max_blob_bytes)
