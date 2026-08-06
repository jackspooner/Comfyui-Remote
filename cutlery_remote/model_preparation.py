from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import threading
from typing import Any

from .inventory import normalize_model_type


MODEL_DIGEST_CACHE_VERSION = 1
MODEL_HASH_CHUNK_SIZE = 8 * 1024 * 1024


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required.")
    return text


def _sha256(value: object, *, field: str = "sha256") -> str:
    text = _required_text(value, field=field).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be a lowercase SHA-256 hexadecimal digest.")
    return text


def _canonical_model_name(value: object) -> str:
    text = _required_text(value, field="model_name").replace("\\", "/")
    path = PurePosixPath(text)
    if path.is_absolute() or not path.name or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"model_name must be a safe relative filename, got {value!r}.")
    return str(path)


def _category_from_mapping(value: Mapping[str, Any]) -> str:
    category = value.get("category")
    model_type = value.get("model_type")
    if category is None and model_type is None:
        raise ValueError("Model record requires category or model_type.")
    if category is not None and model_type is not None:
        normalized_category = normalize_model_type(category)
        normalized_model_type = normalize_model_type(model_type)
        if normalized_category != normalized_model_type:
            raise ValueError("Model record category and model_type disagree.")
        return normalized_category
    return normalize_model_type(category if category is not None else model_type)


def _name_from_mapping(value: Mapping[str, Any]) -> str:
    name = value.get("canonical_name")
    model_name = value.get("model_name")
    if name is None:
        name = value.get("name")
    if name is None and model_name is None:
        raise ValueError("Model record requires canonical_name, name, or model_name.")
    if name is not None and model_name is not None and _canonical_model_name(name) != _canonical_model_name(model_name):
        raise ValueError("Model record canonical_name and model_name disagree.")
    return _canonical_model_name(name if name is not None else model_name)


@dataclass(frozen=True)
class ModelIdentity:
    """Canonical model identity used for resolution, manifests, and transfers."""

    category: str
    canonical_name: str
    size: int
    sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", normalize_model_type(self.category))
        object.__setattr__(self, "canonical_name", _canonical_model_name(self.canonical_name))
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError("size must be a non-negative integer.")
        if self.sha256 is not None:
            object.__setattr__(self, "sha256", _sha256(self.sha256))

    @property
    def destination_key(self) -> tuple[str, str]:
        return self.category, self.canonical_name.casefold()

    @property
    def transfer_key(self) -> tuple[str, str, str]:
        if not self.sha256:
            raise ValueError(
                f"Model {self.category}/{self.canonical_name} needs SHA-256 before it can be transferred."
            )
        return self.category, self.canonical_name.casefold(), self.sha256

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "category": self.category,
            "model_type": self.category,
            "canonical_name": self.canonical_name,
            "model_name": self.canonical_name,
            "size": self.size,
        }
        if self.sha256:
            payload["sha256"] = self.sha256
        return payload


@dataclass(frozen=True)
class LocalModelFile:
    """A verified local file together with its canonical remote model identity."""

    path: Path
    identity: ModelIdentity

    def __post_init__(self) -> None:
        path = self.path.resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Model source must be a file: {path}")
        stat = path.stat()
        if stat.st_size != self.identity.size:
            raise ValueError(
                f"Model source size changed for {path}: expected {self.identity.size}, got {stat.st_size}."
            )
        object.__setattr__(self, "path", path)


@dataclass(frozen=True)
class RemoteModelResolution:
    """Validated batch-resolution result for one requested model."""

    identity: ModelIdentity
    present: bool


def model_identity_from_mapping(value: Mapping[str, Any]) -> ModelIdentity:
    if not isinstance(value, Mapping):
        raise ValueError("Model record must be an object.")
    size = value.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("Model record size must be a non-negative integer.")
    sha256 = value.get("sha256", value.get("hash"))
    return ModelIdentity(
        category=_category_from_mapping(value),
        canonical_name=_name_from_mapping(value),
        size=size,
        sha256=None if sha256 in (None, "") else str(sha256),
    )


def build_model_resolution_request(models: Iterable[ModelIdentity]) -> dict[str, Any]:
    """Build the stable request body used by a remote batch model resolver."""

    manifest = build_model_manifest(models)
    return {"models": manifest["models"], "manifest_id": manifest["identity"]}


def validate_model_resolution_response(
    request: Mapping[str, Any],
    response: Any,
) -> list[RemoteModelResolution]:
    """Validate a batch resolver response and bind every entry to its request identity."""

    requested_payload = request.get("models") if isinstance(request, Mapping) else None
    if not isinstance(requested_payload, list):
        raise ValueError("Model resolution request must contain a models array.")
    requested = [model_identity_from_mapping(item) for item in requested_payload]
    _validate_model_conflicts(requested)
    if not isinstance(response, Mapping) or response.get("ok") is not True:
        raise RuntimeError("Remote batch model resolution response was invalid.")
    records = response.get("models", response.get("records"))
    if not isinstance(records, list):
        raise RuntimeError("Remote batch model resolution response must contain a models array.")

    requested_by_destination = {identity.destination_key: identity for identity in requested}
    resolved: dict[tuple[str, str], RemoteModelResolution] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise RuntimeError("Remote batch model resolution returned a non-object record.")
        category = _category_from_mapping(record)
        name = _name_from_mapping(record)
        key = category, name.casefold()
        expected = requested_by_destination.get(key)
        if expected is None:
            raise RuntimeError(f"Remote batch model resolution returned an unrequested model {category}/{name}.")
        if key in resolved:
            raise RuntimeError(f"Remote batch model resolution returned {category}/{name} more than once.")
        present_value = record.get("present", record.get("available", record.get("found")))
        if not isinstance(present_value, bool):
            raise RuntimeError(f"Remote batch model resolution record {category}/{name} requires a boolean present field.")
        if present_value:
            actual = model_identity_from_mapping(record)
            if actual.size != expected.size:
                raise RuntimeError(
                    f"Remote model {category}/{name} conflicts by size: local {expected.size}, remote {actual.size}."
                )
            if expected.sha256 and actual.sha256 and expected.sha256 != actual.sha256:
                raise RuntimeError(f"Remote model {category}/{name} conflicts by SHA-256.")
            resolved[key] = RemoteModelResolution(actual, present=True)
        else:
            resolved[key] = RemoteModelResolution(expected, present=False)
    missing = [identity for identity in requested if identity.destination_key not in resolved]
    if missing:
        descriptions = ", ".join(f"{item.category}/{item.canonical_name}" for item in missing)
        raise RuntimeError(f"Remote batch model resolution omitted requested models: {descriptions}.")
    return [resolved[identity.destination_key] for identity in requested]


class LocalModelDigestCache:
    """Persistent SHA-256 cache whose entries are valid only for one file stat signature."""

    def __init__(self, cache_path: str | Path) -> None:
        self.path = Path(cache_path)
        self._lock = threading.Lock()

    @staticmethod
    def _normalized_path(path: Path) -> str:
        return os.path.normcase(os.path.normpath(str(path.resolve(strict=True))))

    @classmethod
    def _cache_key(cls, path: Path, size: int, mtime_ns: int) -> str:
        return json.dumps([cls._normalized_path(path), size, mtime_ns], separators=(",", ":"))

    def _read_entries(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("version") != MODEL_DIGEST_CACHE_VERSION:
            raise RuntimeError(f"Model digest cache {self.path} has an unsupported format.")
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            raise RuntimeError(f"Model digest cache {self.path} is missing entries.")
        return {
            str(key): value
            for key, value in entries.items()
            if isinstance(value, dict)
        }

    def _write_entries(self, entries: Mapping[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(self.path.parent),
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(
                    {"version": MODEL_DIGEST_CACHE_VERSION, "entries": entries},
                    handle,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def digest_for(self, path: str | Path, *, check_cancelled: Callable[[], None] | None = None) -> tuple[int, str]:
        source = Path(path).resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Model digest source must be a file: {source}")
        stat = source.stat()
        key = self._cache_key(source, stat.st_size, stat.st_mtime_ns)
        with self._lock:
            entries = self._read_entries()
            cached = entries.get(key)
            if cached is not None:
                digest = cached.get("sha256")
                if isinstance(digest, str):
                    return stat.st_size, _sha256(digest)

            digest = hashlib.sha256()
            with source.open("rb") as handle:
                while True:
                    if check_cancelled is not None:
                        check_cancelled()
                    chunk = handle.read(MODEL_HASH_CHUNK_SIZE)
                    if not chunk:
                        break
                    digest.update(chunk)
            if check_cancelled is not None:
                check_cancelled()
            current = source.stat()
            if current.st_size != stat.st_size or current.st_mtime_ns != stat.st_mtime_ns:
                raise RuntimeError(f"Model source changed while hashing: {source}")
            sha256 = digest.hexdigest()
            entries[key] = {
                "path": self._normalized_path(source),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256,
            }
            self._write_entries(entries)
            return stat.st_size, sha256


def local_model_file(
    path: str | Path,
    *,
    category: object,
    canonical_name: object,
    digest_cache: LocalModelDigestCache,
    check_cancelled: Callable[[], None] | None = None,
) -> LocalModelFile:
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"Model source must be a file: {source}")
    size, sha256 = digest_cache.digest_for(source, check_cancelled=check_cancelled)
    return LocalModelFile(source, ModelIdentity(str(category), str(canonical_name), size, sha256))


UniqueCategoryResolver = Callable[[Path], str | Sequence[str] | None]


def infer_unregistered_file_model_input(
    value: object,
    *,
    resolve_unique_category: UniqueCategoryResolver,
    digest_cache: LocalModelDigestCache,
    check_cancelled: Callable[[], None] | None = None,
) -> LocalModelFile:
    """Resolve one direct file input only when the caller can identify one model category."""

    if not isinstance(value, (str, Path)):
        raise ValueError("Unregistered model input is opaque; only a filesystem path can be inferred.")
    source = Path(value).resolve(strict=True)
    if source.is_dir():
        raise ValueError(f"Unregistered model input is a directory and cannot be inferred: {source}")
    if not source.is_file():
        raise ValueError(f"Unregistered model input is not a file: {source}")
    resolved = resolve_unique_category(source)
    if resolved is None:
        raise ValueError(f"No model category uniquely identifies unregistered file input {source}.")
    categories = (resolved,) if isinstance(resolved, str) else tuple(resolved)
    normalized = sorted({normalize_model_type(category) for category in categories})
    if len(normalized) != 1:
        raise ValueError(
            f"Unregistered file input {source} has ambiguous model categories: {', '.join(normalized) or 'none'}."
        )
    return local_model_file(
        source,
        category=normalized[0],
        canonical_name=source.name,
        digest_cache=digest_cache,
        check_cancelled=check_cancelled,
    )


def _validate_model_conflicts(models: Iterable[ModelIdentity]) -> list[ModelIdentity]:
    unique: dict[tuple[str, str], ModelIdentity] = {}
    for model in models:
        existing = unique.get(model.destination_key)
        if existing is not None and (existing.size != model.size or existing.sha256 != model.sha256):
            raise ValueError(
                f"Model destination {model.category}/{model.canonical_name} has conflicting local content."
            )
        unique.setdefault(model.destination_key, model)
    return sorted(unique.values(), key=lambda item: (item.category, item.canonical_name.casefold(), item.canonical_name))


def build_model_manifest(models: Iterable[ModelIdentity]) -> dict[str, Any]:
    """Produce an order-independent, content-sensitive model manifest identity."""

    canonical = _validate_model_conflicts(models)
    records = [model.to_payload() for model in canonical]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return {"identity": hashlib.sha256(encoded).hexdigest(), "models": records}


@dataclass
class _ActiveTransfer:
    identity: ModelIdentity
    key: tuple[str, str, str]
    done: threading.Event
    result: Any = None
    error: BaseException | None = None


class ModelTransferCoordinator:
    """Serialize transfers per target while sharing one transfer outcome with duplicate followers."""

    def __init__(self) -> None:
        self._lock = threading.Condition(threading.Lock())
        self._active_by_target: dict[str, _ActiveTransfer] = {}

    @staticmethod
    def _target_key(target: object) -> str:
        return _required_text(target, field="target").casefold()

    def transfer(
        self,
        target: object,
        model: LocalModelFile,
        *,
        transfer: Callable[[Path, str, str], Any],
        check_cancelled: Callable[[], None] | None = None,
    ) -> Any:
        """Run a staged-transfer callback once; followers receive its exact result or error."""

        identity = model.identity
        transfer_key = identity.transfer_key
        target_key = self._target_key(target)
        active: _ActiveTransfer
        leader = False
        with self._lock:
            while True:
                current = self._active_by_target.get(target_key)
                if current is not None and current.identity.destination_key == identity.destination_key and current.identity != identity:
                    raise RuntimeError(
                        f"Target {target!r} already has conflicting content for {identity.category}/{identity.canonical_name}."
                    )
                if current is not None and current.key == transfer_key:
                    active = current
                    break
                if current is None:
                    active = _ActiveTransfer(identity, transfer_key, threading.Event())
                    self._active_by_target[target_key] = active
                    leader = True
                    break
                if check_cancelled is not None:
                    check_cancelled()
                self._lock.wait(timeout=0.1)

        if not leader:
            while not active.done.wait(timeout=0.1):
                if check_cancelled is not None:
                    check_cancelled()
            if active.error is not None:
                raise active.error
            return active.result

        try:
            if check_cancelled is not None:
                check_cancelled()
            result = transfer(model.path, identity.category, identity.canonical_name)
            if check_cancelled is not None:
                check_cancelled()
            active.result = result
            return result
        except BaseException as error:
            active.error = error
            raise
        finally:
            active.done.set()
            with self._lock:
                if self._active_by_target.get(target_key) is active:
                    del self._active_by_target[target_key]
                self._lock.notify_all()


def prepare_models_for_target(
    target: object,
    models: Iterable[LocalModelFile],
    *,
    resolve_batch: Callable[[dict[str, Any]], Any],
    transfer_coordinator: ModelTransferCoordinator,
    transfer: Callable[[Path, str, str], Any],
    check_cancelled: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Resolve a complete manifest, reject conflicts, then stage-copy only missing models."""

    local_models = list(models)
    identities = _validate_model_conflicts(model.identity for model in local_models)
    by_destination = {model.identity.destination_key: model for model in local_models}
    request = build_model_resolution_request(identities)
    resolutions = validate_model_resolution_response(request, resolve_batch(request))
    results: list[dict[str, Any]] = []
    for resolution in resolutions:
        if check_cancelled is not None:
            check_cancelled()
        local = by_destination[resolution.identity.destination_key]
        if resolution.present:
            if local.identity.sha256 and not resolution.identity.sha256:
                raise RuntimeError(
                    f"Remote model {local.identity.category}/{local.identity.canonical_name} did not provide SHA-256; "
                    "its same-name content cannot be verified."
                )
            results.append({"identity": local.identity, "present": True, "transferred": False})
            continue
        result = transfer_coordinator.transfer(
            target,
            local,
            transfer=transfer,
            check_cancelled=check_cancelled,
        )
        results.append({"identity": local.identity, "present": False, "transferred": True, "result": result})
    return {"manifest": build_model_manifest(identities), "models": results}
