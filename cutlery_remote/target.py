from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
import urllib.parse
from typing import Any

try:
    from ..cutlery_config import data_path
except ImportError:  # pragma: no cover - supports direct module imports in tests.
    from cutlery_config import data_path


_PLAIN_TARGET_RE = re.compile(r"^(?P<host>[A-Za-z0-9_.-]+):(?P<port>[1-9][0-9]{0,4})$")
_CURLY_TARGET_RE = re.compile(r"^(?P<host>[A-Za-z0-9_.-]+):\{(?P<port>[1-9][0-9]{0,4})\}$")
_CUTLERY_TARGET_RE = re.compile(r"^cutlery://(?P<host>[A-Za-z0-9_.-]+):(?P<port>[1-9][0-9]{0,4})$")
_CUTLERY_ALIAS_RE = re.compile(r"^cutlery://(?P<alias>[A-Za-z0-9_.-]+)$")
_LEGACY_CUTLERY_ALIAS_RE = re.compile(r"^cutlery/(?P<alias>[A-Za-z0-9_.-]+)$")
_GROUP_LABEL_SEPARATOR = " // "
LEGACY_LOCAL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "cutlery.local.json"
LOCAL_CONFIG_PATH = data_path("config.json")
_TARGET_FIELDS = frozenset({
    "base_url",
    "display_label",
    "copy_host",
    "copy_root",
    "worker_python",
    "worker_comfy_root",
    "worker_idle_seconds",
    "expose_node_prefixes",
})


@dataclass(frozen=True)
class RemoteTarget:
    scheme: str
    host: str
    port: int
    canonical: str
    base_url: str
    display_label: str


@dataclass(frozen=True)
class TrustedRemoteTarget:
    """Resolved outbound target whose origin is safe to receive the shared token."""

    name: str
    base_url: str
    canonical: str
    display_label: str
    copy_host: str | None = None
    copy_root: str | None = None
    worker_python: str | None = None
    worker_comfy_root: str | None = None
    worker_idle_seconds: int = 600
    expose_node_prefixes: tuple[str, ...] = ()


def remote_target_endpoint(value: object) -> str:
    """Return the endpoint or alias portion of a labelled editor group title.

    Group labels are presentation-only. Callers must resolve this returned
    value before deciding whether a target is trusted or receives credentials.
    """

    text = str(value or "").strip()
    endpoint, separator, _label = text.partition(_GROUP_LABEL_SEPARATOR)
    return endpoint.strip() if separator else text


def remote_target_alias(value: object) -> str | None:
    """Return a target alias from its canonical or legacy spelling."""

    text = remote_target_endpoint(value)
    match = _CUTLERY_ALIAS_RE.fullmatch(text) or _LEGACY_CUTLERY_ALIAS_RE.fullmatch(text)
    return match.group("alias") if match else None


def parse_remote_target(title: object) -> RemoteTarget | None:
    text = remote_target_endpoint(title)
    match = _PLAIN_TARGET_RE.match(text) or _CURLY_TARGET_RE.match(text) or _CUTLERY_TARGET_RE.match(text)
    if match is None:
        return None

    host = match.group("host")
    port = int(match.group("port"))
    if port > 65535:
        return None

    return RemoteTarget(
        scheme="http",
        host=host,
        port=port,
        canonical=f"cutlery://{host}:{port}",
        base_url=f"http://{host}:{port}",
        display_label=f"{host}:{port}",
    )


def normalize_remote_base_url(value: object) -> str:
    text = str(value or "").strip().rstrip("/")
    parsed = urllib.parse.urlsplit(text)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Remote target base_url must use http or https.")
    if not parsed.hostname or parsed.port is None:
        raise ValueError("Remote target base_url must include an explicit host and port.")
    if parsed.username or parsed.password:
        raise ValueError("Remote target base_url must not contain user information.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("Remote target base_url must not contain a path, query, or fragment.")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{parsed.scheme}://{host}:{parsed.port}"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _normalized_target_records(
    raw_targets: Any,
    *,
    location: Path,
    reject_unknown_fields: bool = False,
) -> dict[str, dict[str, Any]]:
    if raw_targets is None:
        return {}
    if not isinstance(raw_targets, dict):
        raise ValueError(f"{location.name} remote_targets must be an object keyed by target alias.")
    result: dict[str, dict[str, Any]] = {}
    for raw_name, raw_record in raw_targets.items():
        name = str(raw_name or "").strip()
        if not name or _CUTLERY_ALIAS_RE.fullmatch(f"cutlery://{name}") is None:
            raise ValueError(f"Invalid Cutlery remote target alias {raw_name!r}.")
        if not isinstance(raw_record, dict):
            raise ValueError(f"Cutlery remote target {name!r} must be a JSON object.")
        unknown_fields = sorted(set(raw_record) - _TARGET_FIELDS)
        if reject_unknown_fields and unknown_fields:
            raise ValueError(
                f"Cutlery remote target {name!r} contains unsupported fields: {', '.join(unknown_fields)}."
            )
        record = {field: raw_record[field] for field in _TARGET_FIELDS if field in raw_record}
        record["base_url"] = normalize_remote_base_url(record.get("base_url"))
        if "display_label" in record:
            record["display_label"] = str(record.get("display_label") or "").strip()
        if "copy_host" in record:
            record["copy_host"] = str(record.get("copy_host") or "").strip()
        if "copy_root" in record:
            record["copy_root"] = str(record.get("copy_root") or "").strip().replace("\\", "/").rstrip("/")
        if "worker_python" in record:
            record["worker_python"] = str(record.get("worker_python") or "").strip()
        if "worker_comfy_root" in record:
            record["worker_comfy_root"] = str(record.get("worker_comfy_root") or "").strip()
        if "worker_idle_seconds" in record:
            idle_seconds = record["worker_idle_seconds"]
            if isinstance(idle_seconds, bool) or not isinstance(idle_seconds, int) or idle_seconds < 1:
                raise ValueError(f"Cutlery remote target {name!r} worker_idle_seconds must be a positive integer.")
        if "expose_node_prefixes" in record:
            prefixes = record["expose_node_prefixes"]
            if not isinstance(prefixes, list) or not prefixes or any(
                not isinstance(prefix, str) or not prefix.strip() for prefix in prefixes
            ):
                raise ValueError(
                    f"Cutlery remote target {name!r} expose_node_prefixes must be a non-empty string array."
                )
            record["expose_node_prefixes"] = [prefix.strip() for prefix in prefixes]
        if bool(record.get("worker_python")) != bool(record.get("worker_comfy_root")):
            raise ValueError(
                f"Cutlery remote target {name!r} must configure worker_python and worker_comfy_root together."
            )
        if record.get("worker_python"):
            parsed = urllib.parse.urlsplit(record["base_url"])
            if parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
                raise ValueError(f"Cutlery remote target {name!r} may launch a worker only on loopback.")
        result[name] = record
    return result


def _migrate_legacy_config() -> None:
    if LOCAL_CONFIG_PATH.exists() or not LEGACY_LOCAL_CONFIG_PATH.is_file():
        return
    payload = json.loads(LEGACY_LOCAL_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{LEGACY_LOCAL_CONFIG_PATH.name} must contain a JSON object.")
    remote_targets = _normalized_target_records(
        payload.get("remote_targets", {}),
        location=LEGACY_LOCAL_CONFIG_PATH,
    )
    _write_json_atomic(LOCAL_CONFIG_PATH, {"remote_targets": remote_targets})


def _load_target_config(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    selected_path = Path(path) if path is not None else LOCAL_CONFIG_PATH
    if path is None:
        _migrate_legacy_config()
    if not selected_path.is_file():
        return {}
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{selected_path.name} must contain a JSON object.")
    if path is None:
        unknown_fields = sorted(set(payload) - {"remote_targets"})
        if unknown_fields:
            raise ValueError(
                "CUTLERY_DATA_DIR/config.json contains unsupported top-level fields: "
                + ", ".join(unknown_fields)
                + "."
            )
    return _normalized_target_records(
        payload.get("remote_targets", {}),
        location=selected_path,
        reject_unknown_fields=path is None,
    )


def configured_remote_targets(path: str | Path | None = None) -> dict[str, TrustedRemoteTarget]:
    targets: dict[str, TrustedRemoteTarget] = {}
    for name, record in _load_target_config(path).items():
        base_url = normalize_remote_base_url(record.get("base_url"))
        copy_host = str(record.get("copy_host") or "").strip() or None
        copy_root = str(record.get("copy_root") or "").strip().replace("\\", "/").rstrip("/") or None
        targets[name] = TrustedRemoteTarget(
            name=name,
            base_url=base_url,
            canonical=f"cutlery://{name}",
            display_label=str(record.get("display_label") or name).strip() or name,
            copy_host=copy_host,
            copy_root=copy_root,
            worker_python=str(record.get("worker_python") or "").strip() or None,
            worker_comfy_root=str(record.get("worker_comfy_root") or "").strip() or None,
            worker_idle_seconds=int(record.get("worker_idle_seconds", 600)),
            expose_node_prefixes=tuple(record.get("expose_node_prefixes") or ()),
        )
    return targets


def resolve_trusted_remote_target(
    value: object,
    *,
    config_path: str | Path | None = None,
) -> TrustedRemoteTarget:
    """Resolve an alias or exact configured origin before attaching credentials.

    Every origin, including a loopback tunnel, must be registered in
    the canonical ``CUTLERY_DATA_DIR/config.json`` file. Otherwise an untrusted
    local listener could capture and reuse the shared remote token.
    """

    supplied_text = str(value or "").strip()
    text = remote_target_endpoint(supplied_text)
    if not text:
        raise ValueError("Cutlery remote target is required.")
    targets = configured_remote_targets(config_path)

    alias = remote_target_alias(text) or text
    if alias in targets:
        return targets[alias]

    parsed_title = parse_remote_target(text)
    candidate_base_url = parsed_title.base_url if parsed_title is not None else None
    if candidate_base_url is None:
        try:
            candidate_base_url = normalize_remote_base_url(text)
        except ValueError:
            candidate_base_url = None

    if candidate_base_url is not None:
        normalized_candidate = normalize_remote_base_url(candidate_base_url)
        for target in targets.values():
            if normalized_candidate == target.base_url:
                return target

    configured = ", ".join(sorted(targets)) or "none"
    config_label = Path(config_path).name if config_path is not None else "CUTLERY_DATA_DIR/config.json"
    raise ValueError(
        f"Cutlery remote target {supplied_text!r} is not trusted. Register it under remote_targets in "
        f"{config_label}; configured aliases: {configured}."
    )
