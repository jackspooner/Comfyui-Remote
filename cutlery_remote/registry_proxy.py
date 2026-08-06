from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any


MAX_REGISTRY_PAYLOAD_BYTES = 512 * 1024
SHARED_EXTENSION_MODULE = "_cutlery_remote_registry_extensions"


class RegistryProxyRequestError(ValueError):
    """Raised before a browser registry request can make an outbound call."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class RegistryOperation:
    method: str
    path: str
    allowed_payload_fields: frozenset[str]


REGISTRY_OPERATIONS: dict[str, RegistryOperation] = {
    "remote_clip.choices": RegistryOperation(
        method="GET",
        path="/cutlery/remote/clip/choices",
        allowed_payload_fields=frozenset(),
    ),
}


def _install_shared_registry_operations() -> set[str]:
    extension = sys.modules.get(SHARED_EXTENSION_MODULE)
    contracts = getattr(extension, "registry_operations", None)
    if not isinstance(contracts, dict):
        return set()

    installed: set[str] = set()
    for registry_id, contract in contracts.items():
        if not isinstance(registry_id, str) or not isinstance(contract, (tuple, list)) or len(contract) != 3:
            continue
        method, path, fields = contract
        normalized_method = str(method or "").upper()
        normalized_path = str(path or "")
        if normalized_method not in {"GET", "POST"} or not normalized_path.startswith("/cutlery/"):
            continue
        if not isinstance(fields, (set, frozenset, tuple, list)) or not all(isinstance(field, str) for field in fields):
            continue
        REGISTRY_OPERATIONS[registry_id] = RegistryOperation(
            method=normalized_method,
            path=normalized_path,
            allowed_payload_fields=frozenset(fields),
        )
        installed.add(registry_id)
    return installed


_install_shared_registry_operations()


def prepare_registry_operation(
    registry: object,
    payload: object,
) -> tuple[str, RegistryOperation, dict[str, Any]]:
    registry_id = str(registry or "").strip()
    operation = REGISTRY_OPERATIONS.get(registry_id)
    if operation is None:
        allowed = ", ".join(sorted(REGISTRY_OPERATIONS))
        raise RegistryProxyRequestError(
            "unknown_registry",
            f"Unknown Cutlery remote registry {registry_id!r}. Allowed registry ids: {allowed}.",
        )

    if payload is None:
        normalized_payload: dict[str, Any] = {}
    elif isinstance(payload, dict):
        normalized_payload = dict(payload)
    else:
        raise RegistryProxyRequestError(
            "invalid_registry_payload",
            "payload must be a JSON object.",
        )

    unknown_fields = sorted(set(normalized_payload) - operation.allowed_payload_fields)
    if unknown_fields:
        raise RegistryProxyRequestError(
            "unsupported_registry_payload_fields",
            (
                f"Registry {registry_id!r} does not accept payload fields: "
                f"{', '.join(str(field) for field in unknown_fields)}."
            ),
        )

    try:
        encoded = json.dumps(
            normalized_payload,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RegistryProxyRequestError(
            "invalid_registry_payload",
            f"payload must contain JSON-compatible values: {exc}",
        ) from exc
    if len(encoded) > MAX_REGISTRY_PAYLOAD_BYTES:
        raise RegistryProxyRequestError(
            "registry_payload_too_large",
            (
                f"Registry payload is {len(encoded)} bytes; "
                f"the limit is {MAX_REGISTRY_PAYLOAD_BYTES} bytes."
            ),
        )

    return registry_id, operation, normalized_payload
