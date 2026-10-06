"""WF3 boundary ports and the serializers/features they require."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REMOTE_LORA_CHAIN_FEATURE = "remote_lora_chain_boundary_v1"


@dataclass(frozen=True)
class BoundaryType:
    socket_type: str
    serializer: str
    allow_remote_input: bool = True
    allow_remote_output: bool = True
    required_feature: str | None = None


BOUNDARY_TYPES = {
    "image": BoundaryType("IMAGE", "tensor"),
    "mask": BoundaryType("MASK", "tensor", allow_remote_output=False),
    "latent": BoundaryType("LATENT", "latent", allow_remote_output=False),
    "conditioning": BoundaryType("CONDITIONING", "conditioning", allow_remote_output=False),
    "string": BoundaryType("STRING", "primitive"),
    "float": BoundaryType("FLOAT", "primitive"),
    "int": BoundaryType("INT", "primitive"),
    "bool": BoundaryType("BOOLEAN", "primitive"),
    "json": BoundaryType("*", "primitive"),
    "audio": BoundaryType("AUDIO", "tensor"),
    "video": BoundaryType("VIDEO", "primitive", allow_remote_input=False),
    "cutlery_lora_chain": BoundaryType("CUTLERY_LORA_CHAIN", "cutlery_lora_chain", allow_remote_output=False, required_feature=REMOTE_LORA_CHAIN_FEATURE),
}
BOUNDARY_PORT_TYPE_ALIASES = {"*": "json", "object": "json", "integer": "int", "number": "float", "boolean": "bool"}
SUPPORTED_BOUNDARY_PORT_TYPES = frozenset(BOUNDARY_TYPES)
BOUNDARY_SOCKET_TYPES = {port_type: definition.socket_type for port_type, definition in BOUNDARY_TYPES.items()}
BOUNDARY_SERIALIZER_REQUIREMENTS = {port_type: definition.serializer for port_type, definition in BOUNDARY_TYPES.items()}
BOUNDARY_FEATURE_REQUIREMENTS = {port_type: definition.required_feature for port_type, definition in BOUNDARY_TYPES.items() if definition.required_feature}
REMOTE_INBOUND_DISALLOWED_TYPES = frozenset(port_type for port_type, definition in BOUNDARY_TYPES.items() if not definition.allow_remote_input)
REMOTE_OUTBOUND_DISALLOWED_TYPES = frozenset(port_type for port_type, definition in BOUNDARY_TYPES.items() if not definition.allow_remote_output)


def normalize_boundary_port_type(value: Any) -> str:
    """Normalize public aliases to the canonical Cutlery boundary type name."""

    port_type = str(value or "").strip().lower()
    return BOUNDARY_PORT_TYPE_ALIASES.get(port_type, port_type)


def required_boundary_serializers(*port_groups: list[dict[str, str]]) -> set[str]:
    """Return serializers required by all declared boundary ports."""

    return {
        definition.serializer
        for ports in port_groups
        for port in ports
        if (definition := BOUNDARY_TYPES.get(normalize_boundary_port_type(port.get("type"))))
    }


def required_boundary_features(*port_groups: list[dict[str, str]]) -> set[str]:
    """Return feature flags required by all declared boundary ports."""

    return {
        definition.required_feature
        for ports in port_groups
        for port in ports
        if (definition := BOUNDARY_TYPES.get(normalize_boundary_port_type(port.get("type")))) and definition.required_feature
    }
