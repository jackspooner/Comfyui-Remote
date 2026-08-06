from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from .cutlery_config import data_path
    from .cutlery_remote.target import configured_remote_targets
except ImportError:  # pragma: no cover - supports direct module imports in tests.
    from cutlery_config import data_path
    from cutlery_remote.target import configured_remote_targets


_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


def _proxy_enabled() -> bool:
    return os.environ.get("CUTLERY_REMOTE_PROXY_NODES_ENABLED", "1").strip().casefold() in _ENABLED_VALUES


def _catalog_path(target_name: str) -> Path:
    return data_path("remote-node-catalogs", f"{target_name}.json")


def _proxy_node_class(target_name: str, class_type: str, definition: dict[str, Any]):
    input_types = definition.get("input")
    if not isinstance(input_types, dict):
        raise ValueError(f"Remote node catalog {target_name!r} node {class_type!r} has no input schema.")
    return_types = tuple(definition.get("output") or ())
    return_names = tuple(definition.get("output_name") or return_types)
    category = str(definition.get("category") or "Remote")

    @classmethod
    def INPUT_TYPES(cls):
        return input_types

    def _remote_proxy_error(self, **_kwargs):
        raise RuntimeError(
            f"{class_type} is a proxy for cutlery://{target_name} and must be placed inside that remote group."
        )

    attributes = {
        "INPUT_TYPES": INPUT_TYPES,
        "RETURN_TYPES": return_types,
        "RETURN_NAMES": return_names,
        "FUNCTION": "_remote_proxy_error",
        "CATEGORY": f"Remote/{target_name}/{category}",
        "_remote_proxy_error": _remote_proxy_error,
        "DESCRIPTION": (
            f"Executes {class_type} on cutlery://{target_name}. "
            f"Keep this node and its runtime-object connections inside that remote group."
        ),
    }
    if "output_is_list" in definition:
        attributes["OUTPUT_IS_LIST"] = tuple(bool(value) for value in definition["output_is_list"])
    if definition.get("input_is_list"):
        attributes["INPUT_IS_LIST"] = True
    if definition.get("output_node"):
        attributes["OUTPUT_NODE"] = True
    return type(f"CutleryRemoteProxy_{target_name}_{class_type}", (), attributes)


def load_remote_proxy_nodes() -> tuple[dict[str, Any], dict[str, str]]:
    node_classes: dict[str, Any] = {}
    display_names: dict[str, str] = {}
    if not _proxy_enabled():
        return node_classes, display_names
    for target_name, target in configured_remote_targets().items():
        if not target.expose_node_prefixes:
            continue
        path = _catalog_path(target_name)
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        nodes = payload.get("nodes") if isinstance(payload, dict) else None
        if not isinstance(nodes, dict):
            raise ValueError(f"Remote node catalog {path} must contain a nodes object.")
        for class_type, definition in nodes.items():
            if not any(class_type.startswith(prefix) for prefix in target.expose_node_prefixes):
                continue
            if not isinstance(definition, dict):
                raise ValueError(f"Remote node catalog {path} node {class_type!r} must be an object.")
            node_classes[class_type] = _proxy_node_class(target_name, class_type, definition)
            display_name = str(definition.get("display_name") or class_type)
            display_names[class_type] = f"{display_name} [Remote: {target_name}]"
    return node_classes, display_names


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = load_remote_proxy_nodes()
