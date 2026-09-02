from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .boundary_types import SUPPORTED_BOUNDARY_PORT_TYPES, normalize_boundary_port_type
from .cache_policy import REMOTE_GROUP_CACHE_POLICY_REMOTE, REMOTE_GROUP_CACHE_POLICY_SENDER_V1
from .model_inputs import iter_loader_model_inputs
from .target import remote_target_alias, remote_target_endpoint


MAX_PORTS = 64
COLLAPSED_NODE_HEIGHT = 30
GROUP_EXECUTOR_CLASS = "CutleryRemoteGroupExecutor"
GROUP_VALUE_EXECUTOR_CLASS = "CutleryRemoteGroupValueExecutor"
TARGET_RE = re.compile(r"^([A-Za-z0-9_.-]+):([1-9][0-9]{0,4})$")
CURLY_TARGET_RE = re.compile(r"^([A-Za-z0-9_.-]+):\{([1-9][0-9]{0,4})\}$")
CUTLERY_TARGET_RE = re.compile(r"^cutlery://([A-Za-z0-9_.-]+):([1-9][0-9]{0,4})$")
OUTBOUND_BLOB_ADAPTERS = {
    "mask": ("WF3MaskToBlob", "mask", "WF3MaskFromBlob"),
    "latent": ("WF3LatentToBlob", "latent", "WF3LatentFromBlob"),
    "conditioning": ("WF3ConditioningToBlob", "conditioning", "WF3ConditioningFromBlob"),
}


DefinitionResolver = Callable[..., Any] | Mapping[Any, Any]
ModelRefResolver = Callable[[dict[str, Any]], list[dict[str, Any]]]


@dataclass(frozen=True)
class RemoteGroupCompilation:
    """Detailed result for remote-group compilation.

    ``definition_resolver`` passed to ``compile_editor_remote_groups_detailed``
    is intentionally narrow: it maps a class type to a mapping containing
    ``local`` and ``remote`` node-definition payloads. The payloads may use the
    normalized shape returned by ``node_definitions`` or a compact
    ``{"inputs": ..., "outputs": ...}`` shape. Keeping this boundary data-only
    lets this compiler validate relocation without importing ComfyUI globals.
    """

    compiled: dict[str, Any]
    remaps: dict[str, str]
    targets: list[str]
    relocations: list[dict[str, Any]]
    model_refs: list[dict[str, Any]]
    preparation_manifests: list[dict[str, Any]]


@dataclass(frozen=True)
class _EditorProjection:
    nodes: dict[str, dict[str, Any]]
    groups: list[dict[str, Any]]
    links: list[tuple[str, int, str, int, str]]
    identities: dict[str, dict[str, Any]]


def _remote_target(title: Any) -> str | None:
    text = remote_target_endpoint(title)
    alias = remote_target_alias(text)
    if alias:
        return f"cutlery://{alias}"
    match = TARGET_RE.fullmatch(text) or CURLY_TARGET_RE.fullmatch(text) or CUTLERY_TARGET_RE.fullmatch(text)
    if match is None:
        return None
    port = int(match.group(2))
    if port > 65535:
        return None
    return f"{match.group(1)}:{port}"


def _bounds(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        normalized = tuple(float(item) for item in value[:4])
    except (TypeError, ValueError):
        return None
    return normalized


def _node_bounds(node: dict[str, Any]) -> tuple[float, float, float, float]:
    pos = node.get("pos") if isinstance(node.get("pos"), (list, tuple)) else (0, 0)
    size = node.get("size") if isinstance(node.get("size"), (list, tuple)) else (0, 0)
    flags = node.get("flags") if isinstance(node.get("flags"), dict) else {}
    return (
        float(pos[0] if len(pos) > 0 else 0),
        float(pos[1] if len(pos) > 1 else 0),
        float(size[0] if len(size) > 0 else 0),
        float(COLLAPSED_NODE_HEIGHT if flags.get("collapsed") else size[1] if len(size) > 1 else 0),
    )


def _contains(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    outer_x, outer_y, outer_width, outer_height = outer
    inner_x, inner_y, inner_width, inner_height = inner
    return (
        inner_x >= outer_x
        and inner_y >= outer_y
        and inner_x + inner_width <= outer_x + outer_width
        and inner_y + inner_height <= outer_y + outer_height
    )


def _overlaps(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> bool:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    return (
        first_x < second_x + second_width
        and first_x + first_width > second_x
        and first_y < second_y + second_height
        and first_y + first_height > second_y
    )


def _prompt_link(value: Any, prompt: dict[str, Any]) -> tuple[str, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    source_id, source_slot = value
    if not isinstance(source_id, str) or source_id not in prompt:
        return None
    if not isinstance(source_slot, int) or source_slot < 0:
        return None
    return source_id, source_slot


def _subgraph_definitions(workflow: dict[str, Any]) -> dict[str, dict[str, Any]]:
    definitions = workflow.get("definitions")
    values = definitions.get("subgraphs") if isinstance(definitions, Mapping) else None
    if isinstance(values, Mapping):
        return {
            str(definition.get("id") or definition_id): definition
            for definition_id, definition in values.items()
            if isinstance(definition, dict)
        }
    elif isinstance(values, list):
        candidates = values
    else:
        candidates = ()
    return {
        str(definition["id"]): definition
        for definition in candidates
        if isinstance(definition, dict) and definition.get("id") is not None
    }


def _editor_projection(workflow: dict[str, Any]) -> _EditorProjection:
    definitions = _subgraph_definitions(workflow)
    nodes: dict[str, dict[str, Any]] = {}
    groups: list[dict[str, Any]] = []
    links: list[tuple[str, int, str, int, str]] = []
    identities: dict[str, dict[str, Any]] = {}

    def visit(
        graph: Mapping[str, Any],
        prefix: str,
        parent_instance: str,
        ancestors: frozenset[str],
    ) -> None:
        local_nodes = {
            str(node["id"]): node
            for node in graph.get("nodes") or []
            if isinstance(node, dict) and node.get("id") is not None
        }
        for local_id, node in local_nodes.items():
            qualified_id = f"{prefix}{local_id}"
            nodes[qualified_id] = node
            identities[qualified_id] = {
                "display_node_id": local_id,
                "parent_node_id": node.get("parentId") or parent_instance,
                "real_node_id": node.get("real_node_id") or (local_id if prefix else ""),
                "subgraph_instance": node.get("subgraph_instance") or parent_instance,
            }

        for group in graph.get("groups") or []:
            if not isinstance(group, dict):
                continue
            target = _remote_target(group.get("title"))
            bounds = _bounds(group.get("bounding"))
            if target and bounds:
                groups.append(
                    {
                        "target": target,
                        "bounds": bounds,
                        "inside_ids": set(),
                        "node_ids": frozenset(f"{prefix}{local_id}" for local_id in local_nodes),
                    }
                )

        for link in graph.get("links") or []:
            if not isinstance(link, (list, tuple)) or len(link) < 6:
                continue
            source_id = str(link[1])
            target_id = str(link[3])
            if source_id not in local_nodes or target_id not in local_nodes:
                continue
            try:
                source_slot = int(link[2])
                target_slot = int(link[4])
            except (TypeError, ValueError):
                continue
            links.append(
                (
                    f"{prefix}{source_id}",
                    source_slot,
                    f"{prefix}{target_id}",
                    target_slot,
                    str(link[5] or "").strip(),
                )
            )

        for local_id, node in local_nodes.items():
            definition_id = str(node.get("type") or node.get("class_type") or "").strip()
            definition = definitions.get(definition_id)
            if definition is None or definition_id in ancestors:
                continue
            qualified_id = f"{prefix}{local_id}"
            visit(
                definition,
                f"{qualified_id}:",
                qualified_id,
                ancestors | {definition_id},
            )

    visit(workflow, "", "", frozenset())
    return _EditorProjection(nodes, groups, links, identities)


def _remote_groups(
    workflow: dict[str, Any],
    prompt: dict[str, Any],
    projection: _EditorProjection | None = None,
) -> list[dict[str, Any]]:
    projection = projection or _editor_projection(workflow)
    nodes = projection.nodes
    groups = copy.deepcopy(projection.groups)

    for node_id, node in nodes.items():
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, dict):
            continue
        class_type = str(node.get("type") or node.get("class_type") or "").strip()
        if class_type != str(prompt_node.get("class_type") or "").strip():
            continue
        node_bounds = _node_bounds(node)
        matches = [
            group
            for group in groups
            if node_id in group["node_ids"] and _contains(group["bounds"], node_bounds)
        ]
        partial = [
            group
            for group in groups
            if node_id in group["node_ids"]
            and _overlaps(group["bounds"], node_bounds)
            and not _contains(group["bounds"], node_bounds)
        ]
        if partial:
            raise ValueError(
                f"Node {node_id} partially overlaps a Cutlery remote group. Move it fully inside or outside."
            )
        if len(matches) > 1:
            raise ValueError(f"Node {node_id} is inside multiple Cutlery remote groups.")
        if matches:
            matches[0]["inside_ids"].add(node_id)
    for group in groups:
        group.pop("node_ids", None)
    return [group for group in groups if group["inside_ids"]]


def _socket_type(
    projection: _EditorProjection,
    source_id: str,
    source_slot: int,
    target_id: str,
    target_input_name: str,
) -> str:
    nodes = projection.nodes
    source = nodes.get(source_id, {})
    outputs = source.get("outputs") if isinstance(source.get("outputs"), list) else []
    if source_slot < len(outputs) and isinstance(outputs[source_slot], dict):
        value = str(outputs[source_slot].get("type") or "").strip()
        if value and value != "*":
            return value

    target = nodes.get(target_id, {})
    for input_record in target.get("inputs") or []:
        if not isinstance(input_record, dict) or str(input_record.get("name") or "") != target_input_name:
            continue
        value = str(input_record.get("type") or "").strip()
        if value and value != "*":
            return value

    for link in projection.links:
        if (
            link[0] == source_id
            and link[1] == source_slot
            and link[2] == target_id
        ):
            return link[4]
    return ""


def _boundary_type(
    projection: _EditorProjection,
    group: dict[str, Any],
    link: tuple[str, int],
    direction: str,
    target_id: str,
    target_input_name: str,
) -> str:
    socket_type = _socket_type(projection, link[0], link[1], target_id, target_input_name)
    port_type = normalize_boundary_port_type(socket_type)
    if port_type not in SUPPORTED_BOUNDARY_PORT_TYPES:
        shown_type = socket_type or "unknown"
        raise ValueError(
            f'Cutlery remote group "{group["target"]}" {direction} boundary uses unsupported socket type '
            f'"{shown_type}". Loaded runtime objects such as MODEL, CLIP, and VAE cannot cross machines.'
        )
    if direction == "inbound" and port_type == "video":
        raise ValueError(
            f'Cutlery remote group "{group["target"]}" has an inbound VIDEO boundary, which is not supported.'
        )
    if direction == "outbound" and port_type == "cutlery_lora_chain":
        raise ValueError(
            f'Cutlery remote group "{group["target"]}" has an outbound CUTLERY_LORA_CHAIN boundary, '
            "which is not supported."
        )
    return port_type


def _unique_id(prompt: dict[str, Any], preferred: str) -> str:
    if preferred not in prompt:
        return preferred
    suffix = 2
    while f"{preferred}_{suffix}" in prompt:
        suffix += 1
    return f"{preferred}_{suffix}"


def _check_capacity(group: dict[str, Any], direction: str, count: int) -> None:
    if count > MAX_PORTS:
        raise ValueError(
            f'Cutlery remote group "{group["target"]}" requires {count} distinct {direction} boundary values, '
            f"but {GROUP_EXECUTOR_CLASS} supports at most {MAX_PORTS}."
        )


def _is_supported_boundary(socket_type: str) -> bool:
    return normalize_boundary_port_type(socket_type) in SUPPORTED_BOUNDARY_PORT_TYPES


def _definition_pair(
    resolver: DefinitionResolver | None,
    class_type: str,
    target: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if resolver is None:
        raise ValueError(
            f"Cannot relocate {class_type!r}: node definition metadata is required to verify the remote schema."
        )
    if callable(resolver):
        try:
            entry = resolver(target, class_type)
        except TypeError:
            entry = resolver(class_type)
    else:
        entry = resolver.get((target, class_type), resolver.get(class_type))
    if not isinstance(entry, Mapping):
        raise ValueError(f"Cannot relocate {class_type!r}: no local and remote node definitions were supplied.")
    local = entry.get("local")
    remote = entry.get("remote")
    if not isinstance(local, Mapping) or not isinstance(remote, Mapping):
        raise ValueError(
            f"Cannot relocate {class_type!r}: definition metadata must contain local and remote definitions."
        )
    return local, remote


def _schema_signature(definition: Mapping[str, Any]) -> dict[str, Any]:
    if definition.get("ok") is False:
        raise ValueError(f"Cannot relocate {definition.get('class_type')!r}: supplied node definition is invalid.")
    signature = definition.get("signature")
    if isinstance(signature, Mapping):
        return {
            "input_is_list": bool(signature.get("input_is_list", False)),
            "inputs": signature.get("inputs", {}),
            "outputs": signature.get("outputs", []),
        }
    return {
        "input_is_list": bool(definition.get("input_is_list", False)),
        "inputs": definition.get("inputs", {}),
        "outputs": definition.get("outputs", []),
    }


def _definition_input_names(definition: Mapping[str, Any]) -> set[str]:
    inputs = definition.get("inputs", {})
    if not isinstance(inputs, Mapping):
        return set()
    names: set[str] = set()
    for section in ("required", "optional", "hidden"):
        section_inputs = inputs.get(section)
        if isinstance(section_inputs, Mapping):
            names.update(str(name) for name in section_inputs)
    if not names:
        names.update(str(name) for name in inputs)
    return names


def _definition_flag(definition: Mapping[str, Any], name: str) -> bool:
    if bool(definition.get(name)) or bool(definition.get(name.upper())):
        return True
    metadata = definition.get("metadata")
    if isinstance(metadata, Mapping) and (bool(metadata.get(name)) or bool(metadata.get(name.upper()))):
        return True
    cache = definition.get("cache")
    return isinstance(cache, Mapping) and bool(cache.get(name.lower()))


def _group_executor_class(
    prompt: dict[str, Any],
    group: dict[str, Any],
    outbound: list[dict[str, Any]],
    resolver: DefinitionResolver | None,
    partial_execution_targets: set[str],
) -> str:
    if group["inside_ids"] & partial_execution_targets or not outbound:
        return GROUP_EXECUTOR_CLASS
    if resolver is None:
        return GROUP_VALUE_EXECUTOR_CLASS
    for node_id in sorted(group["inside_ids"]):
        node = prompt.get(node_id)
        if not isinstance(node, Mapping):
            continue
        class_type = str(node.get("class_type") or "").strip()
        if not class_type:
            continue
        try:
            local, remote = _definition_pair(resolver, class_type, group["target"])
        except ValueError:
            continue
        if _definition_flag(local, "output_node") and _definition_flag(remote, "output_node"):
            return GROUP_EXECUTOR_CLASS
    return GROUP_VALUE_EXECUTOR_CLASS


def _cache_declared_inputs_only(definition: Mapping[str, Any]) -> bool:
    cache = definition.get("cache")
    return isinstance(cache, Mapping) and cache.get("declared_inputs_only") is True


def _group_cache_policy(
    wrapper_class: str,
    prompt: dict[str, Any],
    remote_ids: set[str],
    resolver: DefinitionResolver | None,
    target: str,
) -> str:
    if wrapper_class != GROUP_VALUE_EXECUTOR_CLASS or resolver is None:
        return REMOTE_GROUP_CACHE_POLICY_REMOTE
    for node_id in sorted(remote_ids):
        node = prompt.get(node_id)
        if not isinstance(node, Mapping):
            return REMOTE_GROUP_CACHE_POLICY_REMOTE
        class_type = str(node.get("class_type") or "").strip()
        if not class_type:
            return REMOTE_GROUP_CACHE_POLICY_REMOTE
        try:
            local, remote = _definition_pair(resolver, class_type, target)
        except ValueError:
            return REMOTE_GROUP_CACHE_POLICY_REMOTE
        if not (_cache_declared_inputs_only(local) and _cache_declared_inputs_only(remote)):
            return REMOTE_GROUP_CACHE_POLICY_REMOTE
    return REMOTE_GROUP_CACHE_POLICY_SENDER_V1


def _validate_relocatable_recipe(
    prompt: dict[str, Any],
    node_id: str,
    resolver: DefinitionResolver | None,
    target: str,
) -> None:
    node = prompt.get(node_id)
    if not isinstance(node, Mapping):
        raise ValueError(f"Cannot relocate producer {node_id!r}: it is missing from the API prompt.")
    class_type = str(node.get("class_type") or "").strip()
    if not class_type:
        raise ValueError(f"Cannot relocate producer {node_id!r}: it has no class_type.")
    local, remote = _definition_pair(resolver, class_type, target)
    if _definition_flag(local, "output_node") or _definition_flag(remote, "output_node"):
        raise ValueError(f"Cannot relocate producer {node_id!r} ({class_type}): OUTPUT_NODE recipes are not safe to duplicate.")
    if _definition_flag(local, "not_idempotent") or _definition_flag(remote, "not_idempotent"):
        raise ValueError(f"Cannot relocate producer {node_id!r} ({class_type}): NOT_IDEMPOTENT recipes are not safe to duplicate.")
    if _schema_signature(local) != _schema_signature(remote):
        raise ValueError(f"Cannot relocate producer {node_id!r} ({class_type}): local and remote schemas differ.")
    declared_inputs = _definition_input_names(local)
    recipe_inputs = node.get("inputs", {})
    if not isinstance(recipe_inputs, Mapping):
        raise ValueError(f"Cannot relocate producer {node_id!r} ({class_type}): inputs must be a mapping.")
    unknown_inputs = sorted(set(str(name) for name in recipe_inputs) - declared_inputs)
    if unknown_inputs:
        raise ValueError(
            f"Cannot relocate producer {node_id!r} ({class_type}): recipe inputs are absent from the schema: "
            f"{', '.join(unknown_inputs)}."
        )


def _group_membership(groups: list[dict[str, Any]]) -> dict[str, int]:
    owners: dict[str, int] = {}
    for index, group in enumerate(groups):
        for node_id in group["inside_ids"]:
            owners[node_id] = index
    return owners


def _reject_cross_target_links(prompt: dict[str, Any], owners: dict[str, int], groups: list[dict[str, Any]]) -> None:
    for target_id, node in prompt.items():
        if not isinstance(node, Mapping):
            continue
        target_group = owners.get(target_id)
        if target_group is None:
            continue
        for value in (node.get("inputs") or {}).values():
            link = _prompt_link(value, prompt)
            if link is None:
                continue
            source_group = owners.get(link[0])
            if source_group is not None and source_group != target_group:
                raise ValueError(
                    "Cutlery remote groups cannot depend on another remote target "
                    f"({groups[source_group]['target']} -> {groups[target_group]['target']})."
                )


def _plan_group_relocation(
    projection: _EditorProjection,
    prompt: dict[str, Any],
    group: dict[str, Any],
    group_index: int,
    owners: dict[str, int],
    groups: list[dict[str, Any]],
    resolver: DefinitionResolver | None,
) -> set[str]:
    relocated: set[str] = set()
    visiting: set[str] = set()
    remote_ids = group["inside_ids"]

    def visit(node_id: str) -> None:
        owner = owners.get(node_id)
        if owner == group_index:
            return
        if owner is not None:
            raise ValueError(
                f'Cutlery remote group "{group["target"]}" relocation crosses remote target '
                f'"{groups[owner]["target"]}" through producer {node_id!r}.'
            )
        if node_id in relocated:
            return
        if node_id in visiting:
            raise ValueError(
                f'Cutlery remote group "{group["target"]}" relocation closure contains a dependency cycle at {node_id!r}.'
            )
        visiting.add(node_id)
        _validate_relocatable_recipe(prompt, node_id, resolver, group["target"])
        node = prompt[node_id]
        for input_name, value in (node.get("inputs") or {}).items():
            link = _prompt_link(value, prompt)
            if link is None:
                continue
            if link[0] in remote_ids:
                continue
            socket_type = _socket_type(projection, link[0], link[1], node_id, input_name)
            if not _is_supported_boundary(socket_type):
                visit(link[0])
        visiting.remove(node_id)
        relocated.add(node_id)

    for node_id in sorted(remote_ids):
        node = prompt.get(node_id)
        if not isinstance(node, Mapping):
            continue
        for input_name, value in (node.get("inputs") or {}).items():
            link = _prompt_link(value, prompt)
            if link is None or link[0] in remote_ids:
                continue
            socket_type = _socket_type(projection, link[0], link[1], node_id, input_name)
            if not _is_supported_boundary(socket_type):
                visit(link[0])
    return relocated


def _model_refs(
    remote_prompt: dict[str, Any],
    model_ref_resolver: ModelRefResolver | None = None,
) -> list[dict[str, Any]]:
    refs = []
    for ref in iter_loader_model_inputs(remote_prompt) or ():
        refs.append(
            {
                "node_id": ref.node_id,
                "class_type": ref.class_type,
                "input_name": ref.input_name,
                "model_types": list(ref.model_types),
                "model_name": ref.model_name,
            }
        )
    if model_ref_resolver is not None:
        registered = {(ref["node_id"], ref["input_name"]) for ref in refs}
        for ref in model_ref_resolver(remote_prompt):
            if (ref.get("node_id"), ref.get("input_name")) not in registered:
                refs.append(dict(ref))
    return sorted(refs, key=lambda value: (value["node_id"], value["input_name"]))


def _unique_remote_id(remote_prompt: dict[str, Any], preferred: str) -> str:
    if preferred not in remote_prompt:
        return preferred
    suffix = 2
    while f"{preferred}_{suffix}" in remote_prompt:
        suffix += 1
    return f"{preferred}_{suffix}"


def _preload_workflow(
    projection: _EditorProjection,
    remote_prompt: dict[str, Any],
    relocated_ids: set[str],
    group_index: int,
) -> tuple[dict[str, Any], list[str]]:
    eligible: set[str] = set()
    remaining = set(relocated_ids)
    while remaining:
        resolved = set()
        for node_id in remaining:
            node = remote_prompt.get(node_id)
            if not isinstance(node, Mapping):
                continue
            dependencies = []
            complete = True
            for value in (node.get("inputs") or {}).values():
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    continue
                source_id, source_slot = value
                if not isinstance(source_id, str) or not isinstance(source_slot, int) or source_slot < 0:
                    continue
                if source_id not in relocated_ids:
                    complete = False
                    break
                dependencies.append(source_id)
            if complete and all(dependency in eligible for dependency in dependencies):
                resolved.add(node_id)
        if not resolved:
            break
        eligible.update(resolved)
        remaining.difference_update(resolved)

    preload = {
        node_id: copy.deepcopy(remote_prompt[node_id])
        for node_id in sorted(eligible)
    }
    editor_nodes = projection.nodes
    runtime_sources = []
    for node_id in sorted(eligible):
        outputs = editor_nodes.get(node_id, {}).get("outputs") or []
        for output_index, output in enumerate(outputs):
            socket_type = str(output.get("type") or "") if isinstance(output, Mapping) else ""
            if not _is_supported_boundary(socket_type):
                runtime_sources.append([node_id, output_index])
    if runtime_sources:
        preload_id = _unique_remote_id(preload, f"cutlery_remote_preload_{group_index + 1}")
        preload[preload_id] = {
            "class_type": "CutleryRemoteModelPreload",
            "inputs": {
                f"value_{index}": source
                for index, source in enumerate(runtime_sources, start=1)
            },
        }
        return preload, [preload_id]
    return preload, []


def _progress_map(
    projection: _EditorProjection,
    remote_ids: set[str],
    remote_prompt: dict[str, Any],
    preload_helper_ids: list[str],
    wrapper_id: str,
    target: str,
) -> dict[str, dict[str, Any]]:
    result = {}
    for node_id in sorted(remote_ids):
        identity = projection.identities.get(node_id, {})
        result[node_id] = {
            "api_node_id": node_id,
            "display_node_id": identity.get("display_node_id", node_id),
            "parent_node_id": identity.get("parent_node_id", ""),
            "real_node_id": identity.get("real_node_id", ""),
            "subgraph_instance": identity.get("subgraph_instance", ""),
            "group": wrapper_id,
            "target": target,
            "visible": True,
        }
    for node_id in sorted((set(remote_prompt) - remote_ids) | set(preload_helper_ids)):
        result[node_id] = {
            "api_node_id": "",
            "display_node_id": "",
            "parent_node_id": "",
            "real_node_id": "",
            "subgraph_instance": "",
            "group": wrapper_id,
            "target": target,
            "visible": False,
        }
    return result


def _compile_group(
    projection: _EditorProjection,
    prompt: dict[str, Any],
    group: dict[str, Any],
    group_index: int,
    relocated_ids: set[str] | None = None,
    model_ref_resolver: ModelRefResolver | None = None,
    definition_resolver: DefinitionResolver | None = None,
    partial_execution_targets: set[str] | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    inside_ids = group["inside_ids"]
    remote_ids = inside_ids | (relocated_ids or set())
    wrapper_id = _unique_id(prompt, f"cutlery_remote_group_{group_index + 1}")
    output_boundary_id = _unique_id(prompt, f"cutlery_remote_output_{group_index + 1}")
    remote_prompt = {
        node_id: copy.deepcopy(prompt[node_id])
        for node_id in sorted(remote_ids)
        if node_id in prompt
    }
    inbound: list[dict[str, Any]] = []
    outbound: list[dict[str, Any]] = []
    inbound_by_link: dict[tuple[str, int, str], dict[str, Any]] = {}
    outbound_by_link: dict[tuple[str, int], dict[str, Any]] = {}

    for node_id, node in remote_prompt.items():
        for input_name, value in list((node.get("inputs") or {}).items()):
            link = _prompt_link(value, prompt)
            if link is None or link[0] in remote_ids:
                continue
            port_type = _boundary_type(projection, group, link, "inbound", node_id, input_name)
            key = (link[0], link[1], port_type)
            if key not in inbound_by_link:
                _check_capacity(group, "inbound", len(inbound) + 1)
                port = {
                    "index": len(inbound),
                    "name": f"input_{len(inbound) + 1}",
                    "type": port_type,
                    "source": [link[0], link[1]],
                    "boundary_id": _unique_id(
                        prompt,
                        f"cutlery_remote_input_{group_index + 1}_{len(inbound) + 1}",
                    ),
                }
                inbound_by_link[key] = port
                inbound.append(port)
            node["inputs"][input_name] = [inbound_by_link[key]["boundary_id"], 0]

    for node_id, node in prompt.items():
        if node_id in inside_ids:
            continue
        for input_name, value in list((node.get("inputs") or {}).items()):
            link = _prompt_link(value, prompt)
            if link is None or link[0] not in inside_ids:
                continue
            key = (link[0], link[1])
            port_type = _boundary_type(projection, group, link, "outbound", node_id, input_name)
            if key not in outbound_by_link:
                _check_capacity(group, "outbound", len(outbound) + 1)
                port = {
                    "index": len(outbound),
                    "name": f"output_{len(outbound) + 1}",
                    "type": port_type,
                    "source": [link[0], link[1]],
                    "consumers": [],
                }
                outbound_by_link[key] = port
                outbound.append(port)
            outbound_by_link[key]["consumers"].append((node, input_name))

    for port in outbound:
        adapter = OUTBOUND_BLOB_ADAPTERS.get(port["type"])
        if adapter:
            encode_class, encode_input, decode_class = adapter
            encode_id = _unique_id(
                prompt,
                f"cutlery_remote_encode_{group_index + 1}_{port['index'] + 1}",
            )
            decode_id = _unique_id(
                prompt,
                f"cutlery_remote_decode_{group_index + 1}_{port['index'] + 1}",
            )
            remote_prompt[encode_id] = {
                "class_type": encode_class,
                "inputs": {encode_input: port["source"], "compress": True},
            }
            prompt[decode_id] = {
                "class_type": decode_class,
                "inputs": {"blob": [wrapper_id, port["index"]], "device": "auto"},
            }
            port["remote_source"] = [encode_id, 0]
            port["local_source"] = [decode_id, 0]
        for consumer, input_name in port["consumers"]:
            consumer["inputs"][input_name] = port.get("local_source", [wrapper_id, port["index"]])

    if inbound:
        for port in inbound:
            remote_prompt[port["boundary_id"]] = {
                "class_type": "CutleryWorkflowInput",
                "inputs": {
                    "ports_json": json.dumps(
                        [{"name": port["name"], "type": port["type"]}],
                        separators=(",", ":"),
                    )
                },
            }
    else:
        input_boundary_id = _unique_id(prompt, f"cutlery_remote_input_{group_index + 1}")
        remote_prompt[input_boundary_id] = {
            "class_type": "CutleryWorkflowInput",
            "inputs": {"ports_json": "[]"},
        }

    output_inputs = {
        "ports_json": json.dumps(
            [{"name": port["name"], "type": "json" if port["type"] in OUTBOUND_BLOB_ADAPTERS else port["type"]} for port in outbound],
            separators=(",", ":"),
        )
    }
    for index, port in enumerate(outbound, start=1):
        output_inputs[f"value_{index}"] = port.get("remote_source", port["source"])
    remote_prompt[output_boundary_id] = {
        "class_type": "CutleryWorkflowOutput",
        "inputs": output_inputs,
    }

    wrapper_class = _group_executor_class(
        prompt,
        group,
        outbound,
        definition_resolver,
        partial_execution_targets or set(),
    )
    cache_policy = _group_cache_policy(
        wrapper_class,
        prompt,
        remote_ids,
        definition_resolver,
        group["target"],
    )

    for node_id in inside_ids:
        prompt.pop(node_id, None)
    model_refs = _model_refs(remote_prompt, model_ref_resolver)
    preload_workflow, preload_helper_ids = _preload_workflow(
        projection,
        remote_prompt,
        relocated_ids or set(),
        group_index,
    )
    preparation_manifest = {
        "target": group["target"],
        "model_refs": model_refs,
        "preload_node_ids": sorted(node_id for node_id in preload_workflow if node_id in remote_ids),
    }
    progress_map = _progress_map(
        projection,
        remote_ids,
        remote_prompt,
        preload_helper_ids,
        wrapper_id,
        group["target"],
    )
    preparation_id = _unique_id(prompt, f"cutlery_remote_prepare_{group_index + 1}")
    prompt[preparation_id] = {
        "class_type": "CutleryRemoteGroupPreparation",
        "inputs": {
            "remote_base_url": group["target"],
            "remote_workflow_json": json.dumps(remote_prompt, separators=(",", ":")),
            "model_refs_json": json.dumps(model_refs, separators=(",", ":"), sort_keys=True),
            "preparation_manifest_json": json.dumps(preparation_manifest, separators=(",", ":"), sort_keys=True),
            "preload_workflow_json": json.dumps(preload_workflow, separators=(",", ":")),
            "timeout_seconds": 300,
            "cache_policy": cache_policy,
        },
    }
    wrapper_inputs = {
        "remote_base_url": group["target"],
        "remote_workflow_json": json.dumps(remote_prompt, separators=(",", ":")),
        "input_ports_json": json.dumps(
            [{"name": port["name"], "type": port["type"]} for port in inbound],
            separators=(",", ":"),
        ),
        "output_ports_json": output_inputs["ports_json"],
        "timeout_seconds": 300,
        "cache_policy": cache_policy,
        "progress_map_json": json.dumps(progress_map, separators=(",", ":"), sort_keys=True),
        "model_refs_json": json.dumps(model_refs, separators=(",", ":"), sort_keys=True),
        "preparation_manifest_json": json.dumps(preparation_manifest, separators=(",", ":"), sort_keys=True),
        "preparation": [preparation_id, 0],
    }
    for index, port in enumerate(inbound, start=1):
        wrapper_inputs[f"value_{index}"] = port["source"]
    prompt[wrapper_id] = {
        "class_type": wrapper_class,
        "inputs": wrapper_inputs,
    }
    return wrapper_id, model_refs, preparation_manifest


def compile_editor_remote_groups_detailed(
    workflow: dict[str, Any],
    prompt: dict[str, Any],
    *,
    definition_resolver: DefinitionResolver | None = None,
    model_ref_resolver: ModelRefResolver | None = None,
    partial_execution_targets: Iterable[object] | None = None,
) -> RemoteGroupCompilation:
    """Compile remote groups and, when needed, relocate safe producer closures.

    Runtime-object inputs are reconstructed on the destination only after the
    supplied resolver proves that the local and remote node schemas match. The
    resolver is not consulted for ordinary serializable boundaries.
    """

    projection = _editor_projection(workflow)
    groups = _remote_groups(workflow, prompt, projection)
    if not groups:
        return RemoteGroupCompilation(prompt, {}, [], [], [], [])
    compiled = copy.deepcopy(prompt)
    selected_targets = {str(node_id) for node_id in partial_execution_targets or ()}
    owners = _group_membership(groups)
    _reject_cross_target_links(compiled, owners, groups)
    relocation_sets = [
        _plan_group_relocation(
            projection,
            compiled,
            group,
            index,
            owners,
            groups,
            definition_resolver,
        )
        for index, group in enumerate(groups)
    ]
    remaps: dict[str, str] = {}
    targets: list[str] = []
    relocations: list[dict[str, Any]] = []
    model_refs: list[dict[str, Any]] = []
    preparation_manifests: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        wrapper_id, group_model_refs, preparation_manifest = _compile_group(
            projection,
            compiled,
            group,
            index,
            relocation_sets[index],
            model_ref_resolver,
            definition_resolver,
            selected_targets,
        )
        remaps.update({node_id: wrapper_id for node_id in group["inside_ids"]})
        targets.append(group["target"])
        relocated_node_ids = sorted(relocation_sets[index])
        relocation = {
            "group_index": index,
            "wrapper_id": wrapper_id,
            "target": group["target"],
            "relocated_node_ids": relocated_node_ids,
        }
        relocations.append(relocation)
        model_refs.extend({"target": group["target"], **ref} for ref in group_model_refs)
        preparation_manifests.append(preparation_manifest)

    relocated_nodes = set().union(*relocation_sets) if relocation_sets else set()
    retained_relocated_ids: set[str] = set()
    for node_id in relocated_nodes:
        for consumer_id, consumer in compiled.items():
            if not isinstance(consumer, Mapping):
                continue
            for value in (consumer.get("inputs") or {}).values():
                link = _prompt_link(value, compiled)
                if link is not None and link[0] == node_id:
                    if consumer_id not in relocated_nodes:
                        retained_relocated_ids.add(node_id)
    pending = list(retained_relocated_ids)
    while pending:
        node_id = pending.pop()
        node = compiled.get(node_id)
        if not isinstance(node, Mapping):
            continue
        for value in (node.get("inputs") or {}).values():
            link = _prompt_link(value, compiled)
            if link is not None and link[0] in relocated_nodes and link[0] not in retained_relocated_ids:
                retained_relocated_ids.add(link[0])
                pending.append(link[0])
    removed_relocated_ids = []
    for node_id in sorted(relocated_nodes - retained_relocated_ids):
        compiled.pop(node_id, None)
        removed_relocated_ids.append(node_id)
    removed_set = set(removed_relocated_ids)
    for relocation in relocations:
        relocation["removed_local_node_ids"] = [
            node_id for node_id in relocation["relocated_node_ids"] if node_id in removed_set
        ]

    return RemoteGroupCompilation(
        compiled,
        remaps,
        targets,
        relocations,
        model_refs,
        preparation_manifests,
    )


def editor_remote_group_targets(workflow: dict[str, Any], prompt: dict[str, Any]) -> list[str]:
    projection = _editor_projection(workflow)
    return [group["target"] for group in _remote_groups(workflow, prompt, projection)]


def compile_editor_remote_groups(
    workflow: dict[str, Any],
    prompt: dict[str, Any],
    *,
    definition_resolver: DefinitionResolver | None = None,
    partial_execution_targets: Iterable[object] | None = None,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Compile editor remote groups, preserving the legacy tuple API."""

    result = compile_editor_remote_groups_detailed(
        workflow,
        prompt,
        definition_resolver=definition_resolver,
        partial_execution_targets=partial_execution_targets,
    )
    return result.compiled, result.remaps, result.targets
