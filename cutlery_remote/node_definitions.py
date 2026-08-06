from __future__ import annotations

import math
import sys
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from .model_inputs import LOADER_MODEL_INPUTS


DEFAULT_MAX_CLASS_TYPES = 128
SHARED_EXTENSION_MODULE = "_cutlery_remote_registry_extensions"
DEFAULT_MAX_OPTIONS_PER_INPUT = 4096
INPUT_SECTIONS = ("required", "optional", "hidden")
DYNAMIC_INPUT_TYPES = {
    "COMFY_DYNAMICCOMBO_V3",
    "COMFY_DYNAMICSLOT_V3",
}
UPLOAD_CONFIG_KEYS = {
    "image_upload",
    "audio_upload",
    "video_upload",
    "file_upload",
}
BROWSER_OWNED_INPUT_REGISTRIES = {
    "CutleryRemoteClipTextEncode": {
        "text_encoder": "cutlery.remote_clip.v1",
        "clip_type": "cutlery.remote_clip.v1",
    },
    "CutleryRemoteDualClipTextEncode": {
        "clip_name1": "cutlery.remote_clip.v1",
        "clip_name2": "cutlery.remote_clip.v1",
        "clip_type": "cutlery.remote_clip.v1",
    },
    "CutleryRemoteTextEncodeQwenImageEditPlus": {
        "text_encoder": "cutlery.remote_clip.v1",
        "vae_name": "cutlery.remote_clip.v1",
    },
}


def _install_shared_browser_input_registries() -> set[str]:
    extension = sys.modules.get(SHARED_EXTENSION_MODULE)
    contracts = getattr(extension, "browser_input_registries", None)
    if not isinstance(contracts, dict):
        return set()

    installed: set[str] = set()
    for class_type, inputs in contracts.items():
        if not isinstance(class_type, str) or not isinstance(inputs, dict):
            continue
        if not all(isinstance(name, str) and isinstance(registry, str) for name, registry in inputs.items()):
            continue
        BROWSER_OWNED_INPUT_REGISTRIES[class_type] = dict(inputs)
        installed.add(class_type)
    return installed


_install_shared_browser_input_registries()


class NodeDefinitionRequestError(ValueError):
    """Raised when a node-definition batch request is invalid before class inspection."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _default_node_class_mappings() -> Mapping[str, Any]:
    import nodes  # type: ignore

    return nodes.NODE_CLASS_MAPPINGS


def _validate_positive_limit(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NodeDefinitionRequestError(
            "invalid_limit",
            f"{name} must be a positive integer.",
        )
    return value


def _normalize_class_types(class_types: object, max_class_types: int) -> tuple[list[str], int]:
    if not isinstance(class_types, (list, tuple)):
        raise NodeDefinitionRequestError(
            "invalid_class_types",
            "class_types must be an array of non-empty strings.",
        )
    if len(class_types) > max_class_types:
        raise NodeDefinitionRequestError(
            "class_limit_exceeded",
            f"class_types contains {len(class_types)} entries; the limit is {max_class_types}.",
        )

    normalized: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(class_types):
        if not isinstance(value, str) or not value.strip():
            raise NodeDefinitionRequestError(
                "invalid_class_type",
                f"class_types[{index}] must be a non-empty string.",
            )
        class_type = value.strip()
        if class_type not in seen:
            seen.add(class_type)
            normalized.append(class_type)
    return normalized, len(class_types)


def _type_label(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, type):
        return value.__name__
    return str(value)


def _json_scalar(value: object) -> tuple[bool, object]:
    if value is None or isinstance(value, (str, bool, int)):
        return True, value
    if isinstance(value, float) and math.isfinite(value):
        return True, value
    return False, None


def _error(code: str, message: str, **context: object) -> dict[str, Any]:
    return {"code": code, "message": message, **context}


def _failed_definition(
    class_type: str,
    *,
    code: str,
    message: str,
    missing: bool = False,
) -> dict[str, Any]:
    cache = {
        "declared_inputs_only": False,
        "has_change_fingerprint": False,
        "not_idempotent": False,
        "output_node": False,
    }
    return {
        "ok": False,
        "missing": missing,
        "class_type": class_type,
        "source": None,
        "input_is_list": False,
        "inputs": {section: {} for section in INPUT_SECTIONS},
        "outputs": [],
        "cache": cache,
        "signature": {
            "input_is_list": False,
            "inputs": {section: [] for section in INPUT_SECTIONS},
            "outputs": [],
            "cache": cache,
        },
        "errors": [_error(code, message)],
    }


def _cache_contract(node_class: Any) -> dict[str, bool]:
    has_change_fingerprint = callable(getattr(node_class, "IS_CHANGED", None))
    try:
        from comfy_api.internal import _ComfyNodeInternal, first_real_override

        if isinstance(node_class, type) and issubclass(node_class, _ComfyNodeInternal):
            has_change_fingerprint = first_real_override(node_class, "fingerprint_inputs") is not None
    except (ImportError, TypeError):
        pass
    not_idempotent = bool(getattr(node_class, "NOT_IDEMPOTENT", False))
    output_node = bool(getattr(node_class, "OUTPUT_NODE", False))
    return {
        "declared_inputs_only": not has_change_fingerprint and not not_idempotent,
        "has_change_fingerprint": has_change_fingerprint,
        "not_idempotent": not_idempotent,
        "output_node": output_node,
    }


def _read_node_info(class_type: str, node_class: Any) -> tuple[str, Mapping[str, Any]]:
    get_node_info = getattr(node_class, "GET_NODE_INFO_V1", None)
    if callable(get_node_info):
        info = get_node_info()
        if not isinstance(info, Mapping):
            raise TypeError(
                f"{class_type}.GET_NODE_INFO_V1() returned {type(info).__name__}, expected a mapping."
            )
        return "GET_NODE_INFO_V1", info

    input_types = getattr(node_class, "INPUT_TYPES", None)
    if not callable(input_types):
        raise TypeError(f"{class_type} does not define INPUT_TYPES().")
    inputs = input_types()
    if not isinstance(inputs, Mapping):
        raise TypeError(
            f"{class_type}.INPUT_TYPES() returned {type(inputs).__name__}, expected a mapping."
        )

    return_types = getattr(node_class, "RETURN_TYPES", ())
    return_names = getattr(node_class, "RETURN_NAMES", return_types)
    output_is_list = getattr(node_class, "OUTPUT_IS_LIST", None)
    return "INPUT_TYPES", {
        "input": inputs,
        "is_input_list": getattr(node_class, "INPUT_IS_LIST", False),
        "output": return_types,
        "output_name": return_names,
        "output_is_list": output_is_list,
    }


def _input_error_entry(
    *,
    input_type: str,
    error: dict[str, Any],
    materializable: bool,
    upload_backed: bool,
) -> dict[str, Any]:
    return {
        "kind": "error",
        "type": input_type,
        "materializable": materializable,
        "upload_backed": upload_backed,
        "error": error,
    }


def _normalize_options(
    options: Sequence[object],
    *,
    class_type: str,
    section: str,
    input_name: str,
    input_type: str,
    materializable: bool,
    upload_backed: bool,
    max_options_per_input: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if len(options) > max_options_per_input:
        error = _error(
            "option_limit_exceeded",
            (
                f"{class_type}.{input_name} exposes {len(options)} options; "
                f"the per-input limit is {max_options_per_input}."
            ),
            section=section,
            input_name=input_name,
            option_count=len(options),
            limit=max_options_per_input,
        )
        return (
            _input_error_entry(
                input_type=input_type,
                error=error,
                materializable=materializable,
                upload_backed=upload_backed,
            ),
            error,
        )

    normalized: list[object] = []
    for index, option in enumerate(options):
        is_valid, scalar = _json_scalar(option)
        if not is_valid:
            error = _error(
                "invalid_option",
                (
                    f"{class_type}.{input_name} option {index} has non-JSON-scalar "
                    f"type {type(option).__name__}."
                ),
                section=section,
                input_name=input_name,
                option_index=index,
            )
            return (
                _input_error_entry(
                    input_type=input_type,
                    error=error,
                    materializable=materializable,
                    upload_backed=upload_backed,
                ),
                error,
            )
        normalized.append(scalar)

    return (
        {
            "kind": "combo",
            "type": input_type,
            "materializable": materializable,
            "upload_backed": upload_backed,
            "option_count": len(normalized),
            "options": normalized,
        },
        None,
    )


def _normalize_input(
    raw_spec: object,
    *,
    class_type: str,
    section: str,
    input_name: str,
    max_options_per_input: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    materializable = input_name in LOADER_MODEL_INPUTS.get(class_type, {})
    browser_registry = BROWSER_OWNED_INPUT_REGISTRIES.get(class_type, {}).get(input_name)
    if browser_registry is not None:
        if isinstance(raw_spec, (list, tuple)) and raw_spec:
            input_type = "COMBO" if isinstance(raw_spec[0], (list, tuple)) else _type_label(raw_spec[0])
            config = raw_spec[1] if len(raw_spec) > 1 and isinstance(raw_spec[1], Mapping) else {}
        else:
            input_type = _type_label(raw_spec)
            config = {}
        return {
            "kind": "dynamic",
            "type": input_type,
            "materializable": False,
            "upload_backed": any(config.get(key) is True for key in UPLOAD_CONFIG_KEYS),
            "registry": browser_registry,
        }, None
    if isinstance(raw_spec, str):
        return {
            "kind": "noncombo",
            "type": raw_spec,
            "materializable": False,
        }, None
    if not isinstance(raw_spec, (list, tuple)) or not raw_spec:
        error = _error(
            "malformed_input",
            f"{class_type}.{input_name} has an invalid input specification.",
            section=section,
            input_name=input_name,
        )
        return (
            _input_error_entry(
                input_type="UNKNOWN",
                error=error,
                materializable=materializable,
                upload_backed=False,
            ),
            error,
        )

    primary = raw_spec[0]
    config = raw_spec[1] if len(raw_spec) > 1 and isinstance(raw_spec[1], Mapping) else {}
    input_type = _type_label(primary)
    upload_backed = any(config.get(key) is True for key in UPLOAD_CONFIG_KEYS)

    if input_type in DYNAMIC_INPUT_TYPES:
        return {
            "kind": "dynamic",
            "type": input_type,
            "materializable": False,
            "upload_backed": upload_backed,
        }, None

    if isinstance(primary, (list, tuple)):
        return _normalize_options(
            primary,
            class_type=class_type,
            section=section,
            input_name=input_name,
            input_type="COMBO",
            materializable=materializable,
            upload_backed=upload_backed,
            max_options_per_input=max_options_per_input,
        )

    if input_type == "COMBO":
        options = config.get("options")
        if options is None or callable(options):
            return {
                "kind": "dynamic",
                "type": input_type,
                "materializable": materializable,
                "upload_backed": upload_backed,
            }, None
        if not isinstance(options, (list, tuple)):
            error = _error(
                "malformed_combo",
                f"{class_type}.{input_name} COMBO options must be an array.",
                section=section,
                input_name=input_name,
            )
            return (
                _input_error_entry(
                    input_type=input_type,
                    error=error,
                    materializable=materializable,
                    upload_backed=upload_backed,
                ),
                error,
            )
        return _normalize_options(
            options,
            class_type=class_type,
            section=section,
            input_name=input_name,
            input_type=input_type,
            materializable=materializable,
            upload_backed=upload_backed,
            max_options_per_input=max_options_per_input,
        )

    return {
        "kind": "noncombo",
        "type": input_type,
        "materializable": False,
    }, None


def _normalize_inputs(
    class_type: str,
    raw_inputs: object,
    *,
    max_options_per_input: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    inputs: dict[str, dict[str, Any]] = {section: {} for section in INPUT_SECTIONS}
    signature: dict[str, list[dict[str, Any]]] = {section: [] for section in INPUT_SECTIONS}
    errors: list[dict[str, Any]] = []
    if not isinstance(raw_inputs, Mapping):
        errors.append(
            _error(
                "malformed_inputs",
                f"{class_type} node definition input must be a mapping.",
            )
        )
        return inputs, signature, errors

    for section_name in raw_inputs:
        if section_name not in INPUT_SECTIONS:
            errors.append(
                _error(
                    "unsupported_input_section",
                    f"{class_type} exposes unsupported input section {section_name!r}.",
                    section=str(section_name),
                )
            )

    for section in INPUT_SECTIONS:
        raw_section = raw_inputs.get(section, {})
        if raw_section is None:
            continue
        if not isinstance(raw_section, Mapping):
            errors.append(
                _error(
                    "malformed_input_section",
                    f"{class_type} input section {section!r} must be a mapping.",
                    section=section,
                )
            )
            continue

        for raw_name, raw_spec in raw_section.items():
            if not isinstance(raw_name, str) or not raw_name:
                errors.append(
                    _error(
                        "invalid_input_name",
                        f"{class_type} input names must be non-empty strings.",
                        section=section,
                    )
                )
                continue
            entry, entry_error = _normalize_input(
                raw_spec,
                class_type=class_type,
                section=section,
                input_name=raw_name,
                max_options_per_input=max_options_per_input,
            )
            inputs[section][raw_name] = entry
            signature[section].append(
                {
                    "name": raw_name,
                    "kind": entry["kind"],
                    "type": entry["type"],
                }
            )
            if entry_error is not None:
                errors.append(entry_error)

    return inputs, signature, errors


def _as_sequence(value: object) -> Sequence[object] | None:
    if isinstance(value, (list, tuple)):
        return value
    return None


def _normalize_outputs(
    class_type: str,
    info: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    raw_outputs = info.get("output", ())
    output_types = _as_sequence(raw_outputs)
    if output_types is None:
        return [], [
            _error(
                "malformed_outputs",
                f"{class_type} node definition output must be an array.",
            )
        ]

    raw_names = info.get("output_name")
    output_names = _as_sequence(raw_names) if raw_names is not None else None
    if output_names is not None and len(output_names) != len(output_types):
        errors.append(
            _error(
                "output_name_count_mismatch",
                (
                    f"{class_type} exposes {len(output_types)} output types but "
                    f"{len(output_names)} output names."
                ),
            )
        )

    raw_is_list = info.get("output_is_list")
    output_is_list = _as_sequence(raw_is_list) if raw_is_list is not None else None
    if output_is_list is not None and len(output_is_list) != len(output_types):
        errors.append(
            _error(
                "output_list_count_mismatch",
                (
                    f"{class_type} exposes {len(output_types)} output types but "
                    f"{len(output_is_list)} output list flags."
                ),
            )
        )

    outputs: list[dict[str, Any]] = []
    for index, raw_type in enumerate(output_types):
        output_type = _type_label(raw_type)
        output_name = (
            _type_label(output_names[index])
            if output_names is not None and index < len(output_names)
            else output_type
        )
        is_list = bool(output_is_list[index]) if output_is_list is not None and index < len(output_is_list) else False
        outputs.append(
            {
                "index": index,
                "type": output_type,
                "name": output_name,
                "is_list": is_list,
            }
        )
    return outputs, errors


def _normalize_definition(
    class_type: str,
    source: str,
    info: Mapping[str, Any],
    *,
    cache: Mapping[str, bool],
    max_options_per_input: int,
) -> dict[str, Any]:
    inputs, input_signature, input_errors = _normalize_inputs(
        class_type,
        info.get("input"),
        max_options_per_input=max_options_per_input,
    )
    outputs, output_errors = _normalize_outputs(class_type, info)
    input_is_list = bool(info.get("is_input_list", False))
    errors = [*input_errors, *output_errors]
    return {
        "ok": not errors,
        "missing": False,
        "class_type": class_type,
        "source": source,
        "input_is_list": input_is_list,
        "inputs": inputs,
        "outputs": outputs,
        "cache": dict(cache),
        "signature": {
            "input_is_list": input_is_list,
            "inputs": input_signature,
            "outputs": outputs,
            "cache": dict(cache),
        },
        "errors": errors,
    }


def build_node_definitions_payload(
    class_types: object,
    *,
    node_class_mappings: Mapping[str, Any] | None = None,
    max_class_types: int = DEFAULT_MAX_CLASS_TYPES,
    max_options_per_input: int = DEFAULT_MAX_OPTIONS_PER_INPUT,
) -> dict[str, Any]:
    """Inspect a bounded batch of installed ComfyUI node classes.

    The payload contains every declared input, but only concrete COMBO options.
    Dynamic inputs are explicitly marked so callers do not mistake them for empty
    registries. Definition failures are isolated to their class entry.
    """

    class_limit = _validate_positive_limit(max_class_types, "max_class_types")
    option_limit = _validate_positive_limit(max_options_per_input, "max_options_per_input")
    normalized_types, requested_count = _normalize_class_types(class_types, class_limit)
    mappings = node_class_mappings if node_class_mappings is not None else _default_node_class_mappings()
    if not isinstance(mappings, Mapping):
        raise NodeDefinitionRequestError(
            "invalid_node_class_mappings",
            "node_class_mappings must be a mapping.",
        )

    definitions: dict[str, dict[str, Any]] = {}
    for class_type in normalized_types:
        node_class = mappings.get(class_type)
        if node_class is None:
            definitions[class_type] = _failed_definition(
                class_type,
                code="node_class_missing",
                message=f"Node class {class_type!r} is not installed.",
                missing=True,
            )
            continue
        try:
            source, info = _read_node_info(class_type, node_class)
            definitions[class_type] = _normalize_definition(
                class_type,
                source,
                info,
                cache=_cache_contract(node_class),
                max_options_per_input=option_limit,
            )
        except Exception as exc:
            definitions[class_type] = _failed_definition(
                class_type,
                code="definition_error",
                message=f"Could not inspect node class {class_type!r}: {exc}",
            )

    return {
        "ok": all(definition["ok"] for definition in definitions.values()),
        "requested_count": requested_count,
        "definition_count": len(definitions),
        "definitions": definitions,
    }
