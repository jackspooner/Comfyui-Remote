from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import importlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any
import urllib.parse
import uuid
import subprocess

from .cutlery_interrupt import (
    read_response_bytes_async,
    request_bytes as interruptible_request_bytes,
    request_bytes_uninterruptible,
    throw_if_interrupted,
)
from .cutlery_config import REMOTE_SERVER_ENV, data_path, strict_bool
from .cutlery_features import feature_disabled_response
from .cutlery_lora_chain import (
    CUTLERY_LORA_CHAIN_PORT_TYPE,
    materialize_lora_chain_names,
)
from .cutlery_remote.auth import build_auth_headers, configured_remote_token, is_authorized
from .cutlery_remote.blobs import BlobStore, sha256_bytes
from .cutlery_remote.boundary_types import (
    BOUNDARY_PORT_TYPE_ALIASES as REMOTE_PORT_TYPE_ALIASES,
    REMOTE_INBOUND_DISALLOWED_TYPES,
    REMOTE_OUTBOUND_DISALLOWED_TYPES,
    SUPPORTED_BOUNDARY_PORT_TYPES as REMOTE_BOUNDARY_PORT_TYPES,
)
from .cutlery_remote.capabilities import (
    REMOTE_MODEL_PRELOAD_FEATURE,
    REMOTE_PROGRESS_FEATURE,
    REMOTE_RUNTIME_OBJECT_RELOCATION_FEATURE,
    build_capabilities_payload,
    required_features_for_boundary_ports,
    required_features_for_workflow,
    required_serializers_for_boundary_ports,
    validate_remote_group_capabilities,
)
from .cutlery_remote.inventory import CANONICAL_MODEL_TYPES, find_local_model_by_filename, local_model_inventory, resolve_model_name
from .cutlery_remote.local_worker import lease_remote_target
from .cutlery_remote.group_compiler import compile_editor_remote_groups_detailed, editor_remote_group_targets
from .cutlery_remote.lora_materialization import materialize_remote_lora_file
from .cutlery_remote.model_inputs import iter_loader_model_inputs
from .cutlery_remote.model_preparation import (
    LocalModelDigestCache,
    ModelTransferCoordinator,
    local_model_file,
    model_identity_from_mapping,
    prepare_models_for_target,
)
from .cutlery_remote.model_transfer import copy_model_file_to_remote
from .cutlery_remote.node_definitions import NodeDefinitionRequestError, build_node_definitions_payload
from .cutlery_remote.dotenv import env_value
from .cutlery_remote.progress import ProgressMirror, parse_progress_mapping
from .cutlery_remote.remote_job import RemoteExecutionJob
from .cutlery_remote.registry_proxy import (
    RegistryProxyRequestError,
    prepare_registry_operation,
)
from .cutlery_remote.serialization import VALUE_BUNDLE_SCHEMA, decode_value_bundle, encode_value_bundle
from .cutlery_remote.target import (
    TrustedRemoteTarget,
    normalize_remote_base_url,
    parse_remote_target,
    resolve_trusted_remote_target,
)
from .cutlery_remote.trellis_progress import TrellisTqdmProgress

try:
    import aiohttp
    from aiohttp import web
    from server import PromptServer
except Exception:
    aiohttp = None
    web = None
    PromptServer = None


LOGGER = logging.getLogger("cutlery.remote.routes")
CATEGORY = "Cutlery/Remote"
MAX_REMOTE_GROUP_PORTS = 64
REMOTE_GROUP_CACHE_POLICY_REMOTE = "remote"
REMOTE_GROUP_CACHE_POLICY_SENDER_V1 = "sender-v1"
REMOTE_EARLY_MODEL_PRELOAD_ENV = "CUTLERY_REMOTE_EARLY_MODEL_PRELOAD_ENABLED"
REMOTE_EARLY_MODEL_PRELOAD_ENABLED = strict_bool(REMOTE_EARLY_MODEL_PRELOAD_ENV, True)
REMOTE_RESPONSE_LIMIT_MB_ENV = "CUTLERY_REMOTE_RESPONSE_LIMIT_MB"
DEFAULT_REMOTE_RESPONSE_LIMIT_MB = 384
REMOTE_RESPONSE_LIMIT_BYTES = DEFAULT_REMOTE_RESPONSE_LIMIT_MB * 1024 * 1024
MAX_REMOTE_MEDIA_ITEM_BYTES = 128 * 1024 * 1024
MAX_REMOTE_MEDIA_TOTAL_BYTES = 256 * 1024 * 1024
MAX_REMOTE_STREAM_MESSAGE_BYTES = (MAX_REMOTE_MEDIA_TOTAL_BYTES * 4 // 3) + (4 * 1024 * 1024)
MAX_REMOTE_MEDIA_CACHE_BYTES = 512 * 1024 * 1024
REMOTE_MEDIA_CACHE_MAX_AGE_SECONDS = 6 * 60 * 60
REMOTE_MEDIA_TEMP_MAX_AGE_SECONDS = 10 * 60
VALUE_NAMES = tuple(f"value_{index + 1}" for index in range(MAX_REMOTE_GROUP_PORTS))
REMOTE_MODEL_PLACEHOLDER = "Select a remote model"
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
REMOTE_PORT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_REMOTE_MEDIA_LOCK = threading.RLock()
_REMOTE_MEDIA_PATH_REFS: dict[Path, int] = {}
_REMOTE_MEDIA_PROMPT_PATHS: dict[str, set[Path]] = {}
_REMOTE_MODEL_DIGEST_CACHE = LocalModelDigestCache(data_path("remote_model_digests.json"))
_REMOTE_MODEL_TRANSFERS = ModelTransferCoordinator()
_REMOTE_PROGRESS_LOCK = threading.RLock()
_REMOTE_PROGRESS_CONTRIBUTIONS: dict[tuple[str, str, str], tuple[float, float]] = {}


class RemoteHttpError(RuntimeError):
    def __init__(self, message: str, *, status_code: int, payload: dict[str, Any] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload or {}


def _remote_response_limit_bytes(default: int = REMOTE_RESPONSE_LIMIT_BYTES) -> int:
    raw = env_value(REMOTE_RESPONSE_LIMIT_MB_ENV)
    if not raw:
        return default
    try:
        value_mb = int(str(raw).strip())
    except ValueError:
        return default
    if value_mb <= 0:
        return default
    return value_mb * 1024 * 1024


def default_blob_store() -> BlobStore:
    return BlobStore(data_path("remote_blobs"))


def _json_response(payload: dict[str, Any], status: int = 200):
    if web is None:
        return payload
    return web.json_response(payload, status=status)


def _remote_server_disabled_response():
    return feature_disabled_response(
        "remote_server",
        code="remote_server_disabled",
        env_var=REMOTE_SERVER_ENV,
        web_module=web,
    )


def _clean_base_url(value: object) -> str:
    text = str(value or "").strip().rstrip("/")
    if not text:
        return ""
    parsed_target = parse_remote_target(text)
    if parsed_target is not None:
        return parsed_target.base_url
    try:
        return normalize_remote_base_url(text)
    except ValueError:
        try:
            return resolve_trusted_remote_target(text).base_url
        except ValueError:
            return ""


def _request_headers(request: Any) -> dict[str, str]:
    headers = getattr(request, "headers", None)
    return dict(headers or {})


async def _request_json(request: Any) -> dict[str, Any]:
    json_fn = getattr(request, "json", None)
    if not callable(json_fn):
        return {}
    payload = json_fn()
    if inspect.isawaitable(payload):
        payload = await payload
    return payload if isinstance(payload, dict) else {}


def _query_value(request: Any, key: str, default: str = "") -> str:
    query = getattr(request, "query", None)
    if query is None:
        return default
    getter = getattr(query, "get", None)
    if callable(getter):
        return str(getter(key, default) or "")
    if isinstance(query, dict):
        return str(query.get(key, default) or "")
    return default


def _bool_query(request: Any, key: str, default: bool = False) -> bool:
    value = _query_value(request, key, "1" if default else "0").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _remote_json(
    method: str,
    base_url: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    token: str | None = None,
    timeout_seconds: float | None = None,
    on_cancel: Any | None = None,
) -> dict[str, Any]:
    trusted_target = resolve_trusted_remote_target(base_url)
    clean_base = trusted_target.base_url
    url = f"{clean_base}{path}"
    headers = {"Accept": "application/json"}
    headers.update(build_auth_headers(str(token or "")))
    data = None
    if body is not None:
        data = _json_bytes(body)
        headers["Content-Type"] = "application/json"
    with lease_remote_target(trusted_target):
        response = interruptible_request_bytes(
            method.upper(),
            url,
            body=data,
            headers=headers,
            timeout_s=timeout_seconds or 60.0,
            max_response_bytes=_remote_response_limit_bytes(),
            description=f"Cutlery Remote {method.upper()} {path}",
            logger=LOGGER,
            on_cancel=on_cancel,
        )
    if response.status >= 400:
        try:
            payload = json.loads(response.body.decode("utf-8") or "{}")
        except Exception:
            payload = {}
        message = payload.get("error") if isinstance(payload, dict) else ""
        raise RemoteHttpError(
            message or f"Remote Cutlery request failed with HTTP {response.status}.",
            status_code=response.status,
            payload=payload if isinstance(payload, dict) else {},
        )
    payload = json.loads(response.body.decode("utf-8") or "{}")
    return payload if isinstance(payload, dict) else {}


def _post_remote_json(
    base_url: str,
    path: str,
    body: dict[str, Any],
    *,
    token: str | None = None,
    timeout_seconds: float | None = None,
    on_cancel: Any | None = None,
) -> dict[str, Any]:
    return _remote_json(
        "POST",
        base_url,
        path,
        body=body,
        token=token,
        timeout_seconds=timeout_seconds,
        on_cancel=on_cancel,
    )


def _get_remote_json(base_url: str, path: str, *, token: str | None = None, timeout_seconds: float | None = None) -> dict[str, Any]:
    return _remote_json("GET", base_url, path, token=token, timeout_seconds=timeout_seconds)


async def _post_remote_json_async(
    base_url: str,
    path: str,
    body: dict[str, Any],
    *,
    token: str | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    if aiohttp is None or not hasattr(aiohttp, "ClientSession"):
        raise RuntimeError("aiohttp client support is unavailable.")
    trusted_target = resolve_trusted_remote_target(base_url)
    url = f"{trusted_target.base_url}{path}"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    headers.update(build_auth_headers(str(token or "")))
    timeout = aiohttp.ClientTimeout(total=max(0.1, float(timeout_seconds)))
    worker_lease = await asyncio.to_thread(lease_remote_target, trusted_target)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=body, headers=headers) as response:
                try:
                    raw = await read_response_bytes_async(
                        response,
                        max_response_bytes=_remote_response_limit_bytes(),
                    )
                    payload = json.loads(raw.decode("utf-8") or "{}")
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"Remote Cutlery request returned invalid JSON with HTTP {response.status}."
                    ) from exc
                if response.status >= 400:
                    message = payload.get("error") if isinstance(payload, dict) else ""
                    raise RemoteHttpError(
                        message or f"Remote Cutlery request failed with HTTP {response.status}.",
                        status_code=response.status,
                        payload=payload if isinstance(payload, dict) else {},
                    )
                return payload if isinstance(payload, dict) else {}
    finally:
        worker_lease.release()


def _interrupt_remote_prompt_best_effort(base_url: str, prompt_id: str, *, token: str | None) -> None:
    clean_prompt_id = str(prompt_id or "").strip()
    if not clean_prompt_id:
        return
    try:
        trusted_target = resolve_trusted_remote_target(base_url)
        path = f"/cutlery/remote/group/{urllib.parse.quote(clean_prompt_id, safe='')}/interrupt"
        response = request_bytes_uninterruptible(
            "POST",
            f"{trusted_target.base_url}{path}",
            body=b"{}",
            headers={"Accept": "application/json", "Content-Type": "application/json", **build_auth_headers(str(token or ""))},
            timeout_s=10.0,
            max_response_bytes=_remote_response_limit_bytes(),
        )
        if response.status >= 400:
            LOGGER.warning(
                "[Cutlery Remote] Prompt-specific interrupt failed target=%s prompt_id=%s status=%s",
                trusted_target.display_label,
                clean_prompt_id,
                response.status,
            )
    except Exception:
        LOGGER.warning(
            "[Cutlery Remote] Prompt-specific interrupt request failed target=%s prompt_id=%s",
            base_url,
            clean_prompt_id,
            exc_info=True,
        )


def _compact_log_json(payload: Any) -> str:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return repr(payload)


def _stream_error_message(error_data: Any) -> str:
    if not isinstance(error_data, dict):
        return str(error_data or "Remote streamed execution failed.")
    direct = str(error_data.get("error") or "").strip()
    if direct:
        return direct
    status = error_data.get("status")
    messages = status.get("messages") if isinstance(status, dict) else None
    if isinstance(messages, list):
        for message in reversed(messages):
            if (
                isinstance(message, (list, tuple))
                and len(message) == 2
                and message[0] == "execution_error"
                and isinstance(message[1], dict)
            ):
                details = message[1]
                exception_type = str(details.get("exception_type") or "").strip()
                exception_message = str(details.get("exception_message") or "").strip()
                if exception_type and exception_message:
                    return f"{exception_type}: {exception_message}"
                if exception_message or exception_type:
                    return exception_message or exception_type
    prompt_id = str(error_data.get("prompt_id") or "").strip()
    suffix = f" for peer prompt {prompt_id}" if prompt_id else ""
    return f"Remote streamed execution failed{suffix}."


def _remote_group_smoke_timeout(timeout_seconds: float) -> float:
    return max(0.1, min(15.0, float(timeout_seconds or 15.0)))


def _log_remote_group_start_and_smoke(
    base_url: str,
    workflow: Any,
    input_ports: list[dict[str, str]],
    output_ports: list[dict[str, str]],
    *,
    token: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    node_count = len(workflow) if isinstance(workflow, dict) else 0
    LOGGER.info(
        "[Cutlery Remote] Remote group detected target=%s nodes=%s inputs=%s outputs=%s",
        base_url,
        node_count,
        len(input_ports),
        len(output_ports),
    )
    try:
        payload = _get_remote_json(
            base_url,
            "/cutlery/remote/capabilities",
            token=token,
            timeout_seconds=_remote_group_smoke_timeout(timeout_seconds),
        )
    except Exception as exc:
        LOGGER.warning(
            "[Cutlery Remote] Remote smoke failed target=%s path=/cutlery/remote/capabilities error=%s",
            base_url,
            exc,
            exc_info=True,
        )
        raise RuntimeError(f"Remote group target {base_url} failed smoke check: {exc}") from exc

    required_features = required_features_for_boundary_ports(
        input_ports,
        output_ports,
    )
    required_features.update(required_features_for_workflow(workflow))
    validated = validate_remote_group_capabilities(
        payload,
        required_serializers=required_serializers_for_boundary_ports(
            input_ports,
            output_ports,
        ),
        required_features=required_features,
    )
    LOGGER.info(
        "[Cutlery Remote] Remote smoke result target=%s path=/cutlery/remote/capabilities result=%s",
        base_url,
        _compact_log_json(validated),
    )
    return validated


def _workflow_class_types(workflow: Any) -> list[str]:
    if not isinstance(workflow, dict):
        return []
    return sorted(
        {
            str(node.get("class_type") or "").strip()
            for node in workflow.values()
            if isinstance(node, dict) and str(node.get("class_type") or "").strip()
        }
    )


def _definition_inputs(definition: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(definition, dict):
        return {}
    sectioned = definition.get("inputs")
    if not isinstance(sectioned, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for section in ("required", "optional", "hidden"):
        values = sectioned.get(section)
        if not isinstance(values, dict):
            continue
        for name, input_definition in values.items():
            if isinstance(input_definition, dict):
                result[str(name)] = input_definition
    return result


def _definition_error_summary(definition: Any) -> str:
    if not isinstance(definition, dict):
        return ""
    errors = definition.get("errors")
    if not isinstance(errors, list):
        return ""
    parts = [str(error).strip() for error in errors if str(error).strip()]
    if not parts:
        return ""
    summary = "; ".join(parts[:3])
    return f": {summary[:500]}"


def _is_prompt_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 2 and isinstance(value[0], (str, int))


def _combo_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    return type(left) is type(right) and left == right


def _combo_options_contain(options: list[Any], selected_value: Any) -> bool:
    return any(_combo_values_equal(option, selected_value) for option in options)


def _preflight_remote_workflow(
    base_url: str,
    workflow: Any,
    *,
    token: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    class_types = _workflow_class_types(workflow)
    if not class_types:
        return {"ok": True, "schema_version": 1, "nodes": {}}
    remote_payload = _post_remote_json(
        base_url,
        "/cutlery/remote/node-definitions",
        {"class_types": class_types},
        token=token,
        timeout_seconds=_remote_group_smoke_timeout(timeout_seconds),
    )
    remote_nodes = remote_payload.get("nodes")
    if remote_payload.get("ok") is not True or not isinstance(remote_nodes, dict):
        raise RuntimeError("Remote node-definition preflight returned an invalid response.")

    local_payload = build_node_definitions_payload(class_types)
    local_definitions = local_payload.get("definitions")
    if not isinstance(local_definitions, dict):
        raise RuntimeError("Local node-definition preflight returned an invalid response.")

    errors: list[str] = []
    for class_type in class_types:
        remote_definition = remote_nodes.get(class_type)
        local_definition = local_definitions.get(class_type)
        if not isinstance(remote_definition, dict) or not remote_definition.get("available"):
            errors.append(f"Node class {class_type!r} is not installed on the remote target.")
            continue
        if not isinstance(local_definition, dict) or local_definition.get("missing"):
            errors.append(f"Node class {class_type!r} is not installed locally.")
            continue
        if remote_definition.get("compatible") is not True:
            errors.append(
                f"Node class {class_type!r} could not be safely inspected on the remote target"
                f"{_definition_error_summary(remote_definition)}."
            )
            continue
        if local_definition.get("ok") is not True:
            errors.append(
                f"Node class {class_type!r} could not be safely inspected locally"
                f"{_definition_error_summary(local_definition)}."
            )
            continue
        if local_definition.get("signature") != remote_definition.get("signature"):
            errors.append(f"Node class {class_type!r} has an incompatible local/remote input or output schema.")
            continue

        remote_inputs = _definition_inputs(remote_definition)
        for node_id, prompt_node in workflow.items():
            if not isinstance(prompt_node, dict) or prompt_node.get("class_type") != class_type:
                continue
            inputs = prompt_node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            for input_name, selected_value in inputs.items():
                input_definition = remote_inputs.get(str(input_name))
                if input_definition is None:
                    errors.append(f"Remote node {node_id} ({class_type}) has no input named {input_name!r}.")
                    continue
                if input_definition.get("kind") == "error":
                    errors.append(f"Remote node {node_id} ({class_type}) input {input_name!r} could not be inspected.")
                    continue
                if input_definition.get("kind") != "combo" or _is_prompt_link(selected_value):
                    continue
                options = input_definition.get("options")
                if not isinstance(options, list):
                    errors.append(f"Remote node {node_id} ({class_type}) input {input_name!r} has invalid combo metadata.")
                    continue
                if (
                    not _combo_options_contain(options, selected_value)
                    and not input_definition.get("materializable")
                ):
                    errors.append(
                        f"Remote node {node_id} ({class_type}) input {input_name!r} does not offer "
                        f"the selected value {selected_value!r}."
                    )

    if errors:
        raise RuntimeError("Remote group compatibility preflight failed: " + " ".join(errors))
    return remote_payload


def _resolve_remote_model(
    base_url: str,
    model_type: object,
    model_name: object,
    *,
    token: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        payload = _post_remote_json(
            base_url,
            "/cutlery/remote/models/resolve",
            {"model_type": model_type, "model_name": model_name},
            token=token,
            timeout_seconds=timeout_seconds,
        )
    except RemoteHttpError as exc:
        if exc.status_code == 404:
            return {"ok": False, "model_type": model_type, "model_name": model_name, "error": str(exc)}
        raise
    return payload if isinstance(payload, dict) else {"ok": False, "error": "Remote model resolve returned an invalid response."}


def _resolve_local_model_batch(body: dict[str, Any]) -> dict[str, Any]:
    records = body.get("models")
    if not isinstance(records, list):
        raise ValueError("models must be an array.")
    if len(records) > 256:
        raise ValueError("models may contain at most 256 entries.")
    import folder_paths

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_record in enumerate(records):
        try:
            expected = model_identity_from_mapping(raw_record)
        except Exception as exc:
            raise ValueError(f"models[{index}] is invalid: {exc}") from exc
        key = expected.destination_key
        if key in seen:
            raise ValueError(
                f"models contains duplicate destination {expected.category}/{expected.canonical_name}."
            )
        seen.add(key)
        resolved = resolve_model_name(expected.category, expected.canonical_name)
        if not resolved.get("ok"):
            results.append(
                {
                    "category": expected.category,
                    "model_type": expected.category,
                    "canonical_name": expected.canonical_name,
                    "model_name": expected.canonical_name,
                    "present": False,
                }
            )
            continue
        canonical_name = str(resolved.get("model_name") or expected.canonical_name)
        path = Path(folder_paths.get_full_path_or_raise(expected.category, canonical_name))
        size, sha256 = _REMOTE_MODEL_DIGEST_CACHE.digest_for(
            path,
            check_cancelled=throw_if_interrupted,
        )
        results.append(
            {
                "category": expected.category,
                "model_type": expected.category,
                "canonical_name": canonical_name,
                "model_name": canonical_name,
                "size": size,
                "sha256": sha256,
                "present": True,
            }
        )
    return {
        "ok": True,
        "manifest_id": str(body.get("manifest_id") or ""),
        "models": results,
    }


def _remote_model_selector_inputs(workflow: Any):
    if not isinstance(workflow, dict):
        return
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or node.get("class_type") != "CutleryRemoteModelName":
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            yield str(node_id), inputs


def _ensure_remote_model_reference(
    base_url: str,
    node_id: str,
    model_type: str,
    selected_name: str,
    *,
    token: str | None,
    timeout_seconds: float,
) -> tuple[str, str]:
    resolved = _resolve_remote_model(base_url, model_type, selected_name, token=token, timeout_seconds=timeout_seconds)
    if resolved.get("ok"):
        return str(resolved.get("model_type") or model_type), str(resolved.get("model_name") or selected_name)

    local = find_local_model_by_filename(model_type, selected_name)
    if not local.get("ok"):
        raise RuntimeError(
            f"Remote model {selected_name!r} is not available on {base_url}, and Cutlery could not find a local "
            f"{model_type!r} file with the same filename to copy. {local.get('error') or ''}".strip()
        )

    remote_name = str(local.get("model_name") or selected_name)
    local_path = str(local.get("path") or "")
    target = resolve_trusted_remote_target(base_url)
    if not target.copy_host or not target.copy_root:
        raise RuntimeError(
            f"Remote model {selected_name!r} is missing on {target.display_label}, but target "
            f"{target.name!r} has no copy_host/copy_root configured in CUTLERY_DATA_DIR/config.json."
        )
    copy_model_file_to_remote(
        local_path,
        local.get("model_type") or model_type,
        remote_name,
        remote_host=target.copy_host,
        remote_root=target.copy_root,
    )
    copied = _resolve_remote_model(base_url, local.get("model_type") or model_type, remote_name, token=token, timeout_seconds=timeout_seconds)
    if not copied.get("ok"):
        raise RuntimeError(
            f"Copied local model {Path(local_path).name!r} for remote node {node_id}, but {base_url} still "
            f"does not resolve {remote_name!r}. Check remote_targets.{target.name}.copy_root and the remote "
            "ComfyUI model paths."
        )
    return str(copied.get("model_type") or local.get("model_type") or model_type), str(copied.get("model_name") or remote_name)


def _ensure_remote_lora_reference(
    base_url: str,
    node_id: str,
    selected_name: str,
    *,
    token: str | None,
    timeout_seconds: float,
) -> str:
    resolved = _resolve_remote_model(
        base_url,
        "loras",
        selected_name,
        token=token,
        timeout_seconds=timeout_seconds,
    )
    if resolved.get("ok"):
        return str(resolved.get("model_name") or selected_name)

    local = find_local_model_by_filename("loras", selected_name)
    if not local.get("ok"):
        raise RuntimeError(
            f"Remote LoRA {selected_name!r} is not available on {base_url}, and Cutlery could not find a local "
            f"'loras' file with the same filename to materialize. {local.get('error') or ''}".strip()
        )
    local_path = Path(str(local.get("path") or ""))
    source_name = str(local.get("model_name") or selected_name)
    LOGGER.info(
        "[Cutlery Remote] Materializing missing LoRA target=%s source=%s bytes=%s node=%s",
        base_url,
        source_name,
        local_path.stat().st_size,
        node_id,
    )
    materialized = materialize_remote_lora_file(
        base_url,
        local_path,
        source_name,
        auth_headers=build_auth_headers(str(token or "")),
        timeout_seconds=timeout_seconds,
        check_cancelled=throw_if_interrupted,
        max_response_bytes=_remote_response_limit_bytes(),
    )
    remote_name = str(materialized.get("name") or "").strip()
    LOGGER.info(
        "[Cutlery Remote] Materialized missing LoRA target=%s source=%s remote_name=%s sha256=%s",
        base_url,
        source_name,
        remote_name,
        materialized.get("sha256"),
    )
    return remote_name


def _model_type_candidates(model_type: Any) -> tuple[str, ...]:
    if isinstance(model_type, (list, tuple)):
        return tuple(str(item).strip() for item in model_type if str(item).strip())
    text = str(model_type or "").strip()
    return (text,) if text else ()


def _ensure_remote_model_reference_any_type(
    base_url: str,
    node_id: str,
    model_type: Any,
    selected_name: str,
    *,
    token: str | None,
    timeout_seconds: float,
) -> tuple[str, str]:
    errors: list[str] = []
    for candidate in _model_type_candidates(model_type):
        try:
            return _ensure_remote_model_reference(
                base_url,
                node_id,
                candidate,
                selected_name,
                token=token,
                timeout_seconds=timeout_seconds,
            )
        except RemoteHttpError:
            raise
        except RuntimeError as exc:
            errors.append(str(exc))
    if errors:
        raise RuntimeError(" ".join(errors))
    raise RuntimeError(f"Remote node {node_id} has no model type candidates for {selected_name!r}.")


def _ensure_remote_workflow_models(base_url: str, workflow: Any, *, token: str | None, timeout_seconds: float) -> Any:
    resolved_cache: dict[tuple[tuple[str, ...], str], tuple[str, str]] = {}

    def ensure(node_id: str, model_type: Any, selected_name: str) -> tuple[str, str]:
        key = (_model_type_candidates(model_type), selected_name)
        if key not in resolved_cache:
            resolved_cache[key] = _ensure_remote_model_reference_any_type(
                base_url,
                node_id,
                model_type,
                selected_name,
                token=token,
                timeout_seconds=timeout_seconds,
            )
        return resolved_cache[key]

    for node_id, inputs in _remote_model_selector_inputs(workflow):
        model_type = str(inputs.get("model_type") or "").strip()
        selected_name = str(inputs.get("model_name") or "").strip()
        if not selected_name or selected_name == REMOTE_MODEL_PLACEHOLDER:
            raise ValueError(f"Cutlery Remote Model Name node {node_id} needs a selected model_name.")
        resolved_type, resolved_name = ensure(node_id, model_type, selected_name)
        inputs["model_type"] = resolved_type
        inputs["model_name"] = resolved_name

    for ref in iter_loader_model_inputs(workflow):
        node = workflow.get(ref.node_id) if isinstance(workflow, dict) else None
        inputs = node.get("inputs") if isinstance(node, dict) else None
        if not isinstance(inputs, dict):
            continue
        _resolved_type, resolved_name = ensure(ref.node_id, ref.model_types, ref.model_name)
        inputs[ref.input_name] = resolved_name
    return workflow


def _encode_remote_group_input_values(
    base_url: str,
    input_ports: list[dict[str, str]],
    kwargs: dict[str, Any],
    *,
    token: str | None,
    timeout_seconds: float,
) -> dict[str, Any]:
    resolved_loras: dict[str, str] = {}
    values: dict[str, Any] = {}

    for port_index, port in enumerate(input_ports):
        value_key = VALUE_NAMES[port_index]
        if value_key not in kwargs:
            continue
        value = kwargs[value_key]
        if port["type"] == CUTLERY_LORA_CHAIN_PORT_TYPE:
            port_name = port["name"]
            entry_count = len(
                value.get("loras", [])
                if isinstance(value, dict) and isinstance(value.get("loras"), list)
                else []
            )
            LOGGER.info(
                "[Cutlery Remote] Preparing inbound LoRA chain target=%s port=%s entries=%s",
                base_url,
                port_name,
                entry_count,
            )

            def resolve_lora_name(selected_name: str, entry_index: int) -> str:
                if selected_name not in resolved_loras:
                    remote_name = _ensure_remote_lora_reference(
                        base_url,
                        f"LoRA chain input {port_name!r} entry {entry_index + 1}",
                        selected_name,
                        token=token,
                        timeout_seconds=timeout_seconds,
                    )
                    resolved_loras[selected_name] = remote_name
                return resolved_loras[selected_name]

            value = materialize_lora_chain_names(value, resolve_lora_name)
            LOGGER.info(
                "[Cutlery Remote] Prepared inbound LoRA chain target=%s port=%s entries=%s",
                base_url,
                port_name,
                len(value["loras"]),
            )
        values[port["name"]] = encode_value_bundle(value)
    return values


def _authorized(request: Any) -> tuple[bool, dict[str, Any] | None, int]:
    token = configured_remote_token()
    if not token:
        LOGGER.warning("[Cutlery Remote] Request rejected because CUTLERY_REMOTE_TOKEN is not configured")
        return (
            False,
            {"ok": False, "error": "Cutlery remote token is not configured on this ComfyUI instance."},
            503,
        )
    if not is_authorized(_request_headers(request), token):
        LOGGER.warning("[Cutlery Remote] Request rejected because authorization failed")
        return False, {"ok": False, "error": "Unauthorized."}, 401
    return True, None, 200


def _decode_remote_values(values: Any) -> dict[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError("values must be a JSON object keyed by workflow input name.")
    decoded: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, dict) and value.get("schema") == VALUE_BUNDLE_SCHEMA:
            decoded[str(key)] = decode_value_bundle(value, max_blob_bytes=MAX_REMOTE_MEDIA_ITEM_BYTES)
        else:
            decoded[str(key)] = value
    return decoded


class _OutboundRemoteMediaBudget:
    def __init__(self) -> None:
        self.total_bytes = 0

    def reserve(self, size: int, *, path: str) -> None:
        if size > MAX_REMOTE_MEDIA_ITEM_BYTES:
            raise ValueError(
                f"{path} is {size} bytes, exceeding the "
                f"{MAX_REMOTE_MEDIA_ITEM_BYTES}-byte outbound media item limit."
            )
        next_total = self.total_bytes + size
        if next_total > MAX_REMOTE_MEDIA_TOTAL_BYTES:
            raise ValueError(
                f"{path} would raise this response's outbound media total to "
                f"{next_total} bytes, exceeding the {MAX_REMOTE_MEDIA_TOTAL_BYTES}-byte limit."
            )
        self.total_bytes = next_total


def _encode_remote_outputs(outputs: Any) -> dict[str, Any]:
    if not isinstance(outputs, dict):
        return {}
    budget = _OutboundRemoteMediaBudget()
    prepared: dict[str, Any] = {}
    for key, record in outputs.items():
        clean_key = str(key)
        prepared[clean_key] = _materialize_output_value_for_bundle(
            record,
            media_budget=budget,
            path=f"Remote workflow output {clean_key!r}",
        )
    return {key: encode_value_bundle(value) for key, value in prepared.items()}


def _image_ref_to_tensor(value: Any, *, path: str = "Remote IMAGE output") -> Any:
    if isinstance(value, list):
        tensors = [
            _image_ref_to_tensor(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
        if not tensors:
            return []
        try:
            import torch

            return torch.cat(tensors, dim=0)
        except Exception as exc:
            raise RuntimeError(f"{path} could not be combined into an IMAGE batch: {exc}") from exc
    if not isinstance(value, dict):
        return value
    source_path = str(value.get("path") or "").strip()
    if not source_path:
        return value
    try:
        import numpy as np
        import torch
        from PIL import Image

        with Image.open(source_path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)
    except Exception as exc:
        LOGGER.warning(
            "[Cutlery Remote] Could not load remote IMAGE output path=%s",
            source_path,
            exc_info=True,
        )
        raise RuntimeError(f"{path} could not read image file {source_path!r}: {exc}") from exc


def _safe_filename(value: object, fallback: str = "media.bin") -> str:
    name = Path(str(value or "").strip()).name
    safe = SAFE_FILENAME_RE.sub("_", name).strip("._")
    return safe or fallback


def _media_ref_to_bundle_value(
    media_type: str,
    value: Any,
    *,
    media_budget: _OutboundRemoteMediaBudget,
    path: str,
) -> Any:
    if isinstance(value, list):
        return [
            _media_ref_to_bundle_value(
                media_type,
                item,
                media_budget=media_budget,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _media_ref_to_bundle_value(
                media_type,
                item,
                media_budget=media_budget,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        )
    if not isinstance(value, dict):
        return value
    source_path = str(value.get("path") or "").strip()
    if not source_path:
        return value
    try:
        source = Path(source_path)
        stat_result = source.stat()
        size = int(stat_result.st_size)
        if not source.is_file():
            raise OSError("path is not a regular file")
        media_budget.reserve(size, path=path)
        with source.open("rb") as stream:
            data = stream.read(size + 1)
        if len(data) != size:
            raise OSError(
                f"file changed while being read (expected {size} bytes, read {len(data)})"
            )
    except ValueError:
        raise
    except Exception as exc:
        LOGGER.warning(
            "[Cutlery Remote] Could not read remote %s output path=%s",
            media_type,
            source_path,
            exc_info=True,
        )
        raise RuntimeError(
            f"{path} could not read {media_type.upper()} file {source_path!r}: {exc}"
        ) from exc
    return {
        "__cutlery_remote_media__": True,
        "media_type": media_type,
        "filename": _safe_filename(value.get("filename") or source_path, f"remote.{media_type}"),
        "content_type": str(value.get("contentType") or value.get("content_type") or "application/octet-stream"),
        "data": data,
    }


def _materialize_output_value_for_bundle(
    record: Any,
    *,
    media_budget: _OutboundRemoteMediaBudget,
    path: str,
) -> Any:
    if not isinstance(record, dict) or "value" not in record:
        return record
    value = record.get("value")
    kind = str(record.get("type") or "").strip().lower()
    if kind == "image":
        return _image_ref_to_tensor(value, path=path)
    if kind in {"audio", "video"}:
        return _media_ref_to_bundle_value(
            kind,
            value,
            media_budget=media_budget,
            path=path,
        )
    return value


def _remote_media_root() -> Path:
    try:
        import folder_paths  # type: ignore

        root = Path(folder_paths.get_input_directory()) / "cutlery_remote_group"
    except Exception:
        root = Path(tempfile.gettempdir()) / "cutlery_remote_group"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _remote_media_file_matches(path: Path, *, digest: str, size: int) -> bool:
    try:
        if path.stat().st_size != size:
            return False
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest() == digest
    except OSError:
        return False


def _remove_remote_media_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        LOGGER.warning("[Cutlery Remote] Could not remove materialized media path=%s", path, exc_info=True)


def _active_remote_media_paths() -> set[Path]:
    with _REMOTE_MEDIA_LOCK:
        return {path for path, count in _REMOTE_MEDIA_PATH_REFS.items() if count > 0}


def _evict_remote_media_cache(*, required_bytes: int = 0) -> None:
    with _REMOTE_MEDIA_LOCK:
        _evict_remote_media_cache_locked(required_bytes=required_bytes)


def _evict_remote_media_cache_locked(*, required_bytes: int = 0) -> None:
    root = _remote_media_root()
    now = time.time()
    active_paths = _active_remote_media_paths()
    cached: list[tuple[Path, int, float]] = []
    total_bytes = 0

    for path in root.iterdir():
        if not path.is_file():
            continue
        try:
            stat_result = path.stat()
        except OSError:
            continue
        age = max(0.0, now - stat_result.st_mtime)
        if path.name.startswith(".cutlery-remote-media-") and path.suffix == ".part":
            if age >= REMOTE_MEDIA_TEMP_MAX_AGE_SECONDS:
                _remove_remote_media_file(path)
            continue
        size = int(stat_result.st_size)
        total_bytes += size
        if path in active_paths:
            continue
        if age >= REMOTE_MEDIA_CACHE_MAX_AGE_SECONDS:
            _remove_remote_media_file(path)
            total_bytes -= size
            continue
        cached.append((path, size, stat_result.st_mtime))

    target_bytes = max(0, MAX_REMOTE_MEDIA_CACHE_BYTES - max(0, required_bytes))
    for path, size, _mtime in sorted(cached, key=lambda record: record[2]):
        if total_bytes <= target_bytes:
            break
        _remove_remote_media_file(path)
        total_bytes -= size
    if total_bytes > target_bytes:
        raise RuntimeError(
            "Cutlery remote media cache cannot reserve "
            f"{required_bytes} bytes within its {MAX_REMOTE_MEDIA_CACHE_BYTES}-byte limit "
            "because active prompt media still owns the remaining files."
        )


def _safe_media_suffix(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if not suffix or len(suffix) > 16 or not re.fullmatch(r"\.[a-z0-9]+", suffix):
        return ".bin"
    return suffix


def _write_remote_media_file(
    value: dict[str, Any],
    *,
    retain: bool = False,
) -> Path:
    data = value.get("data")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("Remote media bundle is missing byte data.")
    payload = bytes(data)
    if len(payload) > MAX_REMOTE_MEDIA_ITEM_BYTES:
        raise ValueError(
            f"Remote media bundle is {len(payload)} bytes, exceeding the "
            f"{MAX_REMOTE_MEDIA_ITEM_BYTES}-byte media item limit."
        )

    with _REMOTE_MEDIA_LOCK:
        filename = _safe_filename(value.get("filename"), "remote-media.bin")
        digest = sha256_bytes(payload)
        root = _remote_media_root()
        path = root / f"{digest}{_safe_media_suffix(filename)}"
        if path.exists():
            if not _remote_media_file_matches(path, digest=digest, size=len(payload)):
                raise RuntimeError(
                    f"Remote media content-address collision at {path}; existing bytes do not match {digest}."
                )
            try:
                os.utime(path, None)
            except OSError:
                pass
            if retain:
                _REMOTE_MEDIA_PATH_REFS[path] = _REMOTE_MEDIA_PATH_REFS.get(path, 0) + 1
            return path

        _evict_remote_media_cache(required_bytes=len(payload))
        temp_path = root / f".cutlery-remote-media-{digest}-{uuid.uuid4().hex}.part"
        try:
            with temp_path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temp_path, path)
            except FileExistsError:
                if not _remote_media_file_matches(path, digest=digest, size=len(payload)):
                    raise RuntimeError(
                        f"Remote media content-address collision at {path}; existing bytes do not match {digest}."
                    )
            except OSError as exc:
                raise RuntimeError(f"Could not atomically promote remote media file {path}: {exc}") from exc
            if not _remote_media_file_matches(path, digest=digest, size=len(payload)):
                raise RuntimeError(f"Promoted remote media file {path} failed SHA-256 verification.")
            if retain:
                _REMOTE_MEDIA_PATH_REFS[path] = _REMOTE_MEDIA_PATH_REFS.get(path, 0) + 1
            return path
        finally:
            _remove_remote_media_file(temp_path)


def _release_remote_media_path(path: Path, *, remove_when_zero: bool = True) -> None:
    with _REMOTE_MEDIA_LOCK:
        count = _REMOTE_MEDIA_PATH_REFS.get(path, 0)
        if count <= 1:
            _REMOTE_MEDIA_PATH_REFS.pop(path, None)
            if remove_when_zero:
                _remove_remote_media_file(path)
        else:
            _REMOTE_MEDIA_PATH_REFS[path] = count - 1


def _retain_remote_media_for_prompt(prompt_id: object, path: Path) -> bool:
    clean_prompt_id = str(prompt_id or "").strip()
    if not clean_prompt_id:
        return False
    with _REMOTE_MEDIA_LOCK:
        paths = _REMOTE_MEDIA_PROMPT_PATHS.setdefault(clean_prompt_id, set())
        if path in paths:
            return True
        paths.add(path)
        _REMOTE_MEDIA_PATH_REFS[path] = _REMOTE_MEDIA_PATH_REFS.get(path, 0) + 1
    return True


def _release_remote_media_prompt(prompt_id: object) -> None:
    clean_prompt_id = str(prompt_id or "").strip()
    if not clean_prompt_id:
        return
    with _REMOTE_MEDIA_LOCK:
        paths = _REMOTE_MEDIA_PROMPT_PATHS.pop(clean_prompt_id, set())
    for path in paths:
        _release_remote_media_path(path)
    try:
        _evict_remote_media_cache()
    except Exception:
        LOGGER.warning(
            "[Cutlery Remote] Media cache eviction failed after prompt cleanup prompt_id=%s",
            clean_prompt_id,
            exc_info=True,
        )


def _remote_audio_batch(values: list[Any], *, path: str) -> Any:
    if not values:
        return values
    if not all(
        isinstance(value, dict)
        and "waveform" in value
        and "sample_rate" in value
        for value in values
    ):
        return values
    sample_rates = {value["sample_rate"] for value in values}
    if len(sample_rates) != 1:
        raise ValueError(f"{path} contains AUDIO items with different sample rates.")
    try:
        import torch

        waveform = torch.cat([value["waveform"] for value in values], dim=0)
    except Exception as exc:
        raise RuntimeError(f"{path} could not be combined into an AUDIO batch: {exc}") from exc
    return {"waveform": waveform, "sample_rate": sample_rates.pop()}


def _materialize_remote_media_value(
    value: Any,
    *,
    prompt_id: object = None,
    path: str = "Remote workflow output",
) -> Any:
    if isinstance(value, list):
        materialized = [
            _materialize_remote_media_value(
                item,
                prompt_id=prompt_id,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
        if all(
            isinstance(item, dict)
            and item.get("__cutlery_remote_media__")
            and str(item.get("media_type") or "").strip().lower() == "audio"
            for item in value
        ):
            return _remote_audio_batch(materialized, path=path)
        return materialized
    if isinstance(value, tuple):
        return tuple(
            _materialize_remote_media_value(
                item,
                prompt_id=prompt_id,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        )
    if not isinstance(value, dict) or not value.get("__cutlery_remote_media__"):
        return value

    media_type = str(value.get("media_type") or "").strip().lower()
    if media_type not in {"audio", "video"}:
        raise ValueError(f"{path} has unsupported remote media type {media_type!r}.")
    path = _write_remote_media_file(value, retain=True)
    if media_type == "video":
        try:
            latest = importlib.import_module("comfy_api.latest")
            result = latest.InputImpl.VideoFromFile(str(path))
        except Exception as exc:
            _release_remote_media_path(path)
            raise RuntimeError(f"Could not materialize remote VIDEO file {path}: {exc}") from exc
        retained_for_prompt = _retain_remote_media_for_prompt(prompt_id, path)
        _release_remote_media_path(path, remove_when_zero=retained_for_prompt)
        return result
    if media_type == "audio":
        try:
            from comfy_extras.nodes_audio import load

            waveform, sample_rate = load(str(path))
            return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        except Exception as exc:
            raise RuntimeError(f"Could not materialize remote AUDIO file {path}: {exc}") from exc
        finally:
            _release_remote_media_path(path)
    raise AssertionError("unreachable")


async def _run_remote_group_body(
    body: dict[str, Any],
    *,
    stream_trellis_progress: bool = False,
) -> tuple[dict[str, Any], int]:
    try:
        from .nodes_wf3_boundary import _run_workflow
    except Exception as exc:
        return {"ok": False, "error": f"Cutlery workflow runner is not available: {exc}"}, 503

    if stream_trellis_progress:
        with TrellisTqdmProgress(str(body["prompt_id"])):
            return await _run_workflow(dict(body))
    return await _run_workflow(dict(body))


class _RemoteProgressSocket:
    def __init__(self, websocket: Any, prompt_id: str, allowed_node_ids: set[str]):
        self.websocket = websocket
        self.prompt_id = prompt_id
        self.allowed_node_ids = frozenset(allowed_node_ids)

    async def send_json(self, message: Any) -> None:
        if not isinstance(message, dict) or message.get("type") != "progress_state":
            return
        data = message.get("data")
        if not isinstance(data, dict) or str(data.get("prompt_id") or "") != self.prompt_id:
            return
        nodes = data.get("nodes")
        if not isinstance(nodes, dict):
            return
        visible_nodes = {
            str(node_id): state
            for node_id, state in nodes.items()
            if str(node_id) in self.allowed_node_ids
        }
        if visible_nodes:
            await self.websocket.send_json(
                {"type": "progress", "data": {**data, "nodes": visible_nodes}}
            )

    async def send_bytes(self, _message: Any) -> None:
        return


async def _cancel_peer_stream_prompt(prompt_id: str) -> None:
    try:
        from .nodes_wf3_boundary import cancel_prompt

        cancel_prompt(prompt_id)
    except Exception:
        LOGGER.warning(
            "[Cutlery Remote] Could not cancel streamed peer prompt prompt_id=%s",
            prompt_id,
            exc_info=True,
        )


async def _read_remote_stream_control(websocket: Any, prompt_id: str) -> None:
    async for message in websocket:
        if aiohttp is not None and message.type == aiohttp.WSMsgType.TEXT:
            try:
                payload = json.loads(message.data)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("type") == "cancel":
                await _cancel_peer_stream_prompt(prompt_id)
                return
        if aiohttp is not None and message.type in {
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.ERROR,
        }:
            await _cancel_peer_stream_prompt(prompt_id)
            return


def _parse_ports_json(raw: object, *, field_name: str = "ports_json") -> list[dict[str, str]]:
    if isinstance(raw, str):
        try:
            records = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} must be valid JSON: {exc}") from exc
    else:
        records = raw
    if not isinstance(records, list):
        raise ValueError(f"{field_name} must be a JSON array.")
    if len(records) > MAX_REMOTE_GROUP_PORTS:
        raise ValueError(
            f"{field_name} declares {len(records)} ports; the maximum is {MAX_REMOTE_GROUP_PORTS}."
        )

    ports: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        record_path = f"{field_name}[{index}]"
        if not isinstance(record, dict):
            raise ValueError(f"{record_path} must be an object with string name and type fields.")
        name = record.get("name")
        if not isinstance(name, str) or not REMOTE_PORT_NAME_RE.fullmatch(name.strip()):
            raise ValueError(f"{record_path}.name must be a valid port identifier.")
        clean_name = name.strip()
        if clean_name in seen:
            raise ValueError(f"{field_name} duplicates port name {clean_name!r}.")
        raw_type = record.get("type")
        if not isinstance(raw_type, str) or not raw_type.strip():
            raise ValueError(f"{record_path}.type must be a non-empty string.")
        port_type = raw_type.strip().lower()
        port_type = REMOTE_PORT_TYPE_ALIASES.get(port_type, port_type)
        if port_type not in REMOTE_BOUNDARY_PORT_TYPES:
            raise ValueError(f"{record_path}.type {raw_type!r} is not a supported remote boundary type.")
        ports.append({"name": clean_name, "type": port_type})
        seen.add(clean_name)
    return ports


def _validate_remote_group_port_directions(
    input_ports: list[dict[str, str]],
    output_ports: list[dict[str, str]],
) -> None:
    for port in input_ports:
        if port["type"] in REMOTE_INBOUND_DISALLOWED_TYPES:
            raise ValueError(
                f"Input port {port['name']!r} uses VIDEO. Local-to-remote VIDEO values are not "
                "serializable by the current Cutlery remote request transport."
            )
    for port in output_ports:
        if port["type"] in REMOTE_OUTBOUND_DISALLOWED_TYPES:
            shown_type = port["type"].upper()
            if port["type"] == CUTLERY_LORA_CHAIN_PORT_TYPE:
                raise ValueError(
                    f"Output port {port['name']!r} uses {shown_type}. Remote-to-local {shown_type} "
                    "values are not supported because their LoRA names belong to the remote machine."
                )
            raise ValueError(
                f"Output port {port['name']!r} uses {shown_type}. Remote-to-local {shown_type} "
                "values are not supported by the current Cutlery remote response transport."
            )


def _workflow_boundary_ports(
    workflow: dict[str, Any],
    *,
    class_type: str,
    allow_multiple: bool = False,
) -> list[dict[str, str]] | None:
    boundary_nodes = [
        (str(node_id), node)
        for node_id, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type") == class_type
    ]
    if not boundary_nodes:
        return None
    if len(boundary_nodes) != 1 and not allow_multiple:
        raise ValueError(
            f"remote_workflow_json must contain exactly one {class_type} node; "
            f"found {len(boundary_nodes)}."
        )
    ports = []
    seen = set()
    for node_id, node in boundary_nodes:
        inputs = node.get("inputs")
        if not isinstance(inputs, dict) or "ports_json" not in inputs:
            raise ValueError(
                f"remote_workflow_json node {node_id!r} ({class_type}) must declare inputs.ports_json."
            )
        node_ports = _parse_ports_json(
            inputs["ports_json"],
            field_name=f"remote_workflow_json[{node_id!r}].inputs.ports_json",
        )
        for port in node_ports:
            if port["name"] in seen:
                raise ValueError(
                    f"remote_workflow_json {class_type} port {port['name']!r} is duplicated across boundary nodes."
                )
            seen.add(port["name"])
            ports.append(port)
    return ports


def _validate_remote_workflow_boundary_contract(
    workflow: dict[str, Any],
    input_ports: list[dict[str, str]],
    output_ports: list[dict[str, str]],
) -> None:
    workflow_inputs = _workflow_boundary_ports(
        workflow,
        class_type="CutleryWorkflowInput",
        allow_multiple=True,
    )
    workflow_outputs = _workflow_boundary_ports(
        workflow,
        class_type="CutleryWorkflowOutput",
    )
    if workflow_inputs is None and workflow_outputs is None and not input_ports and not output_ports:
        return
    if workflow_inputs is None or workflow_outputs is None:
        raise ValueError(
            "remote_workflow_json must contain one or more CutleryWorkflowInput nodes and exactly one "
            "CutleryWorkflowOutput node when the executor declares boundary ports."
        )
    if workflow_inputs != input_ports:
        raise ValueError(
            "remote_workflow_json CutleryWorkflowInput ports do not exactly match input_ports_json."
        )
    if workflow_outputs != output_ports:
        raise ValueError(
            "remote_workflow_json CutleryWorkflowOutput ports do not exactly match output_ports_json."
        )
    _validate_remote_group_port_directions(workflow_inputs, workflow_outputs)


def _decode_output_value(
    value: Any,
    *,
    prompt_id: object = None,
    path: str = "Remote workflow output",
) -> Any:
    if isinstance(value, dict) and value.get("schema") == VALUE_BUNDLE_SCHEMA:
        decoded = decode_value_bundle(value, max_blob_bytes=MAX_REMOTE_MEDIA_ITEM_BYTES)
        return _materialize_remote_media_value(decoded, prompt_id=prompt_id, path=path)
    if isinstance(value, dict) and "value" in value:
        return _materialize_remote_media_value(
            value.get("value"),
            prompt_id=prompt_id,
            path=path,
        )
    return _materialize_remote_media_value(value, prompt_id=prompt_id, path=path)


class _RemoteMediaLifecycleProvider:
    _cutlery_remote_media_lifecycle = True

    def __init__(self) -> None:
        self.cleanup_callback = _release_remote_media_prompt

    def should_cache(self, _context: Any, _value: Any = None) -> bool:
        return False

    async def on_lookup(self, _context: Any) -> None:
        return None

    async def on_store(self, _context: Any, _value: Any) -> None:
        return None

    def on_prompt_start(self, _prompt_id: str) -> None:
        try:
            _evict_remote_media_cache()
        except Exception:
            LOGGER.warning("[Cutlery Remote] Media cache eviction failed at prompt start", exc_info=True)

    def on_prompt_end(self, prompt_id: str) -> None:
        self.cleanup_callback(prompt_id)


def _register_remote_media_lifecycle_provider() -> None:
    try:
        from comfy_execution.cache_provider import _get_cache_providers, register_cache_provider

        for provider in _get_cache_providers():
            if getattr(provider, "_cutlery_remote_media_lifecycle", False):
                provider.cleanup_callback = _release_remote_media_prompt
                return
        register_cache_provider(_RemoteMediaLifecycleProvider())
    except Exception:
        LOGGER.debug(
            "[Cutlery Remote] Prompt lifecycle provider is unavailable; bounded media eviction remains active.",
            exc_info=True,
        )


def _current_execution_prompt_id() -> str:
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
    except Exception:
        return ""
    return str(getattr(context, "prompt_id", "") or "").strip()


def _current_execution_node_id() -> str:
    try:
        from comfy_execution.utils import get_executing_context

        context = get_executing_context()
    except Exception:
        return ""
    return str(getattr(context, "node_id", "") or getattr(context, "unique_id", "") or "").strip()


def _public_node_definitions_payload(payload: dict[str, Any]) -> dict[str, Any]:
    definitions = payload.get("definitions")
    nodes: dict[str, Any] = {}
    if isinstance(definitions, dict):
        for class_type, raw_definition in definitions.items():
            definition = dict(raw_definition) if isinstance(raw_definition, dict) else {}
            sectioned_inputs = definition.get("inputs") if isinstance(definition.get("inputs"), dict) else {}
            input_options: dict[str, dict[str, Any]] = {}
            for section in ("required", "optional", "hidden"):
                section_inputs = sectioned_inputs.get(section)
                if not isinstance(section_inputs, dict):
                    continue
                for input_name, input_definition in section_inputs.items():
                    if isinstance(input_definition, dict) and input_definition.get("kind") == "combo":
                        input_options[str(input_name)] = dict(input_definition)
            definition["available"] = not bool(definition.get("missing"))
            definition["compatible"] = bool(definition.get("ok"))
            definition["input_options"] = input_options
            nodes[str(class_type)] = definition
    return {
        "ok": True,
        "schema_version": 1,
        "complete": bool(payload.get("ok")),
        "requested_count": int(payload.get("requested_count") or 0),
        "definition_count": int(payload.get("definition_count") or len(nodes)),
        "nodes": nodes,
    }


async def _compile_remote_groups_request(body: dict[str, Any]) -> dict[str, Any]:
    workflow = body.get("workflow")
    prompt = body.get("prompt")
    if not isinstance(workflow, dict) or not isinstance(prompt, dict):
        raise ValueError("workflow and prompt must be JSON objects.")
    partial_execution_targets = body.get("partial_execution_targets")
    if partial_execution_targets is not None and not isinstance(partial_execution_targets, list):
        raise ValueError("partial_execution_targets must be an array when supplied.")
    targets = editor_remote_group_targets(workflow, prompt)
    if not targets:
        return {
            "ok": True,
            "prompt": prompt,
            "remaps": {},
            "targets": [],
            "relocations": [],
            "model_refs": [],
            "preparation_manifests": [],
        }
    class_types = _workflow_class_types(prompt)
    local_payload = build_node_definitions_payload(class_types)
    local_definitions = local_payload.get("definitions")
    if not isinstance(local_definitions, dict):
        raise RuntimeError("Local node-definition inspection returned an invalid response.")
    token = configured_remote_token()
    definitions: dict[tuple[str, str], dict[str, Any]] = {}
    for target in targets:
        base_url = _clean_base_url(target)
        if not base_url:
            raise ValueError(f"Remote group target {target!r} is not trusted or configured.")
        payload = await _post_remote_json_async(
            base_url,
            "/cutlery/remote/node-definitions",
            {"class_types": class_types},
            token=token,
            timeout_seconds=30.0,
        )
        remote_nodes = payload.get("nodes")
        if payload.get("ok") is not True or not isinstance(remote_nodes, dict):
            raise RuntimeError(f"Remote target {target!r} returned invalid node definitions.")
        for class_type in class_types:
            definitions[(target, class_type)] = {
                "local": local_definitions.get(class_type, {}),
                "remote": remote_nodes.get(class_type, {}),
            }
    def infer_model_refs(remote_prompt: dict[str, Any]) -> list[dict[str, Any]]:
        inferred = []
        for node_id, node in remote_prompt.items():
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type") or "")
            for input_name, value in (node.get("inputs") or {}).items():
                if not isinstance(value, str) or not value.strip():
                    continue
                matches = []
                for category in CANONICAL_MODEL_TYPES:
                    resolved = find_local_model_by_filename(category, value)
                    if resolved.get("ok"):
                        matches.append((category, resolved))
                unique = {(category, str(match.get("model_name"))) for category, match in matches}
                if len(unique) > 1:
                    choices = ", ".join(f"{category}/{name}" for category, name in sorted(unique))
                    raise ValueError(
                        f"Unregistered model-like input {node_id}.{input_name}={value!r} is ambiguous: {choices}."
                    )
                if len(unique) == 1:
                    category, canonical_name = next(iter(unique))
                    inferred.append(
                        {
                            "node_id": str(node_id),
                            "class_type": class_type,
                            "input_name": str(input_name),
                            "model_types": [category],
                            "model_name": canonical_name,
                            "inferred": True,
                        }
                    )
        return inferred

    result = compile_editor_remote_groups_detailed(
        workflow,
        prompt,
        definition_resolver=definitions,
        model_ref_resolver=infer_model_refs,
        partial_execution_targets=partial_execution_targets,
    )
    return {
        "ok": True,
        "prompt": result.compiled,
        "remaps": result.remaps,
        "targets": result.targets,
        "relocations": result.relocations,
        "model_refs": result.model_refs,
        "preparation_manifests": result.preparation_manifests,
    }


def register_remote_routes() -> None:
    if PromptServer is None or web is None:
        return
    routes = PromptServer.instance.routes
    if getattr(routes, "_cutlery_remote_routes_registered", False):
        return

    @routes.post("/cutlery/remote/compile")
    async def cutlery_remote_compile(request):
        try:
            result = await _compile_remote_groups_request(await _request_json(request))
        except Exception as exc:
            LOGGER.warning("[Cutlery Remote] Canonical remote-group compilation failed", exc_info=True)
            return _json_response({"ok": False, "error": str(exc)}, status=400)
        return _json_response(result)

    @routes.get("/cutlery/remote/capabilities")
    async def cutlery_remote_capabilities(request):
        disabled = _remote_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        return _json_response(build_capabilities_payload())

    @routes.post("/cutlery/remote/node-definitions")
    async def cutlery_remote_node_definitions(request):
        disabled = _remote_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        body = await _request_json(request)
        try:
            result = await asyncio.to_thread(build_node_definitions_payload, body.get("class_types"))
        except NodeDefinitionRequestError as exc:
            return _json_response({"ok": False, "code": exc.code, "error": exc.message}, status=400)
        except Exception as exc:
            LOGGER.exception("[Cutlery Remote] Node-definition inspection failed")
            return _json_response({"ok": False, "error": str(exc)}, status=500)
        return _json_response(_public_node_definitions_payload(result))

    @routes.post("/cutlery/remote/proxy/node-definitions")
    async def cutlery_remote_proxy_node_definitions(request):
        body = await _request_json(request)
        try:
            target = resolve_trusted_remote_target(body.get("target"))
        except ValueError as exc:
            return _json_response({"ok": False, "error": str(exc)}, status=403)
        token = configured_remote_token()
        if not token:
            return _json_response(
                {"ok": False, "error": "Cutlery remote token is not configured on this ComfyUI instance."},
                status=503,
            )
        try:
            result = await asyncio.to_thread(
                _post_remote_json,
                target.base_url,
                "/cutlery/remote/node-definitions",
                {"class_types": body.get("class_types")},
                token=token,
                timeout_seconds=30.0,
            )
        except RemoteHttpError as exc:
            status = exc.status_code if exc.status_code in {400, 401} else 502
            return _json_response({"ok": False, "error": str(exc)}, status=status)
        except Exception as exc:
            LOGGER.warning(
                "[Cutlery Remote] Node-definition proxy failed target=%s",
                target.display_label,
                exc_info=True,
            )
            return _json_response({"ok": False, "error": str(exc)}, status=502)
        return _json_response(result)

    @routes.post("/cutlery/remote/proxy/registry")
    async def cutlery_remote_proxy_registry(request):
        body = await _request_json(request)
        try:
            unknown_fields = sorted(set(body) - {"target", "registry", "payload"})
            if unknown_fields:
                raise RegistryProxyRequestError(
                    "unsupported_registry_request_fields",
                    (
                        "Remote registry proxy does not accept request fields: "
                        f"{', '.join(str(field) for field in unknown_fields)}."
                    ),
                )
            if "payload" not in body:
                raise RegistryProxyRequestError(
                    "invalid_registry_payload",
                    "payload must be supplied as a JSON object.",
                )
            registry_id, operation, registry_payload = prepare_registry_operation(
                body.get("registry"),
                body.get("payload"),
            )
        except RegistryProxyRequestError as exc:
            return _json_response(
                {"ok": False, "code": exc.code, "error": exc.message},
                status=400,
            )
        try:
            target = resolve_trusted_remote_target(body.get("target"))
        except ValueError as exc:
            return _json_response({"ok": False, "error": str(exc)}, status=403)

        token = configured_remote_token()
        if not token:
            return _json_response(
                {"ok": False, "error": "Cutlery remote token is not configured on this ComfyUI instance."},
                status=503,
            )
        try:
            if operation.method == "GET":
                result = await asyncio.to_thread(
                    _get_remote_json,
                    target.base_url,
                    operation.path,
                    token=token,
                    timeout_seconds=30.0,
                )
            else:
                result = await asyncio.to_thread(
                    _post_remote_json,
                    target.base_url,
                    operation.path,
                    registry_payload,
                    token=token,
                    timeout_seconds=30.0,
                )
        except RemoteHttpError as exc:
            status = exc.status_code if exc.status_code in {400, 401, 403, 404, 409, 422} else 502
            return _json_response(
                {
                    "ok": False,
                    "registry": registry_id,
                    "error": str(exc),
                    "upstream_status": exc.status_code,
                },
                status=status,
            )
        except Exception as exc:
            LOGGER.warning(
                "[Cutlery Remote] Registry proxy failed target=%s registry=%s",
                target.display_label,
                registry_id,
                exc_info=True,
            )
            return _json_response(
                {"ok": False, "registry": registry_id, "error": str(exc)},
                status=502,
            )
        return _json_response(
            {
                "ok": True,
                "registry": registry_id,
                "target": target.canonical,
                "payload": result,
            }
        )

    @routes.get("/cutlery/remote/models")
    async def cutlery_remote_models(request):
        model_type = _query_value(request, "model_type")
        include_hashes = _bool_query(request, "include_hashes", False)
        target = _query_value(request, "target")
        if target:
            if include_hashes:
                return _json_response(
                    {
                        "ok": False,
                        "error": (
                            "include_hashes is not available through the browser-facing remote "
                            "inventory proxy; resolve or hash a selected model through an "
                            "authenticated execution path."
                        ),
                    },
                    status=400,
                )
            token = configured_remote_token()
            if not token:
                return _json_response({"ok": False, "error": "Cutlery remote token is not configured on this ComfyUI instance."}, status=503)
            try:
                trusted_target = resolve_trusted_remote_target(target)
                path = "/cutlery/remote/models?" + urllib.parse.urlencode(
                    {
                        "model_type": model_type,
                        "include_hashes": "1" if include_hashes else "0",
                    }
                )
                payload = await asyncio.to_thread(
                    _get_remote_json,
                    trusted_target.base_url,
                    path,
                    token=token,
                    timeout_seconds=30.0,
                )
            except ValueError as exc:
                return _json_response({"ok": False, "error": str(exc)}, status=403)
            except Exception as exc:
                LOGGER.warning(
                    "[Cutlery Remote] Remote model inventory proxy failed target=%s",
                    trusted_target.display_label,
                    exc_info=True,
                )
                return _json_response({"ok": False, "error": str(exc)}, status=502)
            return _json_response(payload)

        disabled = _remote_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        try:
            inventory = await asyncio.to_thread(
                local_model_inventory,
                model_type=model_type or None,
                include_hashes=include_hashes,
            )
        except Exception as exc:
            return _json_response({"ok": False, "error": str(exc)}, status=400)
        models = inventory.get(model_type) if model_type else None
        if model_type and models is None:
            try:
                normalized_type = next(iter(inventory["records"].keys()))
                models = inventory.get(normalized_type, [])
                inventory["model_type"] = normalized_type
            except Exception:
                models = []
        inventory["models"] = models if isinstance(models, list) else []
        return _json_response(inventory)

    @routes.post("/cutlery/remote/models/resolve")
    async def cutlery_remote_models_resolve(request):
        disabled = _remote_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        body = await _request_json(request)
        try:
            result = await asyncio.to_thread(
                resolve_model_name,
                body.get("model_type"),
                body.get("model_name"),
            )
        except Exception as exc:
            return _json_response({"ok": False, "error": str(exc)}, status=400)
        return _json_response(result, status=200 if result.get("ok") else 404)

    @routes.post("/cutlery/remote/models/resolve-batch")
    async def cutlery_remote_models_resolve_batch(request):
        disabled = _remote_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        body = await _request_json(request)
        try:
            result = await asyncio.to_thread(_resolve_local_model_batch, body)
        except Exception as exc:
            return _json_response({"ok": False, "error": str(exc)}, status=400)
        return _json_response(result)

    @routes.post("/cutlery/remote/blobs/exists")
    async def cutlery_remote_blobs_exists(request):
        disabled = _remote_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        body = await _request_json(request)
        hashes = body.get("hashes")
        if not isinstance(hashes, list):
            return _json_response({"ok": False, "error": "hashes must be a list."}, status=400)
        store = default_blob_store()
        present = []
        missing = []
        for blob_hash in hashes:
            text = str(blob_hash or "").strip().lower()
            try:
                target = present if store.has_blob(text) else missing
            except ValueError:
                target = missing
            target.append(text)
        return _json_response({"ok": True, "present": present, "missing": missing})

    @routes.post("/cutlery/remote/blobs")
    async def cutlery_remote_blobs_upload(request):
        disabled = _remote_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        body = await _request_json(request)
        data_b64 = str(body.get("data_b64") or "")
        try:
            data = base64.b64decode(data_b64.encode("ascii"), validate=True)
        except Exception:
            return _json_response({"ok": False, "error": "data_b64 must be valid base64."}, status=400)
        blob = default_blob_store().put_bytes(data)
        expected_hash = str(body.get("hash") or "").strip().lower()
        if expected_hash and expected_hash != blob["hash"]:
            return _json_response({"ok": False, "error": "Uploaded blob hash does not match expected hash."}, status=400)
        return _json_response({"ok": True, "blob": blob})

    @routes.get("/cutlery/remote/group/run-stream")
    async def cutlery_remote_group_run_stream(request):
        disabled = _remote_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        websocket = web.WebSocketResponse(
            heartbeat=30.0,
            max_msg_size=MAX_REMOTE_STREAM_MESSAGE_BYTES,
        )
        await websocket.prepare(request)
        remote_prompt_id = ""
        synthetic_client_id = ""
        control_task = None
        run_task = None
        terminal_sent = False
        try:
            try:
                start = await websocket.receive_json(timeout=30.0)
            except Exception as exc:
                await websocket.send_json(
                    {"type": "error", "data": {"ok": False, "error": f"Invalid stream start message: {exc}"}}
                )
                terminal_sent = True
                return websocket
            if (
                not isinstance(start, dict)
                or start.get("type") != "start"
                or start.get("protocol_version", start.get("protocol")) != 1
            ):
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {
                            "ok": False,
                            "error": "Stream start must use type 'start' and protocol_version 1.",
                        },
                    }
                )
                terminal_sent = True
                return websocket
            remote_prompt_id = str(start.get("prompt_id") or "").strip()
            if not remote_prompt_id:
                await websocket.send_json(
                    {"type": "error", "data": {"ok": False, "error": "prompt_id is required."}}
                )
                terminal_sent = True
                return websocket
            body = {
                "prompt_id": remote_prompt_id,
                "workflow": start.get("workflow"),
                "values": _decode_remote_values(start.get("values")),
                "timeout_seconds": start.get("timeout_seconds", 300),
            }
            synthetic_client_id = f"cutlery-remote-{uuid.uuid4().hex}"
            body["client_id"] = synthetic_client_id
            server = PromptServer.instance
            server.sockets[synthetic_client_id] = _RemoteProgressSocket(
                websocket,
                remote_prompt_id,
                set(body["workflow"]),
            )
            server.sockets_metadata[synthetic_client_id] = {"feature_flags": {}}
            run_task = asyncio.create_task(_run_remote_group_body(body, stream_trellis_progress=True))
            control_task = asyncio.create_task(_read_remote_stream_control(websocket, remote_prompt_id))
            done, _pending = await asyncio.wait(
                {run_task, control_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if control_task in done and not run_task.done():
                await _cancel_peer_stream_prompt(remote_prompt_id)
            result, _result_status = await run_task
            if result.get("ok") and isinstance(result.get("outputs"), dict):
                result = {**result, "outputs": _encode_remote_outputs(result["outputs"])}
            terminal_type = "result" if result.get("ok") else "error"
            await websocket.send_json({"type": terminal_type, "data": result})
            terminal_sent = True
            return websocket
        except asyncio.CancelledError:
            if remote_prompt_id:
                await _cancel_peer_stream_prompt(remote_prompt_id)
            raise
        except Exception as exc:
            LOGGER.warning(
                "[Cutlery Remote] Streamed group execution failed prompt_id=%s",
                remote_prompt_id,
                exc_info=True,
            )
            if remote_prompt_id:
                await _cancel_peer_stream_prompt(remote_prompt_id)
            if not terminal_sent and not websocket.closed:
                await websocket.send_json({"type": "error", "data": {"ok": False, "error": str(exc)}})
            return websocket
        finally:
            if control_task is not None and not control_task.done():
                control_task.cancel()
            if run_task is not None and not run_task.done():
                run_task.cancel()
            if synthetic_client_id:
                PromptServer.instance.sockets.pop(synthetic_client_id, None)
                PromptServer.instance.sockets_metadata.pop(synthetic_client_id, None)
            if not websocket.closed:
                await websocket.close()

    @routes.post("/cutlery/remote/group/preload")
    async def cutlery_remote_group_preload(request):
        disabled = _remote_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        body = dict(await _request_json(request))
        body["values"] = {}
        result, result_status = await _run_remote_group_body(body)
        return _json_response(result, status=result_status)

    @routes.post("/cutlery/remote/group/run")
    async def cutlery_remote_group_run(request):
        disabled = _remote_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        body = dict(await _request_json(request))
        try:
            body["values"] = _decode_remote_values(body.get("values"))
        except Exception as exc:
            LOGGER.warning(
                "[Cutlery Remote] Invalid remote group input value bundle",
                exc_info=True,
            )
            return _json_response(
                {
                    "ok": False,
                    "error": f"Invalid remote group input values: {exc}",
                },
                status=400,
            )
        result, result_status = await _run_remote_group_body(body)
        if result.get("ok") and isinstance(result.get("outputs"), dict):
            try:
                encoded_outputs = _encode_remote_outputs(result.get("outputs"))
            except Exception as exc:
                prompt_id = str(result.get("prompt_id") or body.get("prompt_id") or "").strip()
                LOGGER.exception(
                    "[Cutlery Remote] Remote group output transport failed prompt_id=%s",
                    prompt_id or "<unknown>",
                )
                error_payload = {
                    "ok": False,
                    "error": f"Remote group output transport failed: {exc}",
                }
                if prompt_id:
                    error_payload["prompt_id"] = prompt_id
                return _json_response(error_payload, status=500)
            result = dict(result)
            result["outputs"] = encoded_outputs
        return _json_response(result, status=result_status)

    @routes.post("/cutlery/remote/group/{remote_prompt_id}/interrupt")
    async def cutlery_remote_group_interrupt(request):
        disabled = _remote_server_disabled_response()
        if disabled is not None:
            return disabled
        ok, payload, status = _authorized(request)
        if not ok:
            return _json_response(payload or {}, status=status)
        remote_prompt_id = getattr(getattr(request, "match_info", {}), "get", lambda _key, _default=None: _default)(
            "remote_prompt_id",
            "",
        )
        try:
            from .nodes_wf3_boundary import cancel_prompt, record_prompt_cancellation

            record_prompt_cancellation(remote_prompt_id)
            result = cancel_prompt(remote_prompt_id)
        except ValueError as exc:
            return _json_response({"ok": False, "error": str(exc)}, status=400)
        except Exception as exc:
            LOGGER.warning("[Cutlery Remote] Remote prompt cancellation failed prompt_id=%s", remote_prompt_id, exc_info=True)
            return _json_response({"ok": False, "error": str(exc), "remote_prompt_id": str(remote_prompt_id or "")}, status=500)
        result["remote_prompt_id"] = result.pop("prompt_id")
        result["cancellation_recorded"] = True
        return _json_response(result)

    setattr(routes, "_cutlery_remote_routes_registered", True)


_register_remote_media_lifecycle_provider()
register_remote_routes()


class CutleryRemoteModelName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (
                    list(CANONICAL_MODEL_TYPES),
                    {"tooltip": "ComfyUI model folder category to list on the enclosing remote group target."},
                ),
                "model_name": (
                    "STRING",
                    {"default": REMOTE_MODEL_PLACEHOLDER, "tooltip": "Model name selected from the remote group target."},
                ),
                "remote_target": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Filled by the Cutlery frontend from the enclosing remote group title, for example remote-host:8188.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("model_name",)
    FUNCTION = "select_model"
    CATEGORY = CATEGORY
    DESCRIPTION = "Select a model name from the ComfyUI instance named by the enclosing remote group."
    SEARCH_ALIASES = ["remote model name", "remote checkpoint name", "remote text encoder name"]

    def select_model(self, model_type: str, model_name: str, remote_target: str = ""):
        name = str(model_name or "").strip()
        if not name or name == REMOTE_MODEL_PLACEHOLDER:
            raise ValueError("Cutlery Remote Model Name needs a selected model_name.")
        return (name,)


class CutleryRemoteModelPreload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                name: ("*", {"forceInput": True})
                for name in VALUE_NAMES
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "preload"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Generated peer-side output that asks ComfyUI's model manager to preload relocated objects."

    def preload(self, **kwargs):
        from comfy import model_management
        from comfy.model_patcher import ModelPatcher

        patchers: list[Any] = []
        seen: set[int] = set()

        def add(candidate: Any) -> None:
            if not isinstance(candidate, ModelPatcher) or id(candidate) in seen:
                return
            seen.add(id(candidate))
            patchers.append(candidate)

        for value in kwargs.values():
            add(value)
            add(getattr(value, "patcher", None))
            get_models = getattr(value, "get_models", None)
            if callable(get_models):
                for candidate in get_models() or ():
                    add(candidate)
        if patchers:
            model_management.load_models_gpu(patchers)
        return {}


def _local_preparation_models(
    model_refs: list[dict[str, Any]],
    check_cancelled,
) -> tuple[list[Any], dict[tuple[str, str], str]]:
    models = []
    rewrites: dict[tuple[str, str], str] = {}
    seen: set[tuple[str, str]] = set()
    for ref in model_refs:
        if not isinstance(ref, dict):
            raise ValueError("model_refs_json entries must be objects.")
        model_name = str(ref.get("model_name") or "").strip()
        model_types = ref.get("model_types")
        if not model_name or not isinstance(model_types, list) or not model_types:
            raise ValueError("Every model reference needs model_name and model_types.")
        matches = []
        for model_type in model_types:
            resolved = find_local_model_by_filename(model_type, model_name)
            if resolved.get("ok"):
                matches.append(resolved)
        unique = {
            (str(match.get("model_type")), str(match.get("model_name"))): match
            for match in matches
        }
        if len(unique) != 1:
            choices = ", ".join(f"{category}/{name}" for category, name in unique) or "none"
            raise RuntimeError(
                f"Model reference {model_name!r} must resolve to exactly one local category; found {choices}."
            )
        match = next(iter(unique.values()))
        key = str(match["model_type"]), str(match["model_name"])
        rewrites[(str(ref.get("node_id") or ""), str(ref.get("input_name") or ""))] = key[1]
        if key in seen:
            continue
        seen.add(key)
        models.append(
            local_model_file(
                match["path"],
                category=key[0],
                canonical_name=key[1],
                digest_cache=_REMOTE_MODEL_DIGEST_CACHE,
                check_cancelled=check_cancelled,
            )
        )
    return models, rewrites


def _prepare_remote_models_blocking(
    base_url: str,
    model_refs: list[dict[str, Any]],
    *,
    token: str | None,
    timeout: float,
    cancelled: threading.Event,
    active_processes: set[subprocess.Popen[str]],
    process_lock: threading.Lock,
) -> dict[str, Any]:
    target = resolve_trusted_remote_target(base_url)

    def check_cancelled() -> None:
        throw_if_interrupted()
        if cancelled.is_set():
            raise asyncio.CancelledError("Remote group preparation was cancelled.")

    models, rewrites = _local_preparation_models(model_refs, check_cancelled)
    if not models:
        return {
            "manifest": {"identity": hashlib.sha256(b"[]").hexdigest(), "models": []},
            "models": [],
            "rewrites": rewrites,
        }
    if not target.copy_host or not target.copy_root:
        raise RuntimeError(
            f"Target {target.name!r} needs copy_host and copy_root before missing models can be materialised."
        )

    def process_started(process: subprocess.Popen[str]) -> None:
        with process_lock:
            active_processes.add(process)

    def process_finished(process: subprocess.Popen[str]) -> None:
        with process_lock:
            active_processes.discard(process)

    prepared = prepare_models_for_target(
        target.base_url,
        models,
        resolve_batch=lambda body: _post_remote_json(
            target.base_url,
            "/cutlery/remote/models/resolve-batch",
            body,
            token=token,
            timeout_seconds=timeout,
        ),
        transfer_coordinator=_REMOTE_MODEL_TRANSFERS,
        transfer=lambda path, category, name: copy_model_file_to_remote(
            path,
            category,
            name,
            remote_host=target.copy_host,
            remote_root=target.copy_root,
            check_cancelled=check_cancelled,
            process_started=process_started,
            process_finished=process_finished,
        ),
        check_cancelled=check_cancelled,
    )
    prepared["rewrites"] = rewrites
    return prepared


def _rewrite_prepared_model_inputs(
    workflow: dict[str, Any],
    rewrites: dict[tuple[str, str], str],
) -> dict[str, Any]:
    rewritten = json.loads(json.dumps(workflow))
    for (node_id, input_name), canonical_name in rewrites.items():
        node = rewritten.get(node_id)
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            raise RuntimeError(f"Prepared model input {node_id}.{input_name} is absent from the compiled workflow.")
        node["inputs"][input_name] = canonical_name
    return rewritten


def _model_input_rewrites_for_workflow(
    workflow: dict[str, Any],
    rewrites: dict[tuple[str, str], str],
) -> dict[tuple[str, str], str]:
    return {
        (node_id, input_name): canonical_name
        for (node_id, input_name), canonical_name in rewrites.items()
        if node_id in workflow
    }


def _progress_contribution_key(local_prompt_id: str, remote_prompt_id: str, node_id: str) -> tuple[str, str, str]:
    return local_prompt_id, remote_prompt_id, node_id


def _send_remote_progress(update, mapping, remote_prompt_id: str) -> None:
    if PromptServer is None:
        return
    identity = next(
        (item for item in mapping.values() if item.visible and item.api_node_id == update.node_id),
        None,
    )
    if identity is None:
        return
    key = _progress_contribution_key(update.prompt_id, remote_prompt_id, update.node_id)
    with _REMOTE_PROGRESS_LOCK:
        _REMOTE_PROGRESS_CONTRIBUTIONS[key] = (update.value, update.max_value)
        contributions = [
            value
            for contribution_key, value in _REMOTE_PROGRESS_CONTRIBUTIONS.items()
            if contribution_key[0] == update.prompt_id and contribution_key[2] == update.node_id
        ]
    value = sum(item[0] for item in contributions)
    max_value = sum(item[1] for item in contributions)
    try:
        from comfy_execution.progress import get_progress_state

        local_state = get_progress_state().nodes.get(update.node_id)
    except Exception:
        local_state = None
    if isinstance(local_state, dict) and float(local_state.get("max") or 0) > 0:
        value += float(local_state.get("value") or 0)
        max_value += float(local_state["max"])
    state = "finished" if value >= max_value else "running"
    PromptServer.instance.send_sync(
        "progress_state",
        {
            "prompt_id": update.prompt_id,
            "nodes": {
                update.node_id: {
                    "value": value,
                    "max": max_value,
                    "state": state,
                    "node_id": update.node_id,
                    "prompt_id": update.prompt_id,
                    "display_node_id": identity.display_node_id,
                    "parent_node_id": identity.parent_node_id,
                    "real_node_id": identity.real_node_id,
                }
            },
        },
        PromptServer.instance.client_id,
    )


def _clear_remote_progress(local_prompt_id: str, remote_prompt_id: str) -> None:
    with _REMOTE_PROGRESS_LOCK:
        keys = [
            key
            for key in _REMOTE_PROGRESS_CONTRIBUTIONS
            if key[0] == local_prompt_id and key[1] == remote_prompt_id
        ]
        for key in keys:
            del _REMOTE_PROGRESS_CONTRIBUTIONS[key]


class CutleryRemoteGroupPreparation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "remote_base_url": ("STRING", {"default": ""}),
                "remote_workflow_json": ("STRING", {"default": "{}", "multiline": True}),
                "model_refs_json": ("STRING", {"default": "[]", "multiline": True}),
                "preparation_manifest_json": ("STRING", {"default": "{}", "multiline": True}),
                "preload_workflow_json": ("STRING", {"default": "{}", "multiline": True}),
                "timeout_seconds": ("FLOAT", {"default": 300.0, "min": 0.1, "max": 86400.0}),
            }
        }

    RETURN_TYPES = ("CUTLERY_REMOTE_PREPARATION",)
    FUNCTION = "prepare"
    CATEGORY = CATEGORY
    DESCRIPTION = "Generated asynchronous preparation that stages and optionally preloads peer models."

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    async def prepare(
        self,
        remote_base_url: str,
        remote_workflow_json: str,
        model_refs_json: str,
        preparation_manifest_json: str,
        preload_workflow_json: str,
        timeout_seconds: float = 300.0,
    ):
        base_url = _clean_base_url(remote_base_url)
        if not base_url:
            raise ValueError("Cutlery remote preparation needs a valid remote_base_url.")
        workflow = json.loads(remote_workflow_json or "{}")
        model_refs = json.loads(model_refs_json or "[]")
        manifest = json.loads(preparation_manifest_json or "{}")
        preload_workflow = json.loads(preload_workflow_json or "{}")
        if not isinstance(workflow, dict) or not isinstance(preload_workflow, dict):
            raise ValueError("Remote preparation workflows must be JSON objects.")
        if not isinstance(model_refs, list) or not isinstance(manifest, dict):
            raise ValueError("Remote preparation model refs and manifest are invalid.")
        timeout = max(0.1, float(timeout_seconds))
        token = configured_remote_token()
        capabilities = _log_remote_group_start_and_smoke(
            base_url,
            workflow,
            [],
            [],
            token=token,
            timeout_seconds=timeout,
        )
        required_features = {REMOTE_RUNTIME_OBJECT_RELOCATION_FEATURE, REMOTE_PROGRESS_FEATURE}
        if REMOTE_EARLY_MODEL_PRELOAD_ENABLED and preload_workflow:
            required_features.add(REMOTE_MODEL_PRELOAD_FEATURE)
        validate_remote_group_capabilities(capabilities, required_features=required_features)
        _preflight_remote_workflow(base_url, workflow, token=token, timeout_seconds=timeout)

        cancelled = threading.Event()
        process_lock = threading.Lock()
        active_processes: set[subprocess.Popen[str]] = set()
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(
            None,
            lambda: _prepare_remote_models_blocking(
                base_url,
                model_refs,
                token=token,
                timeout=timeout,
                cancelled=cancelled,
                active_processes=active_processes,
                process_lock=process_lock,
            ),
        )
        try:
            prepared = await task
        except asyncio.CancelledError:
            cancelled.set()
            with process_lock:
                processes = tuple(active_processes)
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            raise
        actual_identity = prepared["manifest"]["identity"]
        rewrites = prepared.get("rewrites") if isinstance(prepared.get("rewrites"), dict) else {}
        prepared_workflow = _rewrite_prepared_model_inputs(workflow, rewrites)
        prepared_preload_workflow = _rewrite_prepared_model_inputs(
            preload_workflow,
            _model_input_rewrites_for_workflow(preload_workflow, rewrites),
        )
        if REMOTE_EARLY_MODEL_PRELOAD_ENABLED and prepared_preload_workflow:
            preload_id = str(uuid.uuid4())
            payload = await _post_remote_json_async(
                base_url,
                "/cutlery/remote/group/preload",
                {
                    "prompt_id": preload_id,
                    "workflow": prepared_preload_workflow,
                    "values": {},
                    "timeout_seconds": timeout,
                },
                token=token,
                timeout_seconds=timeout + 15.0,
            )
            if payload.get("ok") is not True:
                raise RuntimeError(f"Remote model preload failed: {payload.get('error') or 'unknown error'}")
        return ({
            "target": base_url,
            "manifest_identity": actual_identity,
            "compiled_manifest": manifest,
            "workflow": workflow,
            "prepared_workflow": prepared_workflow,
            "preloaded": bool(REMOTE_EARLY_MODEL_PRELOAD_ENABLED and preload_workflow),
        },)


class CutleryRemoteGroupExecutor:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            name: ("*", {"forceInput": True, "tooltip": "Value delivered to this compiled remote input port."})
            for name in VALUE_NAMES
        }
        optional.update(
            {
                "preparation": (
                    "CUTLERY_REMOTE_PREPARATION",
                    {"forceInput": True, "tooltip": "Prepared peer assets and optional model preload state for this execution."},
                ),
                "progress_map_json": (
                    "STRING",
                    {"default": "{}", "multiline": True, "tooltip": "Serialized mapping for mirroring peer progress to local nodes."},
                ),
                "model_refs_json": (
                    "STRING",
                    {"default": "[]", "multiline": True, "tooltip": "Serialized model references required by the remote workflow."},
                ),
                "preparation_manifest_json": (
                    "STRING",
                    {"default": "{}", "multiline": True, "tooltip": "Serialized manifest that identifies prepared remote assets."},
                ),
            }
        )
        return {
            "required": {
                "remote_base_url": ("STRING", {"default": "", "tooltip": "Remote ComfyUI target for this compiled group."}),
                "remote_workflow_json": (
                    "STRING",
                    {"default": "{}", "multiline": True, "tooltip": "Serialized API prompt executed on the remote target."},
                ),
                "input_ports_json": (
                    "STRING",
                    {"default": "[]", "multiline": True, "tooltip": "Serialized boundary-input contract for the remote workflow."},
                ),
                "output_ports_json": (
                    "STRING",
                    {"default": "[]", "multiline": True, "tooltip": "Serialized boundary-output contract returned from the remote workflow."},
                ),
                "timeout_seconds": (
                    "FLOAT",
                    {"default": 300.0, "min": 0.1, "max": 86400.0, "step": 1.0, "tooltip": "Maximum time to wait for peer execution."},
                ),
                "cache_policy": (
                    [REMOTE_GROUP_CACHE_POLICY_REMOTE, REMOTE_GROUP_CACHE_POLICY_SENDER_V1],
                    {"tooltip": "Generated cache policy for this compiled remote group."},
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = tuple("*" for _ in range(MAX_REMOTE_GROUP_PORTS))
    RETURN_NAMES = VALUE_NAMES
    FUNCTION = "run_remote_group"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = "Generated terminal wrapper that executes a compiled group on another Cutlery-enabled ComfyUI instance."

    @classmethod
    def IS_CHANGED(cls, cache_policy: str = REMOTE_GROUP_CACHE_POLICY_REMOTE, **_kwargs):
        if cache_policy == REMOTE_GROUP_CACHE_POLICY_SENDER_V1:
            return cache_policy
        return float("nan")

    def run_remote_group(
        self,
        remote_base_url: str,
        remote_workflow_json: str,
        input_ports_json: str,
        output_ports_json: str,
        timeout_seconds: float = 300.0,
        cache_policy: str = REMOTE_GROUP_CACHE_POLICY_REMOTE,
        **kwargs,
    ):
        execution = self._run_remote_group_async(
            remote_base_url,
            remote_workflow_json,
            input_ports_json,
            output_ports_json,
            timeout_seconds,
            cache_policy,
            **kwargs,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(execution)
        return loop.create_task(execution)

    async def _run_remote_group_async(
        self,
        remote_base_url: str,
        remote_workflow_json: str,
        input_ports_json: str,
        output_ports_json: str,
        timeout_seconds: float = 300.0,
        cache_policy: str = REMOTE_GROUP_CACHE_POLICY_REMOTE,
        **kwargs,
    ):
        if cache_policy not in {REMOTE_GROUP_CACHE_POLICY_REMOTE, REMOTE_GROUP_CACHE_POLICY_SENDER_V1}:
            raise ValueError(f"Unsupported remote group cache_policy {cache_policy!r}.")
        base_url = _clean_base_url(remote_base_url)
        if not base_url:
            raise ValueError("Cutlery Remote Group Executor needs a valid remote_base_url.")
        try:
            workflow = json.loads(str(remote_workflow_json or "{}"))
        except Exception as exc:
            raise ValueError(f"remote_workflow_json must be valid JSON: {exc}") from exc
        if not isinstance(workflow, dict):
            raise ValueError("remote_workflow_json must be a JSON object containing an API prompt.")

        input_ports = _parse_ports_json(input_ports_json, field_name="input_ports_json")
        output_ports = _parse_ports_json(output_ports_json, field_name="output_ports_json")
        _validate_remote_group_port_directions(input_ports, output_ports)
        _validate_remote_workflow_boundary_contract(workflow, input_ports, output_ports)
        local_prompt_id = _current_execution_prompt_id()
        local_node_id = _current_execution_node_id()

        timeout = max(0.1, float(timeout_seconds or 300.0))
        token = configured_remote_token()
        remote_prompt_id = str(uuid.uuid4())
        preparation = kwargs.pop("preparation", None)
        progress_map_json = str(kwargs.pop("progress_map_json", "{}") or "{}")
        kwargs.pop("model_refs_json", None)
        preparation_manifest = json.loads(str(kwargs.pop("preparation_manifest_json", "{}") or "{}"))
        progress_payload = json.loads(progress_map_json)
        progress_mapping = parse_progress_mapping(progress_payload) if progress_payload else {}
        progress_aware = bool(progress_mapping)
        if preparation is not None:
            if not isinstance(preparation, dict):
                raise ValueError("Remote preparation handle is invalid.")
            prepared_target = _clean_base_url(preparation.get("target"))
            if prepared_target != base_url:
                raise ValueError("Remote preparation handle belongs to a different target.")
            prepared_workflow = preparation.get("workflow")
            if not isinstance(prepared_workflow, dict) or prepared_workflow != workflow:
                raise ValueError("Remote preparation handle does not match this compiled workflow.")
            if preparation.get("compiled_manifest") != preparation_manifest:
                raise ValueError("Remote preparation handle has a different compiled manifest.")
            if not str(preparation.get("manifest_identity") or ""):
                raise ValueError("Remote preparation handle is missing its content manifest identity.")
            workflow = preparation.get("prepared_workflow")
            if not isinstance(workflow, dict):
                raise ValueError("Remote preparation handle is missing its canonical prepared workflow.")
        _log_remote_group_start_and_smoke(
            base_url,
            workflow,
            input_ports,
            output_ports,
            token=token,
            timeout_seconds=timeout,
        )
        if preparation is None:
            _preflight_remote_workflow(
                base_url,
                workflow,
                token=token,
                timeout_seconds=timeout,
            )
            workflow = _ensure_remote_workflow_models(base_url, workflow, token=token, timeout_seconds=timeout)
        values = _encode_remote_group_input_values(
            base_url,
            input_ports,
            kwargs,
            token=token,
            timeout_seconds=timeout,
        )
        if not progress_aware:
            interrupt_sent = False

            def interrupt_remote() -> None:
                nonlocal interrupt_sent
                if interrupt_sent:
                    return
                interrupt_sent = True
                _interrupt_remote_prompt_best_effort(base_url, remote_prompt_id, token=token)

            try:
                payload = _post_remote_json(
                    base_url,
                    "/cutlery/remote/group/run",
                    {
                        "prompt_id": remote_prompt_id,
                        "workflow": workflow,
                        "values": values,
                        "timeout_seconds": timeout,
                    },
                    token=token,
                    timeout_seconds=timeout + 15.0,
                    on_cancel=interrupt_remote,
                )
            except BaseException:
                interrupt_remote()
                raise
        else:
            payload = await self._run_streamed(
                base_url,
                workflow,
                values,
                local_prompt_id=local_prompt_id or f"prompt:{uuid.uuid4()}",
                local_node_id=local_node_id,
                remote_prompt_id=remote_prompt_id,
                progress_mapping=progress_mapping,
                token=token,
                timeout=timeout,
            )
        if not payload.get("ok"):
            raise RuntimeError(
                f"Remote group prompt {remote_prompt_id} failed: "
                f"{payload.get('error') or 'Remote group execution failed.'}"
            )

        outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
        result = []
        for index in range(MAX_REMOTE_GROUP_PORTS):
            if index < len(output_ports):
                port_name = output_ports[index]["name"]
                result.append(
                    _decode_output_value(
                        outputs.get(port_name),
                        prompt_id=local_prompt_id,
                        path=f"Remote workflow output {port_name!r}",
                    )
                )
            else:
                result.append(None)
        return tuple(result)

    async def _run_streamed(
        self,
        base_url: str,
        workflow: dict[str, Any],
        values: dict[str, Any],
        *,
        local_prompt_id: str,
        local_node_id: str,
        remote_prompt_id: str,
        progress_mapping,
        token: str | None,
        timeout: float,
    ) -> dict[str, Any]:
        if aiohttp is None:
            raise RuntimeError("aiohttp is required for remote progress streaming.")
        trusted = resolve_trusted_remote_target(base_url)
        parsed = urllib.parse.urlsplit(trusted.base_url)
        websocket_url = urllib.parse.urlunsplit(
            ("wss" if parsed.scheme == "https" else "ws", parsed.netloc, "/cutlery/remote/group/run-stream", "", "")
        )
        owner_id = f"{local_prompt_id}:{local_node_id or remote_prompt_id}"
        job = RemoteExecutionJob(owner_id, (remote_prompt_id,))
        job.start()
        mirror = ProgressMirror(
            local_prompt_id=local_prompt_id,
            remote_prompt_id=remote_prompt_id,
            mapping=progress_mapping,
            emitter=lambda update: _send_remote_progress(update, progress_mapping, remote_prompt_id),
        )
        headers = build_auth_headers(str(token or ""))
        client_timeout = aiohttp.ClientTimeout(total=timeout + 30.0)
        session = aiohttp.ClientSession(timeout=client_timeout)
        websocket = None

        async def abort(_job):
            if websocket is not None and not websocket.closed:
                await websocket.send_json({"type": "cancel", "prompt_id": remote_prompt_id})
                await websocket.close()

        job.register_abort(abort)
        job.register_peer_interrupt(
            lambda _job: asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _interrupt_remote_prompt_best_effort(base_url, remote_prompt_id, token=token),
            )
        )
        job.register_cleanup(lambda _job: session.close())
        job.register_cleanup(lambda _job: _clear_remote_progress(local_prompt_id, remote_prompt_id))
        worker_lease = await asyncio.to_thread(lease_remote_target, trusted)
        try:
            websocket = await session.ws_connect(
                websocket_url,
                headers=headers,
                heartbeat=30.0,
                max_msg_size=MAX_REMOTE_STREAM_MESSAGE_BYTES,
            )
            await websocket.send_json(
                {
                    "type": "start",
                    "protocol_version": 1,
                    "prompt_id": remote_prompt_id,
                    "workflow": workflow,
                    "values": values,
                    "timeout_seconds": timeout,
                }
            )
            async for message in websocket:
                if message.type == aiohttp.WSMsgType.TEXT:
                    envelope = json.loads(message.data)
                    message_type = envelope.get("type")
                    if message_type == "progress":
                        data = envelope.get("data")
                        if not isinstance(data, dict) or str(data.get("prompt_id") or "") != remote_prompt_id:
                            raise RuntimeError("Peer sent progress for an unknown prompt.")
                        nodes = data.get("nodes")
                        if not isinstance(nodes, dict):
                            raise RuntimeError("Peer progress_state message is missing nodes.")
                        for peer_node_id, state in nodes.items():
                            if not isinstance(state, dict):
                                raise RuntimeError("Peer progress_state contains an invalid node state.")
                            mirror.ingest(
                                {
                                    "prompt_id": remote_prompt_id,
                                    "node_id": str(peer_node_id),
                                    "value": state.get("value"),
                                    "max": state.get("max"),
                                }
                            )
                        mirror.flush()
                    elif message_type == "result":
                        payload = envelope.get("data")
                        if not isinstance(payload, dict):
                            raise RuntimeError("Peer returned an invalid streamed result.")
                        mirror.succeed()
                        await job.succeed(payload)
                        return payload
                    elif message_type == "error":
                        error_data = envelope.get("data")
                        error = RuntimeError(_stream_error_message(error_data))
                        mirror.fail()
                        await job.fail(error)
                        raise error
                    else:
                        raise RuntimeError(f"Peer sent unknown stream message type {message_type!r}.")
                elif message.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                    break
            raise RuntimeError("Remote progress stream closed before a result.")
        except asyncio.CancelledError:
            mirror.cancel()
            await job.cancel()
            raise
        except BaseException as exc:
            if not mirror.closed:
                mirror.fail()
            if not job.cleaned:
                await job.fail(exc)
            raise
        finally:
            worker_lease.release()


class CutleryRemoteGroupValueExecutor(CutleryRemoteGroupExecutor):
    OUTPUT_NODE = False
    DESCRIPTION = "Generated dependency wrapper that executes a compiled remote group with downstream local consumers."


NODE_CLASS_MAPPINGS = {
    "CutleryRemoteModelName": CutleryRemoteModelName,
    "CutleryRemoteModelPreload": CutleryRemoteModelPreload,
    "CutleryRemoteGroupPreparation": CutleryRemoteGroupPreparation,
    "CutleryRemoteGroupExecutor": CutleryRemoteGroupExecutor,
    "CutleryRemoteGroupValueExecutor": CutleryRemoteGroupValueExecutor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CutleryRemoteModelName": "Remote Model Name",
    "CutleryRemoteModelPreload": "Remote Model Preload",
    "CutleryRemoteGroupPreparation": "Remote Group Preparation",
    "CutleryRemoteGroupExecutor": "Remote Group Executor",
    "CutleryRemoteGroupValueExecutor": "Remote Group Value Executor",
}
