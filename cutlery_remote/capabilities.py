"""Protocol negotiation for Cutlery remote-workflow peers."""

from __future__ import annotations

from typing import Any

from .boundary_types import (
    REMOTE_LORA_CHAIN_FEATURE,
    required_boundary_features,
    required_boundary_serializers,
)

REMOTE_PROTOCOL_VERSION = 1
REMOTE_RUNTIME_OBJECT_RELOCATION_FEATURE = "remote_runtime_object_relocation_v1"
REMOTE_MODEL_PRELOAD_FEATURE = "remote_model_preload_v1"
REMOTE_PROGRESS_FEATURE = "remote_progress_v1"
WORKFLOW_BLOB_FEATURE = "cutlery_tensor_tree_v2"
WORKFLOW_BLOB_ADAPTER_NODE_TYPES = frozenset({
    "WF3LatentToBlob", "WF3LatentFromBlob", "WF3MaskToBlob", "WF3MaskFromBlob",
    "WF3ConditioningToBlob", "WF3ConditioningFromBlob",
})
SUPPORTED_SERIALIZERS = ("primitive", "tensor", "latent", "conditioning", "cutlery_lora_chain")
REQUIRED_REMOTE_GROUP_FEATURES = ("remote_groups", "remote_node_definitions_v1", "prompt_specific_interrupt")


def build_capabilities_payload() -> dict[str, Any]:
    """Return the capabilities a protocol-v1 peer must advertise."""

    return {
        "ok": True,
        "protocol_version": REMOTE_PROTOCOL_VERSION,
        "auth": {"type": "shared_token"},
        "serializers": list(SUPPORTED_SERIALIZERS),
        "features": {
            "remote_groups": True,
            "content_addressed_blobs": True,
            "remote_clip_text_encode": True,
            "remote_prompt_execution": True,
            "remote_model_inventory": True,
            "remote_node_definitions_v1": True,
            "prompt_specific_interrupt": True,
            REMOTE_RUNTIME_OBJECT_RELOCATION_FEATURE: True,
            REMOTE_MODEL_PRELOAD_FEATURE: True,
            REMOTE_PROGRESS_FEATURE: True,
            WORKFLOW_BLOB_FEATURE: True,
            REMOTE_LORA_CHAIN_FEATURE: True,
        },
    }


def required_serializers_for_boundary_ports(*port_groups: list[dict[str, str]]) -> set[str]:
    return required_boundary_serializers(*port_groups)


def required_features_for_boundary_ports(*port_groups: list[dict[str, str]]) -> set[str]:
    return required_boundary_features(*port_groups)


def required_features_for_workflow(workflow: Any) -> set[str]:
    """Return optional protocol features used by a serialized prompt graph."""

    if not isinstance(workflow, dict):
        return set()
    if any(isinstance(node, dict) and node.get("class_type") in WORKFLOW_BLOB_ADAPTER_NODE_TYPES for node in workflow.values()):
        return {WORKFLOW_BLOB_FEATURE}
    return set()


def validate_remote_group_capabilities(
    payload: Any,
    *,
    required_serializers: set[str] | None = None,
    required_features: set[str] | None = None,
) -> dict[str, Any]:
    """Reject a peer before job submission when its protocol contract differs."""

    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError("Remote capabilities response was invalid.")
    protocol_version = payload.get("protocol_version")
    if protocol_version != REMOTE_PROTOCOL_VERSION:
        raise RuntimeError(
            f"Remote protocol version {protocol_version!r} is incompatible with required version {REMOTE_PROTOCOL_VERSION}."
        )
    features = payload.get("features")
    if not isinstance(features, dict):
        raise RuntimeError("Remote capabilities response is missing features.")
    required_feature_names = set(REQUIRED_REMOTE_GROUP_FEATURES)
    required_feature_names.update(required_features or ())
    missing_features = sorted(name for name in required_feature_names if features.get(name) is not True)
    if missing_features:
        raise RuntimeError(f"Remote Cutlery is missing required features: {', '.join(missing_features)}.")
    serializers = payload.get("serializers")
    if not isinstance(serializers, list) or not all(isinstance(item, str) for item in serializers):
        raise RuntimeError("Remote capabilities response is missing serializers.")
    missing_serializers = sorted(set(required_serializers or ()) - set(serializers))
    if missing_serializers:
        raise RuntimeError("Remote Cutlery is missing required boundary serializers: " + ", ".join(missing_serializers) + ".")
    return payload
