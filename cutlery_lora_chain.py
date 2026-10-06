"""Portable deferred-LoRA recipe contract."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

CUTLERY_LORA_CHAIN = "CUTLERY_LORA_CHAIN"
CUTLERY_LORA_CHAIN_PORT_TYPE = "cutlery_lora_chain"


def normalize_lora_name(name: Any) -> str:
    """Return the forward-slash form used by ComfyUI model registries."""

    return str(name or "").replace("\\", "/").strip()


def clone_lora_chain(chain: Any | None) -> dict[str, Any]:
    """Copy a deferred LoRA recipe without retaining caller-owned entries."""

    if not isinstance(chain, dict) or not isinstance(chain.get("loras"), list):
        return {"loras": []}
    return {"loras": [entry.copy() for entry in chain["loras"] if isinstance(entry, dict)]}


def lora_chain_entries(chain: Any | None) -> list[dict[str, Any]]:
    """Return cloned recipe entries in application order."""

    return clone_lora_chain(chain)["loras"]


def materialize_lora_chain_names(
    chain: Any | None,
    resolve_name: Callable[[str, int], str],
) -> dict[str, Any]:
    """Rewrite effective recipe names to names owned by an execution peer."""

    prepared = clone_lora_chain(chain)
    resolved_names: dict[str, str] = {}
    for index, entry in enumerate(prepared["loras"]):
        selected_name = normalize_lora_name(entry.get("lora_name") or entry.get("name"))
        try:
            effective = float(entry.get("strength_model", 0.0)) != 0.0 or float(entry.get("strength_clip", 0.0)) != 0.0
        except (TypeError, ValueError):
            effective = True
        if not selected_name or not effective:
            continue
        if selected_name not in resolved_names:
            remote_name = str(resolve_name(selected_name, index) or "").strip()
            if not remote_name:
                raise RuntimeError(f"LoRA chain entry {index + 1} resolved to an empty remote name.")
            resolved_names[selected_name] = remote_name
        entry["lora_name"] = resolved_names[selected_name]
    return prepared
