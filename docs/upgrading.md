# Upgrading and compatibility

Cutlery Remote preserves the existing serialized Remote node IDs. Protocol definitions, capability negotiation, value serialization, and deferred LoRA helpers are included in this package; no separate contracts package is needed.

The `cutlery_remote.boundary_types`, `cutlery_remote.capabilities`, `cutlery_remote.serialization`, and `cutlery_lora_chain` module paths continue to provide the runtime contracts used by this package.

## Release history

- Self-contained contracts: protocol types, capabilities, serialization, and deferred LoRA helpers are included; the separate contracts dependency is removed.
