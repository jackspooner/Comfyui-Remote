# Upgrading and compatibility

Cutlery Remote preserves the existing serialized Remote node IDs. Install this package alongside the released `cutlery-workflow-contracts` version required by its `pyproject.toml`.

The first compatible release accepts the historical module-level boundary imports through internal compatibility bridges. New integrations must import public protocol definitions from `cutlery_workflow_contracts`; the bridges are scheduled for removal in `0.2.0`.
