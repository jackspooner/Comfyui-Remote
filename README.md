# Cutlery Remote

Standalone ComfyUI nodes for trusted, peer-to-peer remote execution. It includes generic Remote groups, Remote CLIP, generated preparation/executor nodes, model materialization, progress mirroring, cancellation, and local-worker support.

## Installation

Install this repository into `ComfyUI/custom_nodes/Cutlery-Remote`. Install the released `cutlery-workflow-contracts` dependency, then restart ComfyUI. Cutlery Remote does not require the broader Cutlery Nodes package.

## Quick setup

1. Install Cutlery Remote on each peer and use compatible `cutlery-workflow-contracts` versions.
2. Set `CUTLERY_REMOTE_SERVER_ENABLED=1` only on a peer that accepts generic Remote work. Set `CUTLERY_REMOTE_CLIP_SERVER_ENABLED=1` only on a peer that accepts Remote CLIP work.
3. Configure the same bearer token on both peers and put exact trusted target origins in `CUTLERY_DATA_DIR/config.json`.
4. Use HTTPS outside a private, controlled network.

Read [remote execution](docs/remote-execution.md), [configuration](docs/configuration.md), [security](SECURITY.md), [troubleshooting](docs/troubleshooting.md), and [upgrading](docs/upgrading.md) before exposing a peer.

## Security and network behavior

Inbound roles are disabled by default. Outbound requests are restricted to explicit configured targets, and bearer tokens are attached only after that trust check. Servers perform constant-time bearer validation, bound transfers, verify file hashes, and promote received files atomically. The browser frontend is not a general-purpose origin proxy.

## Testing and release

Run `python -m unittest discover -s tests -v`. The release gate also builds a wheel, checks its exact allowlist, runs the portable suite from a clean clone, verifies two-peer behavior, and requires an independent security review before publishing `0.1.0`.
