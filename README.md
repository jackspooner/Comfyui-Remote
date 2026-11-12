# Cutlery Remote

Cutlery Remote is a standalone ComfyUI extension for running work on trusted remote machines. Choose the mode that matches your workflow:

- **Generic Remote groups** run a bounded section of a workflow on another ComfyUI instance, transferring supported inputs and outputs and mirroring progress and cancellation.
- **Remote CLIP** sends text encoding to a peer that owns the text encoders, returning conditioning to the local workflow.

The extension also supports model materialization and managed local workers. See [remote execution](docs/remote-execution.md) for supported boundaries and compatibility limits.

## Installation

Requires ComfyUI 0.28.0 or newer and Python 3.10 or newer. From your ComfyUI directory:

```bash
git clone https://github.com/jackspooner/Comfyui-Remote.git custom_nodes/Cutlery-Remote
```

Restart ComfyUI after installation. The protocol contracts are bundled; this package declares no additional Python dependencies and does not require the broader Cutlery Nodes package. Install compatible releases and required workflow nodes on both peers.

## Quick setup

1. Install the same compatible Cutlery Remote release on each peer.
2. Set `CUTLERY_REMOTE_SERVER_ENABLED=1` only on a peer that accepts generic Remote work. Set `CUTLERY_REMOTE_CLIP_SERVER_ENABLED=1` only on a peer that accepts Remote CLIP work.
3. Set the same strong, unique `CUTLERY_REMOTE_TOKEN` on the sender and receiver.
4. For [Generic Remote](docs/remote-execution.md#configure-a-generic-remote-peer), register exact trusted target origins in `CUTLERY_DATA_DIR/config.json`. For [Remote CLIP](docs/remote-execution.md#configure-remote-clip), set the client's explicit `CUTLERY_REMOTE_CLIP_BASE_URL` and the documented client/server modes; it does not use the Generic Remote allowlist.
5. Restart after configuration changes. Use HTTPS outside a private, controlled network.

Read [remote execution](docs/remote-execution.md), [configuration](docs/configuration.md), [security](SECURITY.md), [troubleshooting](docs/troubleshooting.md), and [upgrading](docs/upgrading.md) before exposing a peer.

## Security and network behavior

Inbound roles are disabled by default. Enable only the roles you need on explicitly trusted peers. Keep tokens out of workflows and browser code; the browser frontend is not a general-purpose origin proxy. Read the [security model](docs/remote-execution.md#security-model) for authentication, bounded transfers and response limits.

## Testing and release

Run `python -m unittest discover -s tests -v`. Publication has separate [release requirements](docs/remote-execution.md#two-peer-release-gate), including explicit two-peer acceptance.

## License

[Apache-2.0](LICENSE).
