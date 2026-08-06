# Remote execution

Cutlery supports two related but separate ways to use another ComfyUI instance:

- **Generic Remote groups** compile a bounded section of a workflow, discover the target's actual node definitions, transfer supported boundary values, and run that section on the peer.
- **Remote CLIP** sends text-encoding work directly to a configured peer and can materialize selected text encoders, LoRAs, and Qwen input images there.

Both use `CUTLERY_REMOTE_TOKEN` for machine-to-machine bearer authentication. Generic Remote destinations are resolved through a strict target allowlist. Remote CLIP uses one explicit operator-configured base URL.

## Security model

Use a long, unique shared token on participating peers and protect traffic with TLS or a trusted private network. Never put the token in a workflow, node widget, target JSON, screenshot, or issue report.

Remote execution materializes only files the sender explicitly transfers through the authenticated, size-bounded, hash-verified endpoints. It does not download models from the internet.

For gated inbound peer routes, request handling is ordered as follows:

1. disabled role: `403` before authentication, request parsing, hashing, filesystem access, model work, queue work, or network calls;
2. enabled role with no configured token: `503`;
3. enabled role with a missing or incorrect bearer token: `401`;
4. only then parse and execute the request.

Generic Remote browser proxies do not expose the token to browser code. The local backend resolves an allowlisted target, adds the bearer header, and calls the peer. Remote CLIP's local choices and clear-proxy routes similarly keep the token server-side.

**Export Workflow (API Format)** compiles Generic Remote groups before downloading the API prompt. The exported JSON contains generated preparation and executor nodes rather than editor group metadata, so submitting it directly to `/prompt` preserves remote execution.

## Configure a generic Remote peer

On the machine that will execute incoming groups:

```dotenv
CUTLERY_REMOTE_SERVER_ENABLED=1
CUTLERY_REMOTE_TOKEN=<long-unique-shared-token>
CUTLERY_REMOTE_EARLY_MODEL_PRELOAD_ENABLED=1
```

On the machine that sends work, set the same token and register each destination in `CUTLERY_DATA_DIR/config.json`. The client does not need `CUTLERY_REMOTE_SERVER_ENABLED=1` for outbound work.

```json
{
  "remote_targets": {
    "render": {
      "base_url": "https://render.example.test:8188",
      "display_label": "Render peer"
    }
  }
}
```

`config.json` accepts only `remote_targets` at the top level. Each target accepts:

| Field | Required | Meaning |
| --- | ---: | --- |
| `base_url` | yes | `http` or `https` origin with an explicit host and port, and no user information, path, query, or fragment. |
| `display_label` | no | Human-readable label. The alias is used when omitted. |
| `copy_host` | no | SSH/SCP host alias used only for copying a locally available model missing from the peer. |
| `copy_root` | no | Peer's main ComfyUI `models` directory used with `copy_host`. |
| `worker_python` | no | Absolute Python path for a loopback-only ComfyUI worker started on first use. Requires `worker_comfy_root`. |
| `worker_comfy_root` | no | Absolute ComfyUI checkout used by the local worker. |
| `worker_idle_seconds` | no | Quiet period before an owned local worker is stopped. Defaults to 600 seconds. |
| `expose_node_prefixes` | no | Node class prefixes loaded from that target's cached catalog as local remote-proxy nodes. |

Target aliases contain letters, digits, `_`, `-`, or `.`. Callers may use `render`, `cutlery://render`, or the exact configured origin. An arbitrary origin, including an unregistered loopback port, is rejected before the token is attached.

To label a Generic Remote canvas group without changing its destination, append ` // ` and a label to its title: `127.0.0.1:8889 // Name of group` or `cutlery://render // Production renderer`. Cutlery resolves and allowlists only the portion before ` // `; the label is display-only and is never used for token routing.

Local worker launch fields are accepted only for loopback targets. Cutlery launches the worker with a separate user directory and database, keeps it alive while requests are leased, resets the idle timer after every request, and never stops an already-running process it did not start. Proxy nodes preserve the peer's input/output schema but deliberately fail if executed outside their matching `cutlery://<alias>` group; runtime objects therefore remain on the peer and only supported group-boundary values cross between instances. Cached catalogs are stored under `CUTLERY_DATA_DIR/remote-node-catalogs/<alias>.json`, so loading proxy node definitions does not start the worker.

Generic model copying is optional and is not an HTTP upload. When a compiled group selects a model the peer does not have, Cutlery can use the selected target's `copy_host` and `copy_root` with local `ssh` and `scp`, verify size and SHA-256 on the peer, and atomically promote the staged file. The destination preserves the path and filename relative to the local main models directory: for example, `models/loras/Krea2/style.safetensors` becomes `<copy_root>/loras/Krea2/style.safetensors`. Without both fields, a missing remote model fails with setup guidance. Only use this with an SSH host and filesystem root you administer.

LoRAs entering a compiled group through a `CUTLERY_LORA_CHAIN` boundary use the same authenticated, streamed, hash-verified transfer endpoint as Remote CLIP. The peer stores each LoRA at the same relative path and filename under its main `models/loras/` directory. Later executions look up that exact path and skip the upload when it exists. If the path already contains different bytes, the transfer fails instead of overwriting the model.

## Generic Remote route roles

The inbound role controlled by `CUTLERY_REMOTE_SERVER_ENABLED` covers:

| Purpose | Route |
| --- | --- |
| Protocol and serializer capabilities | `GET /cutlery/remote/capabilities` |
| Installed node definitions and combo values | `POST /cutlery/remote/node-definitions` |
| Direct local model inventory | `GET /cutlery/remote/models` without `target` |
| Exact local model resolution | `POST /cutlery/remote/models/resolve` |
| Batched model identity resolution | `POST /cutlery/remote/models/resolve-batch` |
| Blob presence and upload | `POST /cutlery/remote/blobs/exists`; `POST /cutlery/remote/blobs` |
| Compiled group execution | `POST /cutlery/remote/group/run` |
| Progress-aware group execution | `GET /cutlery/remote/group/run-stream` |
| Boundary-independent VRAM preload | `POST /cutlery/remote/group/preload` |
| Prompt-specific cancellation | `POST /cutlery/remote/group/{remote_prompt_id}/interrupt` |

These local browser-facing outbound branches remain available while the inbound role is off:

| Purpose | Route |
| --- | --- |
| Target-specific node definitions | `POST /cutlery/remote/proxy/node-definitions` |
| Allowlisted target widget registry | `POST /cutlery/remote/proxy/registry` |
| Target model inventory | `GET /cutlery/remote/models?target=<configured-target>` |
| Canonical browser compilation | `POST /cutlery/remote/compile` |

The model-inventory route is dual-purpose. With `target`, it is an outbound proxy and rejects `include_hashes=1`; without `target`, it is an authenticated, gated inventory of the current host.

See [the generic Remote OpenAPI contract](cutlery_remote_openapi.yaml) for schemas, operation IDs, errors, and side effects.

## Generic group compatibility

Remote groups may be placed on the root canvas or inside nested ComfyUI subgraphs. The compiler preserves each subgraph instance path when it maps canvas nodes to flattened API-prompt node IDs.

Before execution, the client checks the peer's protocol, required serializers, target node definitions, and prompt-specific interruption support. A group using workflow conditioning-blob adapter nodes also requires the peer capability `cutlery_tensor_tree_v2`. A `CUTLERY_LORA_CHAIN` input requires `remote_lora_chain_boundary_v1`.

Upgrade both peers together before using a newly added boundary serializer or capability. Version-1 pickle-backed workflow blobs are rejected and must be regenerated; there is no unsafe converter.

Python is the canonical group compiler for browser and WF3 execution. For an
inbound MODEL, CLIP, VAE, ControlNet, or other non-serializable runtime wire, it
relocates the complete producer closure to the peer after proving every local
and peer schema matches. OUTPUT_NODE, NOT_IDEMPOTENT, cycles, cross-target
dependencies, and runtime-object outputs are rejected. A remote-only producer
is removed from the local compiled prompt; a mixed local/remote producer is
retained locally and cloned remotely.

Generated groups with local consumers use the non-output
`CutleryRemoteGroupValueExecutor`, so native lazy switches can suppress an
unselected remote branch. Groups with no outbound values, or containing an
original output node confirmed by local and peer definitions, retain the terminal
`CutleryRemoteGroupExecutor` instead.

Preparation runs as a generated dependency-free async node. It hashes selected
single-file model assets through a path/size/mtime digest cache, performs one
peer batch check, stages only missing files through SSH/SCP, verifies size and
SHA-256, and atomically promotes them. Transfers are serialized per target and
identical concurrent requests share the owner. Directory repositories, opaque
provider caches, undeclared sidecars, and ambiguous filenames are rejected.

When `CUTLERY_REMOTE_EARLY_MODEL_PRELOAD_ENABLED=1` (the default), relocated
loader recipes that do not depend on unfinished local boundary values are
queued early on the peer. ComfyUI's normal prompt queue, cache, model manager,
and eviction policy own the loaded objects; Cutlery does not pin models or
create another VRAM cache. Setting the variable to `0` keeps early file staging
but defers VRAM loading until normal remote execution.

Generated executors use the `remote` cache policy. Stable node ids, constants,
links, canonical model names, and preload recipes allow the peer's normal
ComfyUI cache to reuse loader and patcher work. Preparation still performs one
batched availability check on a warm run, while the digest cache avoids
rehashing unchanged files and no transfer occurs.

Boundary directions are intentionally asymmetric:

| Value | Client to peer | Peer to client |
| --- | :---: | :---: |
| Image | yes | yes |
| Mask | yes | yes |
| Latent | yes | yes |
| Conditioning | yes | yes |
| String, integer, float, boolean, JSON | yes | yes |
| Audio | yes | yes |
| Video | no | yes |
| Cutlery LoRA chain | yes | no |
| MODEL, CLIP, VAE, ControlNet, other runtime objects | reconstructed from producer recipe | no |

A graphical Mask, Latent, or Conditioning output is encoded into a bounded
Cutlery tensor-tree blob on the peer and restored to its native ComfyUI type on
the client. Both peers must provide the current WF3 blob adapter nodes and the
`cutlery_tensor_tree_v2` capability.

A compiled boundary accepts at most 64 uniquely named ports in each direction. File-backed audio or video returned by a peer is limited to 128 MiB per item and 256 MiB across one response. The regenerable content-addressed blob cache lives under `CUTLERY_DATA_DIR/remote_blobs`.

Cancellation is prompt-specific. If the client workflow is interrupted or times out, it calls the peer's interrupt route with the remote prompt ID; an already-finished prompt is a safe no-op.

Newly compiled groups stream the peer's standard `progress_state` snapshots.
Peer node ids are mapped back to their original local API/display/subgraph
identity, so a remote KSampler uses the normal progress bar on the original
local node. During streamed execution, finite Trellis worker `tqdm` counters
are also adapted into those native snapshots. Helper nodes and binary previews
are not mirrored, and singular `executing` events are intentionally not proxied.

## Configure Remote CLIP

On the peer that owns the text encoders and will serve requests:

```dotenv
CUTLERY_REMOTE_CLIP_SERVER_ENABLED=1
CUTLERY_REMOTE_CLIP_MODE=remote
CUTLERY_REMOTE_TOKEN=<long-unique-shared-token>
```

On the client that contains the Remote CLIP nodes:

```dotenv
CUTLERY_REMOTE_CLIP_MODE=direct
CUTLERY_REMOTE_CLIP_BASE_URL=https://clip.example.test:8188
CUTLERY_REMOTE_TOKEN=<long-unique-shared-token>
```

Restart both ComfyUI processes. `CUTLERY_REMOTE_CLIP_BASE_URL` is explicit operator configuration and is not read from the generic `remote_targets` allowlist. A host without a scheme is treated as HTTP; configure HTTPS when traffic crosses an untrusted network.

The client role remains usable with `CUTLERY_REMOTE_CLIP_SERVER_ENABLED=0`. The `remote` mode makes `/cutlery/remote/clip/choices` read the serving host's local inventory instead of trying to call another peer; the server-enable flag remains the actual authorization boundary for inbound execution.

## Remote CLIP routes and storage

The inbound role controlled by `CUTLERY_REMOTE_CLIP_SERVER_ENABLED` covers:

| Purpose | Route |
| --- | --- |
| Local text encoder, VAE, LoRA, and loader-type inventory | `GET /cutlery/remote/clip/inventory` |
| Single and dual text encoding | `POST /cutlery/remote/clip/text-encode`; `POST /cutlery/remote/clip/dual-text-encode` |
| Qwen image-edit conditioning | `POST /cutlery/remote/clip/qwen-image-edit-plus` |
| Text encoder, LoRA, and Qwen image materialization | `POST /cutlery/remote/clip/clips/materialize`; `/loras/materialize`; `/images/materialize` |
| Cache unload | `POST /cutlery/remote/clip/unload` |

`GET /cutlery/remote/clip/choices` is an unauthenticated local browser helper. In client mode it makes an authenticated outbound inventory request; in server mode it returns lightweight local choices without hashing files.

The three clear routes have deliberate dual behavior:

- without an incoming `Authorization` header, the local client proxies cleanup to its configured peer;
- with an `Authorization` header, the current host treats the call as an authenticated local-clear request, applies the inbound feature gate, and never proxies it.

The routes are `POST /cutlery/remote/clip/clips/clear`, `/loras/clear`, and `/images/clear`. They delete only files in these scoped Cutlery cache folders. The LoRA route is retained for the legacy `cutlery_remote` cache and does not delete path-preserved LoRAs:

```text
models/text_encoders/cutlery_remote/
models/loras/cutlery_remote/
input/cutlery_remote/qwen/
```

Uploads use temporary files, validate the supplied SHA-256, and atomically finalize. A path-preserved LoRA upload reuses a same-hash destination and rejects a different-hash destination. Defaults are:

| Payload | Limit |
| --- | ---: |
| LoRA | 4096 MiB (4 GiB) |
| CLIP/text encoder | 2048 MiB |
| Qwen input image or Qwen encode request body | 2048 MiB |

The receiver enforces limits against streamed bytes and returns `413` when exceeded. Increase limits only when the network boundary, storage, and model provenance are trusted.

See [the Remote CLIP OpenAPI contract](cutlery_remote_clip_openapi.yaml) for request schemas, materialization headers, response bundles, and cleanup semantics.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `403 ..._disabled` | Enable the matching inbound role on the receiving host and restart ComfyUI. Do not enable the client role merely to send work. |
| `503` token not configured | Set the same non-empty `CUTLERY_REMOTE_TOKEN` on the local proxy/client and receiving peer, then restart. |
| `401 Unauthorized` | Check for token mismatch or a reverse proxy stripping the `Authorization` header. |
| Target is not trusted | Add the exact origin under `remote_targets`; do not work around the allowlist with a different URL spelling or tunnel port. |
| Remote node is missing or incompatible | Update the peer's custom nodes and re-run node-definition preflight. Root `ok: true` means the response envelope succeeded, not that every node is usable. |
| Missing model cannot be copied | For generic loader models, stage it manually or configure that target's `copy_host` and `copy_root` and verify local `ssh`/`scp` access. For an inbound LoRA chain, verify the peer exposes the authenticated LoRA materialization route and permits the file size. |
| Remote CLIP choices say to configure a target | On the client, set `CUTLERY_REMOTE_CLIP_MODE=direct` and a base URL. On the server, use `remote` mode. Restart after changing `.env`. |
| Upload returns `413` | Confirm the file is intended, then raise the matching limit only on the receiving host. |
| A workflow blob is rejected | Regenerate it with the current Cutlery tensor-tree codec; legacy pickle-backed blobs are intentionally unsupported. |

For endpoint-level debugging, first read `GET /cutlery/features` on the receiving host, then preflight generic peers with `/cutlery/remote/capabilities`. The feature endpoint intentionally does not reveal the token, configured targets, local paths, or the workflow-execution flag.
