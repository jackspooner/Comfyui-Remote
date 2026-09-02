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

Generated groups use the `sender-v1` cache policy only when every original and
relocated node is explicitly marked `cache.declared_inputs_only: true` by both
the local and peer node definitions. In that case unchanged inputs reuse the
local remote-group result without dispatching another remote execution job.
Queue-time compilation may still refresh peer node-definition metadata. Missing
or false cache metadata, terminal groups, output-node groups, and
partial-execution groups use the `remote` policy instead. Those groups run
preparation on every queue, so the batched peer availability check still runs
while the digest cache avoids rehashing unchanged files and no transfer occurs.

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

A compiled boundary accepts at most 64 uniquely named ports in each direction. File-backed audio or video returned by a peer is limited to 128 MiB per item and 256 MiB across one response. Before decoding generic Remote JSON, the sender rejects a declared `Content-Length` or streamed/chunked body larger than `CUTLERY_REMOTE_RESPONSE_LIMIT_MB` (384 MiB by default); this accommodates the aggregate media boundary after base64 transport. The regenerable content-addressed blob cache lives under `CUTLERY_DATA_DIR/remote_blobs`.

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

Each accepted Remote CLIP encode is queued as a normal ComfyUI prompt on the serving peer, titled **Remote CLIP Text Encode**. It therefore appears in that peer's queue and job history; the authenticated route waits for the history result and returns the serialized CONDITIONING to the calling node. A timed-out or cancelled handler cancels only its own queued or running prompt.

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
| Remote CLIP JSON response buffered by the client | 256 MiB |

The receiver enforces upload limits against streamed bytes and returns `413` when exceeded. Before decoding JSON, the client rejects Remote CLIP responses whose declared `Content-Length` or streamed body exceeds `CUTLERY_REMOTE_CLIP_RESPONSE_LIMIT_MB`; this also bounds an individual decoded value-bundle blob. Increase limits only when the network boundary, storage, and model provenance are trusted.

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

## Two-peer release gate

`tests/test_two_peer_integration.py` is an explicit, read-only-by-default integration gate for two already-configured peers. It is skipped by the normal portable suite, so a release candidate must run it separately with values supplied by the release operator rather than copied from a peer's `.env` file:

```powershell
$env:CUTLERY_REMOTE_TWO_PEER = "1"
$env:CUTLERY_REMOTE_TWO_PEER_LOCAL_URL = "http://127.0.0.1:8888"
$env:CUTLERY_REMOTE_TWO_PEER_REMOTE_URL = "http://127.0.0.1:8889"
$env:CUTLERY_REMOTE_TWO_PEER_TOKEN = "<shared peer token>"
python -m unittest tests.test_two_peer_integration -v
```

The always-on checks read each peer's feature descriptor, confirm the receiver's authenticated capability preflight, prove that missing and invalid bearer tokens both fail with `401`, verify protocol-skew rejection in the client validator, and ask the sender to reject a deliberately untrusted loopback origin before it can proxy or attach a token. They neither log the token nor write a peer's configuration, files, models, or workflow history; the reviewed execution fixtures below intentionally create transient jobs on their dedicated peer.

The portable suite keeps state-changing checks optional. A release candidate must
set `CUTLERY_REMOTE_TWO_PEER_RELEASE=1`; that mode refuses to make a network
request until every fixture below is present and structurally valid. Supply
only reviewed payloads against dedicated test data and a dedicated pending
cancellation job. The preload fixture runs twice, once for the cold path and
again for the warm path; each run must match its fixture's independently
reviewed evidence.

```powershell
$env:CUTLERY_REMOTE_TWO_PEER_RELEASE = "1"
```

| Variable | Check |
| --- | --- |
| `CUTLERY_REMOTE_TWO_PEER_GROUP_RUN_BODY` | Runs one compiled generic group request on the receiving peer. |
| `CUTLERY_REMOTE_TWO_PEER_BOUNDARY_GROUP_RUN_BODY` | Runs a compiled group with reviewed supported tensor-tree/WF3 boundary inputs and outputs, including the required output-bundle assertion. |
| `CUTLERY_REMOTE_TWO_PEER_STREAM_BODY` | Opens the authenticated progress WebSocket, submits one reviewed progress-emitting group request, and requires streamed progress plus a successful terminal result. |
| `CUTLERY_REMOTE_TWO_PEER_PRELOAD_BODY` | An envelope with `request`, `expect_cold`, and `expect_warm` objects. The peer preload route runs twice with `request`; both expected evidence objects must match their respective responses. |
| `CUTLERY_REMOTE_TWO_PEER_CANCEL_FIXTURE` | An envelope with `prompt_id` and `expect`. `expect` must require `cancellation_recorded: true` plus the reviewed queued/running termination outcome. Never point this at a user or production job. |
| `CUTLERY_REMOTE_TWO_PEER_CLIP_TEXT_ENCODE_BODY` | Runs the single-encoder Remote CLIP text-encode route and requires a conditioning bundle. |
| `CUTLERY_REMOTE_TWO_PEER_CLIP_DUAL_TEXT_ENCODE_BODY` | Runs the dual-encoder Remote CLIP route and requires a conditioning bundle. |
| `CUTLERY_REMOTE_TWO_PEER_CLIP_QWEN_IMAGE_EDIT_BODY` | Runs the reviewed Qwen image-edit Remote CLIP route and requires a conditioning bundle. |
| `CUTLERY_REMOTE_TWO_PEER_CLIP_LORA_TEXT_ENCODE_BODY` | Runs text encoding with a reviewed, already-materialized LoRA chain and requires a conditioning bundle. |
| `CUTLERY_REMOTE_TWO_PEER_LORA_MATERIALIZE_FIXTURE` | Upload fixture for a disposable `cutlery_remote/...` LoRA. Requires `path`, `name`, `sha256`, `expect_status: 200`, and matching response evidence. The gate verifies the source digest before upload. |
| `CUTLERY_REMOTE_TWO_PEER_LORA_SIZE_LIMIT_FIXTURE` | Disposable oversized upload fixture. Requires the same file fields, an expected rejection status, `error_contains`, and matching response evidence. Configure the disposable receiver's limit so it is rejected before promotion. |
| `CUTLERY_REMOTE_TWO_PEER_LORA_HASH_MISMATCH_FIXTURE` | Disposable upload fixture for a deliberate SHA-256 header mismatch. Requires the same fields, expected rejection status, `error_contains`, and matching response evidence. The gate verifies the fixture file first, then sends an intentionally wrong declared hash. |
| `CUTLERY_REMOTE_TWO_PEER_LORA_CLEANUP_FIXTURE` | An envelope with `expect`, including `deleted_count` of at least one, proving the dedicated materialized LoRA was removed after the rejection checks. |

Every `*_BODY` and `*_FIXTURE` value is a JSON object, not a path. LoRA fixture
objects refer to an external disposable file via their `path` field; keep those
files and all preparation steps outside this repository so they cannot
accidentally ship in the public archive. The gate does not read a peer's `.env`,
infer a token, or create fixture data. The LoRA lifecycle checks intentionally
write and clear only a dedicated `cutlery_remote/...` test LoRA on a disposable
peer.
