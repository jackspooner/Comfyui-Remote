# Security policy

Cutlery Remote is for explicitly trusted ComfyUI peers. Do not enable inbound roles on an internet-facing ComfyUI instance without TLS, a strong unique bearer token, and a restrictive network policy.

## Defaults and trust boundary

- Generic Remote and Remote CLIP server roles default to disabled.
- Generic Remote resolves a target against `CUTLERY_DATA_DIR/config.json` before it prepares an authorization header. Unknown hosts, aliases, ports, schemes, and paths are rejected. Remote CLIP uses its explicit operator-configured `CUTLERY_REMOTE_CLIP_BASE_URL`; it does not use that allowlist.
- Bearer checks use constant-time comparison. Missing and invalid tokens are rejected before request body parsing.
- Upload sizes are bounded. Hashes are verified and files are atomically promoted only after verification.
- Generic Remote and Remote CLIP clients cap both declared and streamed JSON responses before decoding them. See [remote execution](docs/remote-execution.md) for limits and configuration.

## Reporting a vulnerability

Do not open a public issue for a suspected security vulnerability. Contact the repository maintainer privately with reproduction steps, impact, and affected version.
