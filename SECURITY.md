# Security policy

Cutlery Remote is for explicitly trusted ComfyUI peers. Do not enable inbound roles on an internet-facing ComfyUI instance without TLS, a strong unique bearer token, and a restrictive network policy.

## Defaults and trust boundary

- Generic Remote and Remote CLIP server roles default to disabled.
- The client resolves a target against `CUTLERY_DATA_DIR/config.json` before it prepares an authorization header. Unknown hosts, aliases, ports, schemes, and paths are rejected.
- Bearer checks use constant-time comparison. Missing and invalid tokens are rejected before request body parsing.
- Upload sizes are bounded. Hashes are verified and files are atomically promoted only after verification.

## Reporting a vulnerability

Do not open a public issue for a suspected security vulnerability. Contact the repository maintainer privately with reproduction steps, impact, and affected version.
