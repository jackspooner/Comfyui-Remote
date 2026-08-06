# Troubleshooting

- `remote_server_disabled` or `remote_clip_server_disabled`: enable the matching inbound role and restart the receiving ComfyUI instance.
- `401`: check that both peers use the same non-empty token.
- Untrusted target: add the exact peer origin or alias to `CUTLERY_DATA_DIR/config.json`; do not bypass this check.
- Capability or serializer mismatch: update Cutlery Remote and `cutlery-workflow-contracts` on both peers to a compatible release.
- Hash or size failure: retry with an unchanged source file and confirm the configured transfer limit and peer storage.
