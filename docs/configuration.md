# Configuration

Copy `.env.example` into your ComfyUI environment and set values outside Git. `CUTLERY_REMOTE_SERVER_ENABLED` and `CUTLERY_REMOTE_CLIP_SERVER_ENABLED` are startup gates and remain `0` unless a peer intentionally serves that role.

Trusted outbound peers are configured in `CUTLERY_DATA_DIR/config.json`. Use exact `http://host:port` or `https://host:port` origins; do not use wildcards, a proxy URL, or a browser-provided target. Set `CUTLERY_REMOTE_TOKEN` to the same strong value on peers that authenticate each other.

Remote CLIP limits and timeouts are documented in `.env.example`. `CUTLERY_REMOTE_CLIP_RESPONSE_LIMIT_MB` caps every JSON response buffered from a Remote CLIP peer, including chunked responses that omit `Content-Length`; the client rejects a response before JSON decoding when its declared or streamed size exceeds the limit. Lower the limits when a peer has constrained storage or bandwidth.
