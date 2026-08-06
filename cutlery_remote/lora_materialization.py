from __future__ import annotations

import hashlib
import http.client as http_client
import json
from pathlib import Path
from typing import Callable, Mapping
import urllib.parse


DEFAULT_UPLOAD_CHUNK_SIZE = 1024 * 1024
MATERIALIZE_PATH = "/cutlery/remote/clip/loras/materialize"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DEFAULT_UPLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_remote_lora_file(
    base_url: str,
    local_path: str | Path,
    source_name: str,
    *,
    auth_headers: Mapping[str, str],
    timeout_seconds: float,
    sha256: str = "",
    chunk_size: int = DEFAULT_UPLOAD_CHUNK_SIZE,
    check_cancelled: Callable[[], None] | None = None,
    on_chunk: Callable[[int], None] | None = None,
) -> dict[str, object]:
    """Stream a local LoRA to the remote content-addressed materialization endpoint."""

    path = Path(local_path)
    expected_sha256 = str(sha256 or "").strip().lower() or _sha256_file(path)
    url = urllib.parse.urljoin(f"{str(base_url).rstrip('/')}/", MATERIALIZE_PATH.lstrip("/"))
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError(f"Remote LoRA target is not a valid HTTP(S) URL: {base_url}")

    request_target = urllib.parse.urlunparse(("", "", parsed.path or "/", parsed.params, parsed.query, ""))
    headers = {
        "Content-Type": "application/octet-stream",
        "Accept": "application/json",
        "Content-Length": str(path.stat().st_size),
        "X-Cutlery-Lora-Name": urllib.parse.quote(str(source_name), safe="/.-_"),
        "X-Cutlery-Lora-SHA256": expected_sha256,
        **{str(name): str(value) for name, value in auth_headers.items()},
    }
    connection_cls = http_client.HTTPSConnection if parsed.scheme == "https" else http_client.HTTPConnection
    connection = connection_cls(parsed.hostname, parsed.port, timeout=max(0.1, float(timeout_seconds)))
    try:
        if check_cancelled is not None:
            check_cancelled()
        connection.putrequest("POST", request_target)
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.endheaders()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(max(1, int(chunk_size))), b""):
                if check_cancelled is not None:
                    check_cancelled()
                connection.send(chunk)
                if on_chunk is not None:
                    on_chunk(len(chunk))
        if check_cancelled is not None:
            check_cancelled()
        response = connection.getresponse()
        raw = response.read().decode("utf-8", errors="replace")
        if int(getattr(response, "status", 0) or 0) >= 400:
            raise RuntimeError(f"Remote LoRA materialization failed with HTTP {response.status}: {raw}")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise RuntimeError("Remote LoRA materialization response was not a JSON object.")
        if not payload.get("ok") or not str(payload.get("name") or "").strip():
            raise RuntimeError(str(payload.get("error") or "Remote LoRA materialization failed."))
        return payload
    except Exception as error:
        if isinstance(error, RuntimeError):
            raise
        raise RuntimeError(f"Remote LoRA materialization failed: {error}") from error
    finally:
        connection.close()
