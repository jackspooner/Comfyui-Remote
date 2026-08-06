"""Opt-in release gate for two already-configured local ComfyUI peers.

This suite never writes peer configuration or includes credentials in its
failure messages.  It is deliberately skipped in the normal portable suite;
set CUTLERY_REMOTE_TWO_PEER=1 and all required variables to run it against
real peers.
"""

from __future__ import annotations

import copy
import json
import os
import secrets
import sys
import unittest
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutlery_remote.capabilities import REMOTE_PROTOCOL_VERSION, validate_remote_group_capabilities


GATE_ENV = "CUTLERY_REMOTE_TWO_PEER"
LOCAL_URL_ENV = "CUTLERY_REMOTE_TWO_PEER_LOCAL_URL"
REMOTE_URL_ENV = "CUTLERY_REMOTE_TWO_PEER_REMOTE_URL"
TOKEN_ENV = "CUTLERY_REMOTE_TWO_PEER_TOKEN"
GROUP_RUN_BODY_ENV = "CUTLERY_REMOTE_TWO_PEER_GROUP_RUN_BODY"
PRELOAD_BODY_ENV = "CUTLERY_REMOTE_TWO_PEER_PRELOAD_BODY"
CANCEL_PROMPT_ID_ENV = "CUTLERY_REMOTE_TWO_PEER_CANCEL_PROMPT_ID"
STREAM_BODY_ENV = "CUTLERY_REMOTE_TWO_PEER_STREAM_BODY"


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _origin(value: str, name: str) -> str:
    parsed = urllib.parse.urlsplit(value.strip().rstrip("/"))
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.port is None
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{name} must be an http(s) origin with an explicit host and port.")
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{parsed.scheme}://{host}:{parsed.port}"


@dataclass(frozen=True)
class TwoPeerConfig:
    local_url: str
    remote_url: str
    token: str

    @classmethod
    def from_environment(cls) -> "TwoPeerConfig":
        missing = [name for name in (LOCAL_URL_ENV, REMOTE_URL_ENV, TOKEN_ENV) if not os.environ.get(name, "").strip()]
        if missing:
            raise ValueError(
                f"{GATE_ENV}=1 requires " + ", ".join(missing) + ". "
                "This gate does not read peer .env files or infer credentials."
            )
        return cls(
            local_url=_origin(os.environ[LOCAL_URL_ENV], LOCAL_URL_ENV),
            remote_url=_origin(os.environ[REMOTE_URL_ENV], REMOTE_URL_ENV),
            token=os.environ[TOKEN_ENV].strip(),
        )


def _json_body(value: str, name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{name} must contain a JSON object.") from exc
    if not isinstance(parsed, dict):
        raise AssertionError(f"{name} must contain a JSON object.")
    return parsed


def _request_json(
    method: str,
    url: str,
    *,
    token: str | None = None,
    body: dict[str, Any] | None = None,
    timeout_seconds: float = 20.0,
) -> tuple[int, dict[str, Any]]:
    headers = {"Accept": "application/json"}
    data = None
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            raw = response.read()
    except urllib.error.HTTPError as error:
        status = int(error.code)
        raw = error.read()
    except urllib.error.URLError as error:
        raise AssertionError(f"{method} {url} did not reach its configured peer: {error.reason}") from error
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"{method} {url} did not return a JSON response (HTTP {status}).") from error
    if not isinstance(payload, dict):
        raise AssertionError(f"{method} {url} returned a non-object JSON response (HTTP {status}).")
    return status, payload


class _ConfiguredTwoPeerGate:
    @classmethod
    def setUpClass(cls):
        if not _enabled(os.environ.get(GATE_ENV)):
            raise unittest.SkipTest(
                f"Set {GATE_ENV}=1 plus {LOCAL_URL_ENV}, {REMOTE_URL_ENV}, and {TOKEN_ENV} to run real-peer checks."
            )
        try:
            cls.config = TwoPeerConfig.from_environment()
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc


class TwoPeerIntegrationTests(_ConfiguredTwoPeerGate, unittest.TestCase):
    """Exercise configured peers without modifying their configuration or files."""

    def test_capability_preflight_and_authenticated_ordering(self):
        local_status, local_features = _request_json("GET", f"{self.config.local_url}/cutlery/features")
        remote_status, remote_features = _request_json("GET", f"{self.config.remote_url}/cutlery/features")
        self.assertEqual(local_status, 200)
        self.assertEqual(remote_status, 200)
        self.assertTrue(local_features.get("ok"))
        self.assertTrue(remote_features.get("ok"))
        self.assertTrue(remote_features.get("features", {}).get("remote_server"))

        capability_url = f"{self.config.remote_url}/cutlery/remote/capabilities"
        missing_status, _missing = _request_json("GET", capability_url)
        invalid_status, _invalid = _request_json("GET", capability_url, token=secrets.token_urlsafe(48))
        accepted_status, capabilities = _request_json("GET", capability_url, token=self.config.token)
        self.assertEqual(missing_status, 401, "enabled peer must reject an absent token before request work")
        self.assertEqual(invalid_status, 401, "enabled peer must reject an invalid token before request work")
        self.assertEqual(accepted_status, 200)
        self.assertTrue(capabilities.get("ok"))
        self.assertEqual(capabilities.get("protocol_version"), REMOTE_PROTOCOL_VERSION)
        self.assertTrue(capabilities.get("features", {}).get("prompt_specific_interrupt"))
        self.assertTrue(capabilities.get("features", {}).get("remote_progress_v1"))
        validate_remote_group_capabilities(capabilities)

        incompatible = copy.deepcopy(capabilities)
        incompatible["protocol_version"] = REMOTE_PROTOCOL_VERSION + 1
        with self.assertRaisesRegex(RuntimeError, "incompatible"):
            validate_remote_group_capabilities(incompatible)

    def test_sender_rejects_an_untrusted_loopback_origin_before_proxying(self):
        # Port 1 is deliberately not a peer. A 403 proves the sender did not
        # proxy this browser-controlled target or attach its bearer token.
        status, payload = _request_json(
            "POST",
            f"{self.config.local_url}/cutlery/remote/proxy/node-definitions",
            body={"target": "http://127.0.0.1:1", "class_types": []},
        )
        self.assertEqual(status, 403)
        self.assertFalse(payload.get("ok"))
        self.assertIn("not trusted", str(payload.get("error") or "").lower())

    def test_optional_group_run_fixture(self):
        raw = os.environ.get(GROUP_RUN_BODY_ENV, "").strip()
        if not raw:
            self.skipTest(f"Set {GROUP_RUN_BODY_ENV} to a reviewed compiled group request to execute this optional check.")
        status, payload = _request_json(
            "POST",
            f"{self.config.remote_url}/cutlery/remote/group/run",
            token=self.config.token,
            body=_json_body(raw, GROUP_RUN_BODY_ENV),
            timeout_seconds=600.0,
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"), payload.get("error") or "remote group fixture failed")

    def test_optional_preload_or_materialization_fixture(self):
        raw = os.environ.get(PRELOAD_BODY_ENV, "").strip()
        if not raw:
            self.skipTest(
                f"Set {PRELOAD_BODY_ENV} to a reviewed preload request to check cold/warm materialization on the remote peer."
            )
        status, payload = _request_json(
            "POST",
            f"{self.config.remote_url}/cutlery/remote/group/preload",
            token=self.config.token,
            body=_json_body(raw, PRELOAD_BODY_ENV),
            timeout_seconds=600.0,
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"), payload.get("error") or "remote preload fixture failed")

    def test_optional_prompt_cancellation_fixture(self):
        prompt_id = os.environ.get(CANCEL_PROMPT_ID_ENV, "").strip()
        if not prompt_id:
            self.skipTest(
                f"Set {CANCEL_PROMPT_ID_ENV} only for a dedicated pending test job to verify prompt-specific cancellation."
            )
        quoted_id = urllib.parse.quote(prompt_id, safe="")
        status, payload = _request_json(
            "POST",
            f"{self.config.remote_url}/cutlery/remote/group/{quoted_id}/interrupt",
            token=self.config.token,
            body={},
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"), payload.get("error") or "prompt cancellation fixture failed")
        self.assertEqual(payload.get("remote_prompt_id"), prompt_id)
        self.assertTrue(payload.get("cancellation_recorded"))


class TwoPeerProgressIntegrationTests(_ConfiguredTwoPeerGate, unittest.IsolatedAsyncioTestCase):
    async def test_optional_progress_stream_fixture(self):
        raw = os.environ.get(STREAM_BODY_ENV, "").strip()
        if not raw:
            self.skipTest(
                f"Set {STREAM_BODY_ENV} to a reviewed streamed group request that emits at least one progress envelope."
            )
        try:
            import aiohttp
        except ImportError:
            self.fail("aiohttp is required for the optional Remote progress-stream release fixture.")

        parsed = urllib.parse.urlsplit(self.config.remote_url)
        websocket_url = urllib.parse.urlunsplit(
            ("wss" if parsed.scheme == "https" else "ws", parsed.netloc, "/cutlery/remote/group/run-stream", "", "")
        )
        progress_count = 0
        terminal: dict[str, Any] | None = None
        timeout = aiohttp.ClientTimeout(total=600.0)
        headers = {"Authorization": f"Bearer {self.config.token}"}
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(websocket_url, headers=headers, heartbeat=30.0) as websocket:
                await websocket.send_json(_json_body(raw, STREAM_BODY_ENV))
                async for message in websocket:
                    if message.type != aiohttp.WSMsgType.TEXT:
                        self.fail("Remote progress stream closed without a text terminal envelope.")
                    payload = message.json()
                    if not isinstance(payload, dict):
                        self.fail("Remote progress stream returned a non-object envelope.")
                    if payload.get("type") == "progress":
                        progress_count += 1
                        continue
                    if payload.get("type") in {"result", "error"}:
                        terminal = payload
                        break
        self.assertGreater(progress_count, 0, "The reviewed streamed fixture did not emit remote progress.")
        self.assertIsNotNone(terminal, "Remote progress stream did not return a terminal envelope.")
        self.assertEqual(terminal.get("type"), "result", terminal.get("data"))
        self.assertTrue(terminal.get("data", {}).get("ok"), terminal.get("data"))


if __name__ == "__main__":
    unittest.main()
