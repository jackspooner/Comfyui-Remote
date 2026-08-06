"""Opt-in release gate for two already-configured local ComfyUI peers.

This suite never writes peer configuration or includes credentials in its
failure messages.  It is deliberately skipped in the normal portable suite;
set CUTLERY_REMOTE_TWO_PEER=1 and all required variables to run it against
real peers.
"""

from __future__ import annotations

import copy
import hashlib
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
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutlery_remote.capabilities import REMOTE_PROTOCOL_VERSION, validate_remote_group_capabilities


GATE_ENV = "CUTLERY_REMOTE_TWO_PEER"
RELEASE_ENV = "CUTLERY_REMOTE_TWO_PEER_RELEASE"
LOCAL_URL_ENV = "CUTLERY_REMOTE_TWO_PEER_LOCAL_URL"
REMOTE_URL_ENV = "CUTLERY_REMOTE_TWO_PEER_REMOTE_URL"
TOKEN_ENV = "CUTLERY_REMOTE_TWO_PEER_TOKEN"
GROUP_RUN_BODY_ENV = "CUTLERY_REMOTE_TWO_PEER_GROUP_RUN_BODY"
BOUNDARY_GROUP_RUN_BODY_ENV = "CUTLERY_REMOTE_TWO_PEER_BOUNDARY_GROUP_RUN_BODY"
PRELOAD_BODY_ENV = "CUTLERY_REMOTE_TWO_PEER_PRELOAD_BODY"
CANCEL_FIXTURE_ENV = "CUTLERY_REMOTE_TWO_PEER_CANCEL_FIXTURE"
STREAM_BODY_ENV = "CUTLERY_REMOTE_TWO_PEER_STREAM_BODY"
CLIP_TEXT_ENCODE_BODY_ENV = "CUTLERY_REMOTE_TWO_PEER_CLIP_TEXT_ENCODE_BODY"
CLIP_DUAL_TEXT_ENCODE_BODY_ENV = "CUTLERY_REMOTE_TWO_PEER_CLIP_DUAL_TEXT_ENCODE_BODY"
CLIP_QWEN_IMAGE_EDIT_BODY_ENV = "CUTLERY_REMOTE_TWO_PEER_CLIP_QWEN_IMAGE_EDIT_BODY"
CLIP_LORA_TEXT_ENCODE_BODY_ENV = "CUTLERY_REMOTE_TWO_PEER_CLIP_LORA_TEXT_ENCODE_BODY"
LORA_MATERIALIZE_FIXTURE_ENV = "CUTLERY_REMOTE_TWO_PEER_LORA_MATERIALIZE_FIXTURE"
LORA_SIZE_LIMIT_FIXTURE_ENV = "CUTLERY_REMOTE_TWO_PEER_LORA_SIZE_LIMIT_FIXTURE"
LORA_HASH_MISMATCH_FIXTURE_ENV = "CUTLERY_REMOTE_TWO_PEER_LORA_HASH_MISMATCH_FIXTURE"
LORA_CLEANUP_FIXTURE_ENV = "CUTLERY_REMOTE_TWO_PEER_LORA_CLEANUP_FIXTURE"

RELEASE_FIXTURE_ENVS = (
    GROUP_RUN_BODY_ENV,
    BOUNDARY_GROUP_RUN_BODY_ENV,
    PRELOAD_BODY_ENV,
    CANCEL_FIXTURE_ENV,
    STREAM_BODY_ENV,
    CLIP_TEXT_ENCODE_BODY_ENV,
    CLIP_DUAL_TEXT_ENCODE_BODY_ENV,
    CLIP_QWEN_IMAGE_EDIT_BODY_ENV,
    CLIP_LORA_TEXT_ENCODE_BODY_ENV,
    LORA_MATERIALIZE_FIXTURE_ENV,
    LORA_SIZE_LIMIT_FIXTURE_ENV,
    LORA_HASH_MISMATCH_FIXTURE_ENV,
    LORA_CLEANUP_FIXTURE_ENV,
)


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
    release_mode: bool

    @classmethod
    def from_environment(cls) -> "TwoPeerConfig":
        missing = [name for name in (LOCAL_URL_ENV, REMOTE_URL_ENV, TOKEN_ENV) if not os.environ.get(name, "").strip()]
        release_mode = _enabled(os.environ.get(RELEASE_ENV))
        if release_mode:
            missing.extend(name for name in RELEASE_FIXTURE_ENVS if not os.environ.get(name, "").strip())
        if missing:
            required = ", ".join(dict.fromkeys(missing))
            raise ValueError(
                f"{GATE_ENV}=1 requires {required}. "
                "This gate does not read peer .env files or infer credentials."
            )
        if release_mode:
            try:
                _validate_release_fixtures()
            except AssertionError as exc:
                raise ValueError(str(exc)) from exc
        return cls(
            local_url=_origin(os.environ[LOCAL_URL_ENV], LOCAL_URL_ENV),
            remote_url=_origin(os.environ[REMOTE_URL_ENV], REMOTE_URL_ENV),
            token=os.environ[TOKEN_ENV].strip(),
            release_mode=release_mode,
        )


def _json_body(value: str, name: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{name} must contain a JSON object.") from exc
    if not isinstance(parsed, dict):
        raise AssertionError(f"{name} must contain a JSON object.")
    return parsed


def _fixture_body(name: str, *, release_mode: bool) -> dict[str, Any] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        if release_mode:
            raise AssertionError(f"{RELEASE_ENV}=1 requires reviewed fixture {name}.")
        return None
    return _json_body(raw, name)


def _fixture_object(name: str) -> dict[str, Any]:
    return _json_body(os.environ[name], name)


def _required_object(value: object, name: str, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise AssertionError(f"{name}.{field} must be a non-empty JSON object.")
    return value


def _required_string(value: object, name: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssertionError(f"{name}.{field} must be a non-empty string.")
    return value.strip()


def _required_sha256(value: object, name: str, field: str = "sha256") -> str:
    digest = _required_string(value, name, field).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise AssertionError(f"{name}.{field} must be a lowercase SHA-256 digest.")
    return digest


def _fixture_expectation(value: object, name: str, field: str = "expect") -> dict[str, Any]:
    expectation = _required_object(value, name, field)
    if set(expectation) == {"ok"}:
        raise AssertionError(f"{name}.{field} must include evidence beyond the response envelope.")
    return expectation


def _preload_fixture(name: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fixture = _fixture_object(name)
    return (
        _required_object(fixture.get("request"), name, "request"),
        _fixture_expectation(fixture.get("expect_cold"), name, "expect_cold"),
        _fixture_expectation(fixture.get("expect_warm"), name, "expect_warm"),
    )


def _cancellation_fixture(name: str) -> tuple[str, dict[str, Any]]:
    fixture = _fixture_object(name)
    prompt_id = _required_string(fixture.get("prompt_id"), name, "prompt_id")
    expectation = _fixture_expectation(fixture.get("expect"), name)
    if expectation.get("cancellation_recorded") is not True:
        raise AssertionError(f"{name}.expect.cancellation_recorded must be true.")
    if not any(isinstance(expectation.get(key), bool) for key in ("removed_from_queue", "interrupted_running", "cancelled")):
        raise AssertionError(
            f"{name}.expect must declare one of removed_from_queue, interrupted_running, or cancelled."
        )
    return prompt_id, expectation


def _lora_upload_fixture(name: str) -> dict[str, Any]:
    fixture = _fixture_object(name)
    _required_string(fixture.get("path"), name, "path")
    _required_string(fixture.get("name"), name, "name")
    _required_sha256(fixture.get("sha256"), name)
    status = fixture.get("expect_status")
    if not isinstance(status, int) or not 100 <= status <= 599:
        raise AssertionError(f"{name}.expect_status must be an HTTP status integer.")
    _fixture_expectation(fixture.get("expect"), name)
    error_contains = fixture.get("error_contains")
    if status >= 400 and (not isinstance(error_contains, str) or not error_contains.strip()):
        raise AssertionError(f"{name}.error_contains must describe the expected rejection.")
    if status < 400 and error_contains is not None:
        raise AssertionError(f"{name}.error_contains is only valid for rejection fixtures.")
    return fixture


def _lora_cleanup_fixture(name: str) -> dict[str, Any]:
    fixture = _fixture_object(name)
    expectation = _fixture_expectation(fixture.get("expect"), name)
    if not isinstance(expectation.get("deleted_count"), int) or expectation["deleted_count"] < 1:
        raise AssertionError(f"{name}.expect.deleted_count must prove cleanup removed the test LoRA.")
    return fixture


def _validate_release_fixtures() -> None:
    for name in (
        GROUP_RUN_BODY_ENV,
        BOUNDARY_GROUP_RUN_BODY_ENV,
        STREAM_BODY_ENV,
        CLIP_TEXT_ENCODE_BODY_ENV,
        CLIP_DUAL_TEXT_ENCODE_BODY_ENV,
        CLIP_QWEN_IMAGE_EDIT_BODY_ENV,
        CLIP_LORA_TEXT_ENCODE_BODY_ENV,
    ):
        _fixture_object(name)
    _preload_fixture(PRELOAD_BODY_ENV)
    _cancellation_fixture(CANCEL_FIXTURE_ENV)
    materialize = _lora_upload_fixture(LORA_MATERIALIZE_FIXTURE_ENV)
    if materialize["expect_status"] != 200:
        raise AssertionError(f"{LORA_MATERIALIZE_FIXTURE_ENV}.expect_status must be 200.")
    if not _required_string(materialize.get("name"), LORA_MATERIALIZE_FIXTURE_ENV, "name").replace("\\", "/").startswith("cutlery_remote/"):
        raise AssertionError(f"{LORA_MATERIALIZE_FIXTURE_ENV}.name must be under cutlery_remote/ so cleanup is verifiable.")
    size_limit = _lora_upload_fixture(LORA_SIZE_LIMIT_FIXTURE_ENV)
    if size_limit["expect_status"] != 413:
        raise AssertionError(f"{LORA_SIZE_LIMIT_FIXTURE_ENV}.expect_status must be 413.")
    hash_mismatch = _lora_upload_fixture(LORA_HASH_MISMATCH_FIXTURE_ENV)
    if hash_mismatch["expect_status"] != 400:
        raise AssertionError(f"{LORA_HASH_MISMATCH_FIXTURE_ENV}.expect_status must be 400.")
    _lora_cleanup_fixture(LORA_CLEANUP_FIXTURE_ENV)


def _assert_expected_evidence(actual: object, expected: object, context: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"{context} expected an object but received {type(actual).__name__}.")
        for key, expected_value in expected.items():
            if key not in actual:
                raise AssertionError(f"{context} did not include expected field {key!r}.")
            _assert_expected_evidence(actual[key], expected_value, f"{context}.{key}")
        return
    if actual != expected:
        raise AssertionError(f"{context} expected {expected!r}, received {actual!r}.")


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


def _request_lora_upload(
    url: str,
    *,
    token: str,
    fixture: dict[str, Any],
    expected_sha256: str | None = None,
) -> tuple[int, dict[str, Any]]:
    source = Path(_required_string(fixture.get("path"), "LoRA upload fixture", "path"))
    if not source.is_file():
        raise AssertionError(f"LoRA upload fixture path is not a file: {source}")
    expected_digest = _required_sha256(fixture.get("sha256"), "LoRA upload fixture")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_digest = digest.hexdigest()
    if actual_digest != expected_digest:
        raise AssertionError("LoRA upload fixture contents did not match its reviewed SHA-256.")

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
        "Content-Length": str(source.stat().st_size),
        "X-Cutlery-Lora-Name": urllib.parse.quote(_required_string(fixture.get("name"), "LoRA upload fixture", "name")),
        "X-Cutlery-Lora-SHA256": expected_sha256 or actual_digest,
    }
    with source.open("rb") as handle:
        request = urllib.request.Request(url, data=handle, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=600.0) as response:
                status = int(response.status)
                raw = response.read()
        except urllib.error.HTTPError as error:
            status = int(error.code)
            raw = error.read()
        except urllib.error.URLError as error:
            raise AssertionError(f"POST {url} did not reach its configured peer: {error.reason}") from error
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"POST {url} did not return a JSON response (HTTP {status}).") from error
    if not isinstance(payload, dict):
        raise AssertionError(f"POST {url} returned a non-object JSON response (HTTP {status}).")
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

    def _run_group_fixture(self, fixture_env: str, label: str, *, require_outputs: bool = False):
        body = _fixture_body(fixture_env, release_mode=self.config.release_mode)
        if body is None:
            self.skipTest(f"Set {fixture_env} to a reviewed {label} request to execute this optional check.")
        status, payload = _request_json(
            "POST",
            f"{self.config.remote_url}/cutlery/remote/group/run",
            token=self.config.token,
            body=body,
            timeout_seconds=600.0,
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"), payload.get("error") or f"{label} fixture failed")
        if require_outputs:
            self.assertIsInstance(payload.get("outputs"), dict, f"{label} did not return serialized boundary outputs")

    def test_group_run_fixture(self):
        self._run_group_fixture(GROUP_RUN_BODY_ENV, "remote group")

    def test_boundary_group_run_fixture(self):
        self._run_group_fixture(BOUNDARY_GROUP_RUN_BODY_ENV, "boundary group", require_outputs=True)

    def test_preload_or_materialization_fixture_runs_cold_and_warm(self):
        if not os.environ.get(PRELOAD_BODY_ENV, "").strip():
            self.skipTest(
                f"Set {PRELOAD_BODY_ENV} to a reviewed preload request to check cold/warm materialization on the remote peer."
            )
        body, cold_expectation, warm_expectation = _preload_fixture(PRELOAD_BODY_ENV)
        for state, expectation in (("cold", cold_expectation), ("warm", warm_expectation)):
            with self.subTest(state=state):
                status, payload = _request_json(
                    "POST",
                    f"{self.config.remote_url}/cutlery/remote/group/preload",
                    token=self.config.token,
                    body=body,
                    timeout_seconds=600.0,
                )
                self.assertEqual(status, 200)
                self.assertTrue(payload.get("ok"), payload.get("error") or f"{state} remote preload fixture failed")
                _assert_expected_evidence(payload, expectation, f"{state} preload response")

    def test_prompt_cancellation_fixture(self):
        if not os.environ.get(CANCEL_FIXTURE_ENV, "").strip():
            self.skipTest(
                f"Set {CANCEL_FIXTURE_ENV} only for a dedicated pending test job to verify prompt-specific cancellation."
            )
        prompt_id, expectation = _cancellation_fixture(CANCEL_FIXTURE_ENV)
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
        _assert_expected_evidence(payload, expectation, "prompt cancellation response")

    def test_lora_materialization_size_hash_and_cleanup_fixtures(self):
        fixture_names = (
            LORA_MATERIALIZE_FIXTURE_ENV,
            LORA_SIZE_LIMIT_FIXTURE_ENV,
            LORA_HASH_MISMATCH_FIXTURE_ENV,
            LORA_CLEANUP_FIXTURE_ENV,
        )
        if not all(os.environ.get(name, "").strip() for name in fixture_names):
            self.skipTest(
                "Set the reviewed LoRA materialization, size-limit, hash-mismatch, and cleanup fixtures to verify the lifecycle."
            )
        materialize = _lora_upload_fixture(LORA_MATERIALIZE_FIXTURE_ENV)
        size_limit = _lora_upload_fixture(LORA_SIZE_LIMIT_FIXTURE_ENV)
        hash_mismatch = _lora_upload_fixture(LORA_HASH_MISMATCH_FIXTURE_ENV)
        cleanup = _lora_cleanup_fixture(LORA_CLEANUP_FIXTURE_ENV)
        endpoint = f"{self.config.remote_url}/cutlery/remote/clip/loras/materialize"

        status, payload = _request_lora_upload(endpoint, token=self.config.token, fixture=materialize)
        self.assertEqual(status, materialize["expect_status"])
        self.assertEqual(payload.get("name"), materialize["name"])
        self.assertEqual(payload.get("sha256"), materialize["sha256"])
        self.assertEqual(payload.get("size"), Path(materialize["path"]).stat().st_size)
        self.assertTrue(payload.get("materialized"))
        _assert_expected_evidence(payload, materialize["expect"], "LoRA materialization response")

        status, payload = _request_lora_upload(endpoint, token=self.config.token, fixture=size_limit)
        self.assertEqual(status, size_limit["expect_status"])
        self.assertIn(str(size_limit["error_contains"]).lower(), str(payload.get("error") or "").lower())
        _assert_expected_evidence(payload, size_limit["expect"], "LoRA size-limit response")

        actual_digest = _required_sha256(hash_mismatch.get("sha256"), LORA_HASH_MISMATCH_FIXTURE_ENV)
        incorrect_digest = "0" * 64 if actual_digest != "0" * 64 else "1" * 64
        status, payload = _request_lora_upload(
            endpoint,
            token=self.config.token,
            fixture=hash_mismatch,
            expected_sha256=incorrect_digest,
        )
        self.assertEqual(status, hash_mismatch["expect_status"])
        self.assertIn(str(hash_mismatch["error_contains"]).lower(), str(payload.get("error") or "").lower())
        _assert_expected_evidence(payload, hash_mismatch["expect"], "LoRA hash-mismatch response")

        status, payload = _request_json(
            "POST",
            f"{self.config.remote_url}/cutlery/remote/clip/loras/clear",
            token=self.config.token,
            body={},
            timeout_seconds=600.0,
        )
        self.assertEqual(status, 200)
        _assert_expected_evidence(payload, cleanup["expect"], "LoRA cleanup response")

    def _run_remote_clip_fixture(self, fixture_env: str, path: str, label: str):
        body = _fixture_body(fixture_env, release_mode=self.config.release_mode)
        if body is None:
            self.skipTest(f"Set {fixture_env} to a reviewed {label} request.")
        status, payload = _request_json(
            "POST",
            f"{self.config.remote_url}{path}",
            token=self.config.token,
            body=body,
            timeout_seconds=600.0,
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload.get("ok"), payload.get("error") or f"{label} fixture failed")
        self.assertIsInstance(payload.get("conditioning"), dict, f"{label} did not return a conditioning bundle")

    def test_remote_clip_text_encode_fixture(self):
        self._run_remote_clip_fixture(
            CLIP_TEXT_ENCODE_BODY_ENV,
            "/cutlery/remote/clip/text-encode",
            "Remote CLIP text-encode",
        )

    def test_remote_clip_dual_text_encode_fixture(self):
        self._run_remote_clip_fixture(
            CLIP_DUAL_TEXT_ENCODE_BODY_ENV,
            "/cutlery/remote/clip/dual-text-encode",
            "Remote CLIP dual text-encode",
        )

    def test_remote_clip_qwen_image_edit_fixture(self):
        self._run_remote_clip_fixture(
            CLIP_QWEN_IMAGE_EDIT_BODY_ENV,
            "/cutlery/remote/clip/qwen-image-edit-plus",
            "Remote CLIP Qwen image-edit",
        )

    def test_remote_clip_lora_text_encode_fixture(self):
        self._run_remote_clip_fixture(
            CLIP_LORA_TEXT_ENCODE_BODY_ENV,
            "/cutlery/remote/clip/text-encode",
            "Remote CLIP LoRA materialization and text-encode",
        )


class TwoPeerProgressIntegrationTests(_ConfiguredTwoPeerGate, unittest.IsolatedAsyncioTestCase):
    async def test_progress_stream_fixture(self):
        body = _fixture_body(STREAM_BODY_ENV, release_mode=self.config.release_mode)
        if body is None:
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
                await websocket.send_json(body)
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


class TwoPeerReleaseConfigurationTests(unittest.TestCase):
    """Verify release-mode validation without contacting a ComfyUI peer."""

    def _base_environment(self) -> dict[str, str]:
        return {
            GATE_ENV: "1",
            LOCAL_URL_ENV: "http://127.0.0.1:8888",
            REMOTE_URL_ENV: "http://127.0.0.1:8889",
            TOKEN_ENV: "test-token",
        }

    def _release_fixtures(self) -> dict[str, str]:
        return {
            GROUP_RUN_BODY_ENV: "{}",
            BOUNDARY_GROUP_RUN_BODY_ENV: "{}",
            PRELOAD_BODY_ENV: json.dumps(
                {"request": {"prompt_id": "preload-release"}, "expect_cold": {"prompt_id": "preload-release"}, "expect_warm": {"prompt_id": "preload-release"}}
            ),
            CANCEL_FIXTURE_ENV: json.dumps(
                {"prompt_id": "pending-release-job", "expect": {"cancellation_recorded": True, "cancelled": True}}
            ),
            STREAM_BODY_ENV: "{}",
            CLIP_TEXT_ENCODE_BODY_ENV: "{}",
            CLIP_DUAL_TEXT_ENCODE_BODY_ENV: "{}",
            CLIP_QWEN_IMAGE_EDIT_BODY_ENV: "{}",
            CLIP_LORA_TEXT_ENCODE_BODY_ENV: "{}",
            LORA_MATERIALIZE_FIXTURE_ENV: json.dumps(
                {
                    "path": "C:/release-fixtures/materialize.safetensors",
                    "name": "cutlery_remote/release-materialize.safetensors",
                    "sha256": "a" * 64,
                    "expect_status": 200,
                    "expect": {"name": "cutlery_remote/release-materialize.safetensors", "materialized": True},
                }
            ),
            LORA_SIZE_LIMIT_FIXTURE_ENV: json.dumps(
                {
                    "path": "C:/release-fixtures/oversize.safetensors",
                    "name": "cutlery_remote/release-oversize.safetensors",
                    "sha256": "b" * 64,
                    "expect_status": 413,
                    "expect": {"error": "Uploaded LoRA is too large."},
                    "error_contains": "too large",
                }
            ),
            LORA_HASH_MISMATCH_FIXTURE_ENV: json.dumps(
                {
                    "path": "C:/release-fixtures/hash-mismatch.safetensors",
                    "name": "cutlery_remote/release-hash-mismatch.safetensors",
                    "sha256": "c" * 64,
                    "expect_status": 400,
                    "expect": {"error": "Uploaded LoRA SHA-256 did not match the expected hash."},
                    "error_contains": "sha-256",
                }
            ),
            LORA_CLEANUP_FIXTURE_ENV: json.dumps({"expect": {"deleted_count": 1}}),
        }

    def test_non_release_gate_does_not_require_execution_fixtures(self):
        with patch.dict(os.environ, self._base_environment(), clear=True):
            config = TwoPeerConfig.from_environment()
        self.assertFalse(config.release_mode)

    def test_release_gate_lists_every_missing_reviewed_fixture_before_any_network_work(self):
        environment = self._base_environment()
        environment[RELEASE_ENV] = "1"
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(ValueError, GROUP_RUN_BODY_ENV) as error:
                TwoPeerConfig.from_environment()
        message = str(error.exception)
        for name in RELEASE_FIXTURE_ENVS:
            self.assertIn(name, message)

    def test_release_gate_accepts_all_reviewed_execution_fixtures(self):
        environment = self._base_environment()
        environment.update(self._release_fixtures())
        environment[RELEASE_ENV] = "1"
        with patch.dict(os.environ, environment, clear=True):
            config = TwoPeerConfig.from_environment()
        self.assertTrue(config.release_mode)

    def test_release_gate_rejects_a_malformed_fixture_before_peer_setup(self):
        environment = self._base_environment()
        environment.update(self._release_fixtures())
        environment.update({RELEASE_ENV: "1", GROUP_RUN_BODY_ENV: "not-json"})
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(ValueError, "must contain a JSON object"):
                TwoPeerConfig.from_environment()

    def test_release_gate_rejects_fixture_without_cold_warm_or_cancellation_evidence(self):
        environment = self._base_environment()
        environment.update(self._release_fixtures())
        environment.update(
            {
                RELEASE_ENV: "1",
                PRELOAD_BODY_ENV: json.dumps(
                    {"request": {"prompt_id": "preload-release"}, "expect_cold": {"ok": True}, "expect_warm": {"ok": True}}
                ),
            }
        )
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(ValueError, "evidence beyond the response envelope"):
                TwoPeerConfig.from_environment()

        environment.update(
            {
                PRELOAD_BODY_ENV: self._release_fixtures()[PRELOAD_BODY_ENV],
                CANCEL_FIXTURE_ENV: json.dumps(
                    {"prompt_id": "pending-release-job", "expect": {"cancellation_recorded": True}}
                ),
            }
        )
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(ValueError, "must declare one of"):
                TwoPeerConfig.from_environment()

    def test_release_gate_rejects_lora_fixture_without_rejection_evidence(self):
        environment = self._base_environment()
        environment.update(self._release_fixtures())
        size_fixture = json.loads(environment[LORA_SIZE_LIMIT_FIXTURE_ENV])
        size_fixture.pop("error_contains")
        environment.update({RELEASE_ENV: "1", LORA_SIZE_LIMIT_FIXTURE_ENV: json.dumps(size_fixture)})
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(ValueError, "error_contains"):
                TwoPeerConfig.from_environment()


if __name__ == "__main__":
    unittest.main()
