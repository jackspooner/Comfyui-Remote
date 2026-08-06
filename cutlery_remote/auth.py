from __future__ import annotations

import hmac
from typing import Mapping

from .dotenv import env_value


TOKEN_ENV_VAR = "CUTLERY_REMOTE_TOKEN"


def configured_remote_token() -> str:
    return env_value(TOKEN_ENV_VAR)


def build_auth_headers(token: str) -> dict[str, str]:
    clean_token = str(token or "").strip()
    return {"Authorization": f"Bearer {clean_token}"} if clean_token else {}


def is_authorized(headers: Mapping[str, str] | None, expected_token: str | None = None) -> bool:
    token = str(expected_token if expected_token is not None else configured_remote_token()).strip()
    if not token:
        return False

    auth_header = ""
    if headers is not None:
        auth_header = str(headers.get("Authorization") or headers.get("authorization") or "")
    scheme, _, value = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return False
    return hmac.compare_digest(value.strip(), token)
