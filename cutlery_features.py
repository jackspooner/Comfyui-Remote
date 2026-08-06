from __future__ import annotations

from typing import Any

try:
    from .cutlery_config import get_feature_config
except ImportError:  # pragma: no cover - supports direct module imports in tests.
    from cutlery_config import get_feature_config

try:
    from aiohttp import web
    from server import PromptServer
except Exception:
    web = None
    PromptServer = None


FEATURE_MARKER = "cutlery-features-v1"
FEATURE_FIELDS = {
    "remote_server": "remote_server",
    "remote_clip_server": "remote_clip_server",
    "workflow_run": "workflow_run",
}


def features_payload() -> dict[str, Any]:
    config = get_feature_config()
    return {
        "ok": True,
        "marker": FEATURE_MARKER,
        "features": {
            public_name: bool(getattr(config, field_name))
            for public_name, field_name in FEATURE_FIELDS.items()
        },
    }


def feature_disabled_response(
    field_name: str,
    *,
    code: str,
    env_var: str,
    web_module: Any | None = None,
):
    """Return a stable 403 response when a startup-gated route is disabled."""

    if bool(getattr(get_feature_config(), field_name)):
        return None
    payload = {
        "ok": False,
        "code": code,
        "error": f"This Cutlery feature is disabled. Set {env_var}=1 and restart ComfyUI to enable it.",
    }
    response_web = web if web_module is None else web_module
    if response_web is None:
        return payload, 403
    return response_web.json_response(payload, status=403)


def register_feature_route() -> None:
    if PromptServer is None or web is None or getattr(PromptServer, "instance", None) is None:
        return
    routes = PromptServer.instance.routes
    if getattr(routes, "_cutlery_features_route_registered", False):
        return
    response_web = web

    @routes.get("/cutlery/features")
    async def cutlery_features(_request):
        return response_web.json_response(features_payload())

    setattr(routes, "_cutlery_features_route_registered", True)


register_feature_route()


__all__ = [
    "FEATURE_MARKER",
    "feature_disabled_response",
    "features_payload",
    "register_feature_route",
]
