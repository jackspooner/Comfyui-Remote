"""Remote-only configuration and mutable-data paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

try:
    from .cutlery_remote.dotenv import load_comfy_root_dotenv
except ImportError:  # pragma: no cover - supports direct module imports in tests.
    from cutlery_remote.dotenv import load_comfy_root_dotenv


REMOTE_SERVER_ENV = "CUTLERY_REMOTE_SERVER_ENABLED"
REMOTE_CLIP_SERVER_ENV = "CUTLERY_REMOTE_CLIP_SERVER_ENABLED"
WORKFLOW_RUN_ENV = "CUTLERY_WORKFLOW_RUN_ENABLED"
CUTLERY_DATA_DIR_ENV = "CUTLERY_DATA_DIR"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


@dataclass(frozen=True)
class FeatureConfig:
    remote_server: bool
    remote_clip_server: bool
    workflow_run: bool
    data_dir: Path


def _comfy_root() -> Path:
    try:
        import folder_paths

        base_path = str(getattr(folder_paths, "base_path", "") or "").strip()
        if base_path:
            return Path(base_path).resolve()
    except ImportError:
        pass
    return Path(__file__).resolve().parents[2]


def _value(name: str, default: str, env: Mapping[str, str], dotenv: Mapping[str, str]) -> str:
    return str(env.get(name) or dotenv.get(name) or default).strip()


def strict_bool(
    name: str,
    default: bool,
    *,
    env: Mapping[str, str] | None = None,
    dotenv: Mapping[str, str] | None = None,
) -> bool:
    raw_value = _value(name, "1" if default else "0", os.environ if env is None else env, load_comfy_root_dotenv() if dotenv is None else dotenv)
    value = raw_value.casefold()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    accepted = ", ".join(sorted(_TRUE_VALUES | _FALSE_VALUES))
    raise ValueError(f"{name} must be one of: {accepted}.")


def load_feature_config(
    *,
    env: Mapping[str, str] | None = None,
    dotenv: Mapping[str, str] | None = None,
) -> FeatureConfig:
    env_values = os.environ if env is None else env
    dotenv_values = load_comfy_root_dotenv() if dotenv is None else dotenv
    data_dir_value = _value(CUTLERY_DATA_DIR_ENV, str(_comfy_root() / "user" / "__cutlery_remote"), env_values, dotenv_values)
    data_dir = Path(data_dir_value).expanduser()
    if CUTLERY_DATA_DIR_ENV in env_values and not data_dir.is_absolute():
        raise ValueError(f"{CUTLERY_DATA_DIR_ENV} must be an absolute path.")
    resolved_data_dir = data_dir.resolve()
    package_root = Path(__file__).resolve().parent
    if resolved_data_dir == package_root or resolved_data_dir.is_relative_to(package_root):
        raise ValueError(f"{CUTLERY_DATA_DIR_ENV} must not point inside the installed Cutlery Remote package.")
    return FeatureConfig(
        remote_server=strict_bool(REMOTE_SERVER_ENV, False, env=env_values, dotenv=dotenv_values),
        remote_clip_server=strict_bool(REMOTE_CLIP_SERVER_ENV, False, env=env_values, dotenv=dotenv_values),
        workflow_run=strict_bool(WORKFLOW_RUN_ENV, True, env=env_values, dotenv=dotenv_values),
        data_dir=resolved_data_dir,
    )


@lru_cache(maxsize=1)
def get_feature_config() -> FeatureConfig:
    return load_feature_config()


def data_path(*parts: str) -> Path:
    root = get_feature_config().data_dir
    candidate = root.joinpath(*parts).resolve()
    if candidate != root and not candidate.is_relative_to(root):
        raise ValueError("Cutlery Remote data paths must remain inside CUTLERY_DATA_DIR.")
    return candidate


__all__ = [
    "CUTLERY_DATA_DIR_ENV",
    "FeatureConfig",
    "REMOTE_CLIP_SERVER_ENV",
    "REMOTE_SERVER_ENV",
    "WORKFLOW_RUN_ENV",
    "data_path",
    "get_feature_config",
    "load_feature_config",
    "strict_bool",
]
