from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


def _comfy_root_dotenv_path() -> Path:
    try:
        import folder_paths

        base_path = getattr(folder_paths, "base_path", "")
        if base_path:
            return Path(base_path) / ".env"
    except Exception:
        pass
    return Path(__file__).resolve().parents[3] / ".env"


def _strip_inline_comment(value: str) -> str:
    quote = ""
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = ""
            elif not quote:
                quote = char
            continue
        if char == "#" and not quote:
            return value[:index].rstrip()
    return value.strip()


def parse_dotenv_value(raw_value: str) -> str:
    value = _strip_inline_comment(str(raw_value or "").strip())
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_comfy_root_dotenv(path: str | Path | None = None) -> dict[str, str]:
    dotenv_path = Path(path) if path is not None else _comfy_root_dotenv_path()
    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

    values: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, separator, raw_value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        if not key:
            continue
        values[key] = parse_dotenv_value(raw_value)
    return values


def env_value(name: str, default: str = "", env: Mapping[str, str] | None = None) -> str:
    source = os.environ if env is None else env
    value = str(source.get(name) or "").strip()
    if value:
        return value
    return str(load_comfy_root_dotenv().get(name) or default).strip()
