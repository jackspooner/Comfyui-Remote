from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


_GGUF_MODULE_NAME = "_cutlery_comfyui_gguf"


def _folder_paths():
    import folder_paths

    return folder_paths


def _folder_paths_module(folders: Any | None = None):
    return folders if folders is not None else _folder_paths()


def _folder_key_exists(folder_name: str, folders: Any | None = None) -> bool:
    mapping = getattr(_folder_paths_module(folders), "folder_names_and_paths", None)
    if isinstance(mapping, dict):
        return folder_name in mapping
    return True


def filename_list(folder_name: str, folders: Any | None = None) -> list[str]:
    try:
        return list(_folder_paths_module(folders).get_filename_list(folder_name))
    except Exception:
        return []


def get_gguf_clip_loader():
    package = sys.modules.get(_GGUF_MODULE_NAME)
    try:
        mappings = getattr(package, "NODE_CLASS_MAPPINGS", None)
    except Exception:
        mappings = None
    if isinstance(mappings, dict) and "CLIPLoaderGGUF" in mappings:
        return mappings["CLIPLoaderGGUF"]()

    for module in list(sys.modules.values()):
        try:
            mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
        except Exception:
            continue
        if isinstance(mappings, dict) and "CLIPLoaderGGUF" in mappings:
            return mappings["CLIPLoaderGGUF"]()

    gguf_init = Path(__file__).resolve().parents[1] / "ComfyUI-GGUF" / "__init__.py"
    if gguf_init.exists():
        spec = importlib.util.spec_from_file_location(
            _GGUF_MODULE_NAME,
            gguf_init,
            submodule_search_locations=[str(gguf_init.parent)],
        )
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            sys.modules[_GGUF_MODULE_NAME] = module
            spec.loader.exec_module(module)
            mappings = getattr(module, "NODE_CLASS_MAPPINGS", None)
            if isinstance(mappings, dict) and "CLIPLoaderGGUF" in mappings:
                return mappings["CLIPLoaderGGUF"]()

    raise RuntimeError("ComfyUI-GGUF is required to load CLIP/text encoders from GGUF files.")


def list_clip_text_encoder_names(folders: Any | None = None) -> list[str]:
    gguf_names = filename_list("clip_gguf", folders)
    if not gguf_names and not _folder_key_exists("clip_gguf", folders):
        try:
            get_gguf_clip_loader()
            gguf_names = filename_list("clip_gguf", folders)
        except Exception:
            gguf_names = []
    return sorted({*filename_list("text_encoders", folders), *gguf_names}, key=str.casefold)


def resolve_clip_text_encoder_path(text_encoder: str, folders: Any | None = None) -> str:
    name = str(text_encoder or "")
    folder_paths = _folder_paths_module(folders)
    if name.lower().endswith(".gguf"):
        try:
            return folder_paths.get_full_path_or_raise("clip_gguf", name)
        except Exception:
            get_gguf_clip_loader()
            return folder_paths.get_full_path_or_raise("clip_gguf", name)
    return folder_paths.get_full_path_or_raise("text_encoders", name)


def load_gguf_clip(clip_paths: list[str], clip_type: Any):
    gguf_loader = get_gguf_clip_loader()
    return gguf_loader.load_patcher(clip_paths, clip_type, gguf_loader.load_data(clip_paths))
