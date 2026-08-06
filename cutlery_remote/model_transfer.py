from __future__ import annotations

import base64
import hashlib
import logging
from pathlib import Path
import posixpath
import subprocess
import time
from typing import Any, Callable, Sequence
import uuid

from .dotenv import env_value
from .inventory import (
    MODEL_TRANSFER_STAGING_PREFIX,
    MODEL_TRANSFER_STAGING_SUFFIX,
    normalize_model_type,
)


LOGGER = logging.getLogger("cutlery.remote.model_transfer")

REMOTE_MODEL_COPY_HOST_ENV = "CUTLERY_REMOTE_MODEL_COPY_HOST"
REMOTE_MODEL_COPY_ROOT_ENV = "CUTLERY_REMOTE_MODEL_COPY_ROOT"
DEFAULT_REMOTE_MODEL_COPY_HOST = ""
DEFAULT_REMOTE_MODEL_COPY_ROOT = ""
MODEL_HASH_CHUNK_SIZE = 8 * 1024 * 1024
REMOTE_CLEANUP_TIMEOUT_SECONDS = 30.0

REMOTE_COPY_FOLDER_KEYS = {
    "clip_gguf": "text_encoders",
    "unet_gguf": "diffusion_models",
}


def remote_model_copy_host() -> str:
    host = str(env_value(REMOTE_MODEL_COPY_HOST_ENV, DEFAULT_REMOTE_MODEL_COPY_HOST) or "").strip()
    if not host:
        raise ValueError(
            f"Remote model copy host is not configured. Set {REMOTE_MODEL_COPY_HOST_ENV} "
            "or provide copy_host on the selected Cutlery remote target."
        )
    return host


def remote_model_copy_root() -> str:
    root = str(env_value(REMOTE_MODEL_COPY_ROOT_ENV, DEFAULT_REMOTE_MODEL_COPY_ROOT) or "").strip()
    if not root:
        raise ValueError(
            f"Remote model copy root is not configured. Set {REMOTE_MODEL_COPY_ROOT_ENV} "
            "or provide copy_root on the selected Cutlery remote target."
        )
    return root.replace("\\", "/").rstrip("/")


def _relative_model_parts(model_name: object) -> list[str]:
    text = str(model_name or "").strip().replace("\\", "/").strip("/")
    parts = [part for part in text.split("/") if part and part != "."]
    if not parts:
        raise ValueError("model_name must include a filename.")
    if any(part == ".." for part in parts):
        raise ValueError(f"Refusing to copy model with unsafe relative path {model_name!r}.")
    return parts


def remote_model_destination_folder(model_type: object, model_name: object, *, root: str | None = None) -> str:
    folder_key = normalize_model_type(model_type)
    destination_key = REMOTE_COPY_FOLDER_KEYS.get(folder_key, folder_key)
    root_folder = (root if root is not None else remote_model_copy_root()).replace("\\", "/").rstrip("/")
    parts = _relative_model_parts(model_name)
    folder_parts = [root_folder, destination_key, *parts[:-1]]
    return posixpath.join(*folder_parts).rstrip("/")


def _throw_if_interrupted() -> None:
    try:
        from comfy import model_management
    except Exception:
        return
    throw_interrupted = getattr(model_management, "throw_exception_if_processing_interrupted", None)
    if callable(throw_interrupted):
        throw_interrupted()


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


def _format_command(command: Sequence[str]) -> str:
    formatted = [str(part) for part in command]
    for index, part in enumerate(formatted[:-1]):
        if part.casefold() == "-encodedcommand":
            formatted[index + 1] = "<encoded PowerShell omitted>"
    return subprocess.list2cmdline(formatted)


def _source_integrity(source: Path, check_cancelled: Callable[[], None] | None = None) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as handle:
        while True:
            _throw_if_interrupted()
            if check_cancelled is not None:
                check_cancelled()
            chunk = handle.read(MODEL_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    _throw_if_interrupted()
    if check_cancelled is not None:
        check_cancelled()
    return size, digest.hexdigest()


def _run_interruptible_command(
    command: Sequence[str],
    *,
    description: str,
    timeout_interval: float = 0.5,
    stream_to_console: bool = False,
    check_cancelled: Callable[[], None] | None = None,
    process_started: Callable[[subprocess.Popen[str]], None] | None = None,
    process_finished: Callable[[subprocess.Popen[str]], None] | None = None,
) -> str:
    _throw_if_interrupted()
    if check_cancelled is not None:
        check_cancelled()
    LOGGER.info("[Cutlery Remote] Running %s: %s", description, _format_command(command))
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=None if stream_to_console else subprocess.PIPE,
        stderr=None if stream_to_console else subprocess.STDOUT,
        text=not stream_to_console,
    )
    if process_started is not None:
        process_started(process)
    try:
        if stream_to_console:
            while process.poll() is None:
                time.sleep(timeout_interval)
                _throw_if_interrupted()
                if check_cancelled is not None:
                    check_cancelled()
            output = ""
        else:
            while True:
                try:
                    output, _ = process.communicate(timeout=timeout_interval)
                    break
                except subprocess.TimeoutExpired:
                    _throw_if_interrupted()
                    if check_cancelled is not None:
                        check_cancelled()
    except BaseException:
        _terminate_process(process)
        raise
    finally:
        if process_finished is not None:
            process_finished(process)

    output = output or ""
    if process.returncode != 0:
        raise RuntimeError(f"{description} failed with exit code {process.returncode}: {output.strip()}")
    return output


def _run_uninterruptible_cleanup_command(
    command: Sequence[str],
    *,
    description: str,
    timeout_seconds: float = REMOTE_CLEANUP_TIMEOUT_SECONDS,
) -> None:
    LOGGER.info("[Cutlery Remote] Running %s: %s", description, _format_command(command))
    completed = subprocess.run(
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=max(1.0, float(timeout_seconds)),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{description} failed with exit code {completed.returncode}: {(completed.stdout or '').strip()}"
        )


def _remote_mkdir_command(remote_folder: str) -> str:
    remote_folder_cmd = remote_folder.replace("/", "\\").rstrip("\\")
    if '"' in remote_folder_cmd:
        raise ValueError(f"Remote model folder contains an unsupported quote character: {remote_folder!r}.")
    return f'cmd.exe /d /c if not exist "{remote_folder_cmd}" mkdir "{remote_folder_cmd}"'


def _powershell_literal(value: object) -> str:
    text = str(value)
    if "\x00" in text or "\r" in text or "\n" in text:
        raise ValueError("Remote model path contains an unsupported control character.")
    return "'" + text.replace("'", "''") + "'"


def _powershell_command(script: str) -> str:
    encoded_script = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    return (
        "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass "
        f"-EncodedCommand {encoded_script}"
    )


def _remote_verify_and_promote_command(
    staging_path: str,
    final_path: str,
    *,
    expected_size: int,
    expected_sha256: str,
) -> str:
    stage = _powershell_literal(staging_path.replace("/", "\\"))
    final = _powershell_literal(final_path.replace("/", "\\"))
    expected_hash = _powershell_literal(expected_sha256.lower())
    script = "; ".join(
        (
            "$ErrorActionPreference='Stop'",
            "$ProgressPreference='SilentlyContinue'",
            f"$stage={stage}",
            f"$final={final}",
            f"$expectedSize={int(expected_size)}",
            f"$expectedHash={expected_hash}",
            "if (-not [System.IO.File]::Exists($stage)) { throw 'Staged model file is missing.' }",
            "$actualSize=(Get-Item -LiteralPath $stage).Length",
            (
                "if ($actualSize -ne $expectedSize) "
                "{ throw ('Staged model size mismatch: expected ' + "
                "$expectedSize + ', got ' + $actualSize + '.') }"
            ),
            "$actualHash=(Get-FileHash -LiteralPath $stage -Algorithm SHA256).Hash.ToLowerInvariant()",
            (
                "if ($actualHash -ne $expectedHash) "
                "{ throw ('Staged model SHA-256 mismatch: got ' + $actualHash + '.') }"
            ),
            (
                "$acceptExisting={ if (-not [System.IO.File]::Exists($final)) "
                "{ return $false }; $finalSize=(Get-Item -LiteralPath $final).Length; "
                "if ($finalSize -ne $expectedSize) "
                "{ throw ('Final model file conflicts by size: expected ' + "
                "$expectedSize + ', got ' + $finalSize + '.') }; "
                "$finalHash=(Get-FileHash -LiteralPath $final -Algorithm SHA256).Hash.ToLowerInvariant(); "
                "if ($finalHash -ne $expectedHash) "
                "{ throw ('Final model file conflicts by SHA-256: got ' + $finalHash + '.') }; "
                "[System.IO.File]::Delete($stage); return $true }"
            ),
            (
                "if (-not (& $acceptExisting)) { try { [System.IO.File]::Move($stage, $final) } "
                "catch [System.IO.IOException] { if (-not (& $acceptExisting)) { throw } } }"
            ),
        )
    )
    return _powershell_command(script)


def _remote_remove_staging_command(staging_path: str) -> str:
    stage = _powershell_literal(staging_path.replace("/", "\\"))
    script = "; ".join(
        (
            "$ErrorActionPreference='Stop'",
            f"$stage={stage}",
            "if ([System.IO.File]::Exists($stage)) { [System.IO.File]::Delete($stage) }",
        )
    )
    return _powershell_command(script)


def _new_staging_filename() -> str:
    return (
        f"{MODEL_TRANSFER_STAGING_PREFIX}{uuid.uuid4().hex}"
        f"{MODEL_TRANSFER_STAGING_SUFFIX}"
    )


def _remove_remote_staging_best_effort(host: str, staging_path: str) -> None:
    try:
        _run_uninterruptible_cleanup_command(
            ["ssh", host, _remote_remove_staging_command(staging_path)],
            description="Remote model staging cleanup",
        )
    except BaseException:
        LOGGER.warning(
            "[Cutlery Remote] Failed to remove remote model staging file host=%s path=%s",
            host,
            staging_path,
            exc_info=True,
        )


def copy_model_file_to_remote(
    local_path: str | Path,
    model_type: object,
    remote_model_name: object,
    *,
    remote_host: str | None = None,
    remote_root: str | None = None,
    check_cancelled: Callable[[], None] | None = None,
    process_started: Callable[[subprocess.Popen[str]], None] | None = None,
    process_finished: Callable[[subprocess.Popen[str]], None] | None = None,
) -> dict[str, Any]:
    source = Path(local_path)
    if not source.is_file():
        raise FileNotFoundError(f"Local model file does not exist: {source}")

    host = str(remote_host or "").strip() or remote_model_copy_host()
    root = str(remote_root or "").strip() or remote_model_copy_root()
    remote_parts = _relative_model_parts(remote_model_name)
    remote_folder = remote_model_destination_folder(model_type, remote_model_name, root=root)
    remote_folder_scp = remote_folder.replace("\\", "/").rstrip("/")
    staging_path = posixpath.join(remote_folder_scp, _new_staging_filename())
    final_path = posixpath.join(remote_folder_scp, remote_parts[-1])
    source_size, source_sha256 = (
        _source_integrity(source, check_cancelled)
        if check_cancelled is not None
        else _source_integrity(source)
    )

    LOGGER.info(
        (
            "[Cutlery Remote] Copying missing remote model host=%s model_type=%s "
            "filename=%s size=%s sha256=%s destination=%s"
        ),
        host,
        normalize_model_type(model_type),
        source.name,
        source_size,
        source_sha256,
        remote_folder_scp,
    )
    command_callbacks = {}
    if check_cancelled is not None:
        command_callbacks["check_cancelled"] = check_cancelled
    if process_started is not None:
        command_callbacks["process_started"] = process_started
    if process_finished is not None:
        command_callbacks["process_finished"] = process_finished
    _run_interruptible_command(
        ["ssh", host, _remote_mkdir_command(remote_folder_scp)],
        description="Remote model folder creation",
        stream_to_console=True,
        **command_callbacks,
    )
    staging_may_exist = True
    try:
        _run_interruptible_command(
            ["scp", str(source), f"{host}:{staging_path}"],
            description="Remote model staging copy",
            **command_callbacks,
        )
        _run_interruptible_command(
            [
                "ssh",
                host,
                _remote_verify_and_promote_command(
                    staging_path,
                    final_path,
                    expected_size=source_size,
                    expected_sha256=source_sha256,
                ),
            ],
            description="Remote model integrity verification and promotion",
            stream_to_console=False,
            **command_callbacks,
        )
        staging_may_exist = False
    except BaseException:
        if staging_may_exist:
            _remove_remote_staging_best_effort(host, staging_path)
        raise

    return {
        "ok": True,
        "remote_host": host,
        "remote_folder": remote_folder_scp,
        "remote_model_name": "/".join(remote_parts),
        "size": source_size,
        "sha256": source_sha256,
    }
