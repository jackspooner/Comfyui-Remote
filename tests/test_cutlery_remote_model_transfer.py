import base64
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class RemoteModelTransferTests(unittest.TestCase):
    @staticmethod
    def _decode_powershell_command(command):
        encoded_script = command.rsplit(" ", 1)[-1]
        return base64.b64decode(encoded_script).decode("utf-16le")

    def test_destination_folder_preserves_category_and_model_subfolder(self):
        from cutlery_remote.model_transfer import remote_model_destination_folder

        self.assertEqual(
            remote_model_destination_folder("text_encoder", "sd3/clip_l.safetensors", root="D:/ComfyUI/models"),
            "D:/ComfyUI/models/text_encoders/sd3",
        )

    def test_transfer_commands_do_not_inherit_comfyui_stdin(self):
        from cutlery_remote import model_transfer

        process = mock.Mock()
        process.poll.return_value = 0
        process.communicate.return_value = ("", None)
        process.returncode = 0
        completed = mock.Mock(returncode=0, stdout="")

        with mock.patch.object(model_transfer.subprocess, "Popen", return_value=process) as popen:
            model_transfer._run_interruptible_command(["ssh", "renderhost", "exit"], description="test")
        with mock.patch.object(model_transfer.subprocess, "run", return_value=completed) as run:
            model_transfer._run_uninterruptible_cleanup_command(["ssh", "renderhost", "exit"], description="test")

        self.assertIs(popen.call_args.kwargs["stdin"], model_transfer.subprocess.DEVNULL)
        self.assertIs(run.call_args.kwargs["stdin"], model_transfer.subprocess.DEVNULL)

    def test_transfer_failure_includes_captured_scp_output(self):
        from cutlery_remote import model_transfer

        process = mock.Mock()
        process.communicate.return_value = ("scp: Permission denied", None)
        process.returncode = 1

        with mock.patch.object(model_transfer.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "scp: Permission denied"):
                model_transfer._run_interruptible_command(
                    ["scp", "model.safetensors", "renderhost:D:/models/model.safetensors"],
                    description="Remote model staging copy",
                )

    def test_command_log_omits_encoded_powershell_payload(self):
        from cutlery_remote import model_transfer

        command = model_transfer._format_command(
            ["ssh", "renderhost", "powershell.exe", "-EncodedCommand", "very-long-payload"]
        )

        self.assertIn("<encoded PowerShell omitted>", command)
        self.assertNotIn("very-long-payload", command)

    def test_command_log_omits_multi_line_posix_script(self):
        from cutlery_remote import model_transfer

        script = model_transfer._posix_verify_and_promote_command(
            "/home/jack/models/vae/.cutlery-upload-x.part",
            "/home/jack/models/vae/vae.safetensors",
            expected_size=3,
            expected_sha256=hashlib.sha256(b"vae").hexdigest(),
        )
        formatted = model_transfer._format_command(["ssh", "secondpc", script])

        self.assertIn("<multi-line peer script omitted>", formatted)
        self.assertNotIn("sha256_of", formatted)

    def test_copy_model_file_uses_configured_ssh_mkdir_and_scp_folder_format(self):
        from cutlery_remote import model_transfer

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "vae.safetensors"
            source.write_bytes(b"vae")
            calls = []

            def fake_run(command, *, description, timeout_interval=0.5, stream_to_console=False):
                calls.append((list(command), description, stream_to_console))
                return ""

            configured = {
                model_transfer.REMOTE_MODEL_COPY_HOST_ENV: "renderhost",
                model_transfer.REMOTE_MODEL_COPY_ROOT_ENV: "D:/ComfyUI/models",
            }
            with mock.patch.object(
                model_transfer,
                "env_value",
                side_effect=lambda name, default="": configured.get(name, default),
            ), mock.patch.object(
                model_transfer,
                "_run_interruptible_command",
                side_effect=fake_run,
            ), mock.patch.object(
                model_transfer,
                "_new_staging_filename",
                return_value=".cutlery-upload-fixed.part",
            ), mock.patch.object(
                model_transfer,
                "remote_shell_kind",
                return_value="windows",
            ):
                result = model_transfer.copy_model_file_to_remote(source, "vae", "sdxl/vae.safetensors")

        self.assertEqual(
            calls[0][0],
            [
                "ssh",
                "renderhost",
                'cmd.exe /d /c if not exist "D:\\ComfyUI\\models\\vae\\sdxl" mkdir "D:\\ComfyUI\\models\\vae\\sdxl"',
            ],
        )
        self.assertEqual(
            calls[1][0],
            [
                "scp",
                str(source),
                "renderhost:D:/ComfyUI/models/vae/sdxl/.cutlery-upload-fixed.part",
            ],
        )
        self.assertEqual(calls[2][0][0:2], ["ssh", "renderhost"])
        self.assertIn("-EncodedCommand", calls[2][0][2])
        verification_command = self._decode_powershell_command(calls[2][0][2])
        self.assertIn("Get-FileHash", verification_command)
        self.assertIn("[System.IO.File]::Move($stage, $final)", verification_command)
        self.assertIn("$expectedSize=3", verification_command)
        self.assertIn("$actualSize -ne $expectedSize", verification_command)
        self.assertIn(hashlib.sha256(b"vae").hexdigest(), verification_command)
        self.assertIn("D:\\ComfyUI\\models\\vae\\sdxl\\vae.safetensors", verification_command)
        self.assertEqual([call[2] for call in calls], [True, False, False])
        self.assertTrue(result["ok"])
        self.assertEqual(result["remote_model_name"], "sdxl/vae.safetensors")
        self.assertEqual(result["size"], 3)
        self.assertEqual(result["sha256"], hashlib.sha256(b"vae").hexdigest())

    def test_copy_model_file_uses_target_specific_host_and_root(self):
        from cutlery_remote import model_transfer

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "clip.safetensors"
            source.write_bytes(b"clip")
            calls = []

            with mock.patch.object(
                model_transfer,
                "_run_interruptible_command",
                side_effect=lambda command, **_kwargs: calls.append(list(command)) or "",
            ), mock.patch.object(
                model_transfer,
                "_new_staging_filename",
                return_value=".cutlery-upload-target.part",
            ), mock.patch.object(
                model_transfer,
                "remote_shell_kind",
                return_value="windows",
            ):
                result = model_transfer.copy_model_file_to_remote(
                    source,
                    "text_encoders",
                    "flux/clip.safetensors",
                    remote_host="studio-gpu",
                    remote_root="E:/Comfy/models",
                )

        self.assertEqual(calls[0][0:2], ["ssh", "studio-gpu"])
        self.assertIn("E:\\Comfy\\models\\text_encoders\\flux", calls[0][2])
        self.assertEqual(
            calls[1],
            [
                "scp",
                str(source),
                "studio-gpu:E:/Comfy/models/text_encoders/flux/.cutlery-upload-target.part",
            ],
        )
        self.assertEqual(calls[2][0:2], ["ssh", "studio-gpu"])
        verification_command = self._decode_powershell_command(calls[2][2])
        self.assertIn("E:\\Comfy\\models\\text_encoders\\flux\\clip.safetensors", verification_command)
        self.assertEqual(result["remote_host"], "studio-gpu")

    def test_copy_failure_removes_staging_file_without_promoting(self):
        from cutlery_remote import model_transfer

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "model.safetensors"
            source.write_bytes(b"model")
            calls = []

            def fake_run(command, **_kwargs):
                calls.append(list(command))
                if len(calls) == 3:
                    raise RuntimeError("remote hash mismatch")
                return ""

            with mock.patch.object(
                model_transfer,
                "_run_interruptible_command",
                side_effect=fake_run,
            ), mock.patch.object(
                model_transfer,
                "_new_staging_filename",
                return_value=".cutlery-upload-failed.part",
            ), mock.patch.object(
                model_transfer,
                "remote_shell_kind",
                return_value="windows",
            ), mock.patch.object(
                model_transfer,
                "_remove_remote_staging_best_effort",
            ) as cleanup:
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    model_transfer.copy_model_file_to_remote(
                        source,
                        "checkpoints",
                        "model.safetensors",
                        remote_host="renderhost",
                        remote_root="D:/ComfyUI/models",
                    )

        cleanup.assert_called_once_with(
            "renderhost",
            "D:/ComfyUI/models/checkpoints/.cutlery-upload-failed.part",
            "windows",
        )
        verification_command = self._decode_powershell_command(calls[2][2])
        self.assertIn("[System.IO.File]::Move($stage, $final)", verification_command)
        self.assertIn("$acceptExisting", verification_command)
        self.assertIn("$finalSize -ne $expectedSize", verification_command)
        self.assertIn("$finalHash -ne $expectedHash", verification_command)
        self.assertIn("[System.IO.File]::Delete($stage)", verification_command)
        self.assertIn("catch [System.IO.IOException]", verification_command)

    def test_copy_cancellation_during_scp_removes_partial_staging_file(self):
        from cutlery_remote import model_transfer

        class Cancelled(BaseException):
            pass

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "model.safetensors"
            source.write_bytes(b"model")
            calls = []

            def fake_run(command, **_kwargs):
                calls.append(list(command))
                if command[0] == "scp":
                    raise Cancelled()
                return ""

            with mock.patch.object(
                model_transfer,
                "_run_interruptible_command",
                side_effect=fake_run,
            ), mock.patch.object(
                model_transfer,
                "_new_staging_filename",
                return_value=".cutlery-upload-cancelled.part",
            ), mock.patch.object(
                model_transfer,
                "remote_shell_kind",
                return_value="windows",
            ), mock.patch.object(
                model_transfer,
                "_remove_remote_staging_best_effort",
            ) as cleanup:
                with self.assertRaises(Cancelled):
                    model_transfer.copy_model_file_to_remote(
                        source,
                        "checkpoints",
                        "model.safetensors",
                        remote_host="renderhost",
                        remote_root="D:/ComfyUI/models",
                    )

        cleanup.assert_called_once_with(
            "renderhost",
            "D:/ComfyUI/models/checkpoints/.cutlery-upload-cancelled.part",
            "windows",
        )
        self.assertEqual(len(calls), 2)

    def test_cleanup_failure_does_not_mask_copy_cancellation(self):
        from cutlery_remote import model_transfer

        class Cancelled(BaseException):
            pass

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "model.safetensors"
            source.write_bytes(b"model")

            def fake_run(command, **_kwargs):
                if command[0] == "scp":
                    raise Cancelled()
                return ""

            with mock.patch.object(
                model_transfer,
                "_run_interruptible_command",
                side_effect=fake_run,
            ), mock.patch.object(
                model_transfer,
                "_new_staging_filename",
                return_value=".cutlery-upload-cleanup-failure.part",
            ), mock.patch.object(
                model_transfer,
                "remote_shell_kind",
                return_value="windows",
            ), mock.patch.object(
                model_transfer,
                "_run_uninterruptible_cleanup_command",
                side_effect=RuntimeError("cleanup unavailable"),
            ), mock.patch.object(
                model_transfer.LOGGER,
                "warning",
            ) as warning:
                with self.assertRaises(Cancelled):
                    model_transfer.copy_model_file_to_remote(
                        source,
                        "checkpoints",
                        "model.safetensors",
                        remote_host="renderhost",
                        remote_root="D:/ComfyUI/models",
                    )

        warning.assert_called_once()

    def test_staging_names_are_unique_and_reserved(self):
        from cutlery_remote import model_transfer
        from cutlery_remote.inventory import is_model_transfer_staging_name

        first = model_transfer._new_staging_filename()
        second = model_transfer._new_staging_filename()

        self.assertNotEqual(first, second)
        self.assertTrue(is_model_transfer_staging_name(first))
        self.assertTrue(is_model_transfer_staging_name(f"subdir/{second}"))

    def test_remote_shell_kind_detects_posix_peers(self):
        from cutlery_remote import model_transfer

        for system in ("Linux\n", "Darwin\n", "FreeBSD\n"):
            with mock.patch.object(
                model_transfer.subprocess,
                "run",
                return_value=mock.Mock(returncode=0, stdout=system),
            ):
                self.assertEqual(model_transfer.remote_shell_kind("peer"), "posix")

    def test_remote_shell_kind_treats_git_bash_and_failures_as_windows(self):
        from cutlery_remote import model_transfer

        for completed in (
            mock.Mock(returncode=0, stdout="MINGW64_NT-10.0-19045\n"),
            mock.Mock(returncode=1, stdout="'uname' is not recognized"),
        ):
            with mock.patch.object(model_transfer.subprocess, "run", return_value=completed):
                self.assertEqual(model_transfer.remote_shell_kind("peer"), "windows")

    def test_copy_model_file_uses_posix_commands_for_linux_peers(self):
        from cutlery_remote import model_transfer

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "clip.safetensors"
            source.write_bytes(b"clip")
            calls = []

            with mock.patch.object(
                model_transfer,
                "_run_interruptible_command",
                side_effect=lambda command, **_kwargs: calls.append(list(command)) or "",
            ), mock.patch.object(
                model_transfer,
                "_new_staging_filename",
                return_value=".cutlery-upload-posix.part",
            ), mock.patch.object(
                model_transfer,
                "remote_shell_kind",
                return_value="posix",
            ):
                result = model_transfer.copy_model_file_to_remote(
                    source,
                    "text_encoders",
                    "soulx_singer/clip.safetensors",
                    remote_host="secondpc",
                    remote_root="/home/jack/models",
                )

        self.assertEqual(
            calls[0],
            ["ssh", "secondpc", "mkdir -p -- /home/jack/models/text_encoders/soulx_singer"],
        )
        self.assertEqual(
            calls[1],
            [
                "scp",
                str(source),
                "secondpc:/home/jack/models/text_encoders/soulx_singer/.cutlery-upload-posix.part",
            ],
        )
        self.assertEqual(calls[2][0:2], ["ssh", "secondpc"])
        script = calls[2][2]
        self.assertIn("sha256sum", script)
        self.assertIn('mv -- "$stage" "$final"', script)
        self.assertIn('wc -c < "$stage"', script)
        self.assertIn("/home/jack/models/text_encoders/soulx_singer/clip.safetensors", script)
        self.assertNotIn("Get-FileHash", script)
        self.assertNotIn("powershell", script)
        self.assertNotIn("\\", script)
        self.assertTrue(result["ok"])
        self.assertEqual(result["remote_folder"], "/home/jack/models/text_encoders/soulx_singer")
        self.assertEqual(result["sha256"], hashlib.sha256(b"clip").hexdigest())

    def test_posix_commands_quote_paths_with_spaces_and_quotes(self):
        from cutlery_remote import model_transfer

        folder = "/home/jack/models/vae/my model's dir"
        mkdir = model_transfer._posix_mkdir_command(folder)
        cleanup = model_transfer._posix_remove_staging_command(f"{folder}/.cutlery-upload-x.part")
        script = model_transfer._posix_verify_and_promote_command(
            f"{folder}/.cutlery-upload-x.part",
            f"{folder}/vae's file.safetensors",
            expected_size=7,
            expected_sha256="ab" * 32,
        )

        self.assertTrue(mkdir.startswith("mkdir -p -- "))
        self.assertIn("'", mkdir)
        self.assertTrue(cleanup.startswith("rm -f -- "))
        self.assertIn("'", cleanup)
        self.assertIn("'\"'\"'", script)
        self.assertIn("expected_size=7", script)
        self.assertIn(f"expected_hash={'ab' * 32}", script)

    def test_copy_failure_cleans_up_with_posix_command(self):
        from cutlery_remote import model_transfer

        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "model.safetensors"
            source.write_bytes(b"model")

            def fake_run(command, **_kwargs):
                if command[0] == "scp":
                    raise RuntimeError("scp: No space left on device")
                return ""

            with mock.patch.object(
                model_transfer,
                "_run_interruptible_command",
                side_effect=fake_run,
            ), mock.patch.object(
                model_transfer,
                "_new_staging_filename",
                return_value=".cutlery-upload-nospace.part",
            ), mock.patch.object(
                model_transfer,
                "remote_shell_kind",
                return_value="posix",
            ), mock.patch.object(
                model_transfer,
                "_remove_remote_staging_best_effort",
            ) as cleanup:
                with self.assertRaisesRegex(RuntimeError, "No space left on device"):
                    model_transfer.copy_model_file_to_remote(
                        source,
                        "checkpoints",
                        "model.safetensors",
                        remote_host="secondpc",
                        remote_root="/home/jack/models",
                    )

        cleanup.assert_called_once_with(
            "secondpc",
            "/home/jack/models/checkpoints/.cutlery-upload-nospace.part",
            "posix",
        )


if __name__ == "__main__":
    unittest.main()
