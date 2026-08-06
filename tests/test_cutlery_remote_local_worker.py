import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutlery_remote.local_worker import LocalWorkerManager, _WorkerState  # noqa: E402
from cutlery_remote.target import TrustedRemoteTarget  # noqa: E402


class LocalWorkerManagerTests(unittest.TestCase):
    def test_worker_stays_up_while_leased_then_uses_configured_idle_timeout(self):
        target = TrustedRemoteTarget(
            name="trellis2",
            base_url="http://127.0.0.1:8890",
            canonical="cutlery://trellis2",
            display_label="TRELLIS.2",
            worker_python="C:/Comfy/trellis/python.exe",
            worker_comfy_root="C:/ComfyUI",
            worker_idle_seconds=600,
        )
        manager = LocalWorkerManager()
        process = mock.Mock()
        process.poll.return_value = None

        def fake_start(_target, state):
            state.process = process

        timer = mock.Mock()
        with (
            mock.patch.object(manager, "_is_listening", side_effect=[False, True]),
            mock.patch.object(manager, "_start", side_effect=fake_start),
            mock.patch("cutlery_remote.local_worker.threading.Timer", return_value=timer) as timer_class,
        ):
            first = manager.acquire(target)
            second = manager.acquire(target)
            first.release()
            timer_class.assert_not_called()
            second.release()

        timer_class.assert_called_once_with(600, manager._stop_if_idle, args=("trellis2", mock.ANY))
        timer.start.assert_called_once_with()

    def test_status_reports_worker_lease_idle_and_process_telemetry_without_config_paths(self):
        target = TrustedRemoteTarget(
            name="trellis2",
            base_url="http://127.0.0.1:8890",
            canonical="cutlery://trellis2",
            display_label="TRELLIS.2",
            worker_python="C:/Comfy/trellis/python.exe",
            worker_comfy_root="C:/ComfyUI",
            worker_idle_seconds=600,
        )
        manager = LocalWorkerManager()
        process = mock.Mock(pid=4321)
        process.poll.return_value = None

        def fake_start(_target, state):
            state.process = process

        with (
            mock.patch.object(manager, "_is_listening", return_value=False),
            mock.patch.object(manager, "_start", side_effect=fake_start),
            mock.patch.object(manager, "_process_tree_pids", return_value=[4321, 4322]),
            mock.patch.object(manager, "_process_tree_rss_bytes", return_value=1234),
            mock.patch("cutlery_remote.local_worker.threading.Timer", return_value=mock.Mock()),
            mock.patch(
                "cutlery_remote.local_worker.get_nvidia_process_memory",
                return_value={"source": "nvidia-smi", "used_bytes": None, "error": "nvidia-smi-process-memory-unavailable-wddm"},
            ),
        ):
            lease = manager.acquire(target)
            status = manager.status()
            lease.release()

        worker = status["workers"][0]
        self.assertEqual(worker["name"], "trellis2")
        self.assertEqual(worker["pid"], 4321)
        self.assertEqual(worker["process_pids"], [4321, 4322])
        self.assertEqual(worker["active_leases"], 1)
        self.assertFalse(worker["idle"])
        self.assertEqual(worker["ram_used_bytes"], 1234)
        self.assertIsNone(worker["vram_used_bytes"])
        self.assertEqual(worker["vram_error"], "nvidia-smi-process-memory-unavailable-wddm")
        self.assertNotIn("worker_python", worker)
        self.assertNotIn("worker_comfy_root", worker)

    def test_stop_all_force_stops_active_owned_workers(self):
        manager = LocalWorkerManager()
        state = manager._states.setdefault("trellis2", _WorkerState())
        state.process = mock.Mock()
        state.process.poll.return_value = None
        state.active_leases = 1
        state.shutdown_timer = None
        state.idle_deadline = None
        state.log_handle = None

        with mock.patch.object(manager, "_stop_state") as stop_state:
            result = manager.stop_all()

        stop_state.assert_called_once_with("trellis2", state)
        self.assertEqual(result, {"stopped_targets": ["trellis2"], "active_targets": ["trellis2"], "forced": True})


if __name__ == "__main__":
    unittest.main()
