import asyncio
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cutlery_remote.remote_job import (
    RemoteExecutionCallbackError,
    RemoteExecutionJob,
    RemoteExecutionJobError,
    RemoteExecutionJobRegistry,
    RemoteExecutionState,
)


class RemoteExecutionJobTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_runs_every_callback_and_cleans_once(self):
        job = RemoteExecutionJob("local-1", ("remote-1",))
        calls = []

        job.register_abort(lambda _: calls.append("abort"))
        job.register_terminate(lambda _: calls.append("terminate"))

        async def interrupt(_):
            calls.append("peer_interrupt")

        job.register_peer_interrupt(interrupt)
        job.register_cleanup(lambda _: calls.append("cleanup"))

        self.assertTrue(await job.cancel())
        self.assertFalse(await job.cancel())
        self.assertEqual(calls, ["abort", "terminate", "peer_interrupt", "cleanup"])
        self.assertEqual(job.state, RemoteExecutionState.CANCELLED)
        self.assertTrue(job.cancellation_event.is_set())
        self.assertEqual(job.remote_prompt_ids, frozenset({"remote-1"}))

    async def test_callback_failure_does_not_skip_cleanup(self):
        job = RemoteExecutionJob("local-1")
        calls = []

        def bad_abort(_):
            calls.append("abort")
            raise RuntimeError("expected")

        job.register_abort(bad_abort)
        job.register_terminate(lambda _: calls.append("terminate"))
        job.register_cleanup(lambda _: calls.append("cleanup"))

        with self.assertRaises(RemoteExecutionCallbackError) as raised:
            await job.cancel()

        self.assertEqual(raised.exception.phase, "cancel")
        self.assertEqual(calls, ["abort", "terminate", "cleanup"])

    async def test_registry_releases_terminal_jobs_by_local_prompt_id(self):
        registry = RemoteExecutionJobRegistry()
        job = registry.create("local-1", ("remote-1",))

        self.assertIs(registry.get("local-1"), job)
        await job.succeed({"result": "ok"})

        self.assertEqual(job.state, RemoteExecutionState.SUCCEEDED)
        self.assertEqual(job.result, {"result": "ok"})
        self.assertIsNone(registry.get("local-1"))
        with self.assertRaises(RemoteExecutionJobError):
            await job.fail(RuntimeError("late"))

    async def test_cleanup_is_idempotent_under_concurrent_callers(self):
        job = RemoteExecutionJob("local-1")
        calls = []

        async def cleanup(_):
            await asyncio.sleep(0)
            calls.append("cleanup")

        job.register_cleanup(cleanup)
        first, second = await asyncio.gather(job.cleanup(), job.cleanup())

        self.assertEqual((first, second), (True, False))
        self.assertEqual(calls, ["cleanup"])


if __name__ == "__main__":
    unittest.main()
