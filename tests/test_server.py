"""End-to-end tests against a live server on a throwaway state directory."""
from __future__ import annotations

import io
import json
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from api.common import MonitorError, SettingsStore  # noqa: E402
from api.service import SchedulerService  # noqa: E402
from api.version import APP_VERSION  # noqa: E402
from server import Handler, MonitorInstanceLock, MonitorHTTPServer  # noqa: E402
from tests.support import make_config, write_task  # noqa: E402


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LiveServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        (cls.root / "tasks").mkdir(parents=True, exist_ok=True)
        cls.config = make_config(cls.root)
        cls.config["automation"]["paused"] = True
        cls.service = SchedulerService(cls.config)
        cls.service.settings = SettingsStore(cls.root / ".state" / "settings.json")
        cls.port = free_port()
        cls.server = MonitorHTTPServer(("127.0.0.1", cls.port), Handler, cls.service)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.service.stop()
        cls.server.shutdown()
        cls.server.server_close()
        cls._tmp.cleanup()

    def setUp(self) -> None:
        write_task(self.root, "gb-live-20260920", status="running")

    def _get(self, path: str):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def _post(self, path: str, payload: dict):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_health_reports_hub_stats(self):
        status, data = self._get("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["version"], APP_VERSION)
        self.assertIn("gitCommit", data)
        self.assertIn("hub", data["stats"])

    def test_health_turns_false_when_a_loop_stalls(self):
        health = self.service.health
        health.register("test-loop", 1)
        try:
            health._loops["test-loop"]["lastBeatAt"] -= 10_000
            _status, data = self._get("/api/health")
            self.assertFalse(data["ok"])
            stalled = [loop for loop in data["stats"]["loops"] if loop["name"] == "test-loop"]
            self.assertEqual(stalled[0]["status"], "stalled")
        finally:
            health._loops.pop("test-loop", None)
        _status, data = self._get("/api/health")
        self.assertTrue(data["ok"])

    def test_logs_limit_is_clamped(self):
        with mock.patch.object(self.service.log, "after", wraps=self.service.log.after) as after:
            status, _data = self._get("/api/logs?limit=999999999")
        self.assertEqual(status, 200)
        self.assertEqual(after.call_args.kwargs["limit"], 2000)

    def test_only_failed_requests_are_access_logged(self):
        with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            self._get("/api/version")
            with self.assertRaises(urllib.error.HTTPError):
                self._get("/api/does-not-exist")
        output = stderr.getvalue()
        self.assertNotIn("/api/version", output)
        self.assertIn("/api/does-not-exist", output)

    def test_version_endpoint_reports_release_metadata(self):
        status, data = self._get("/api/version")
        self.assertEqual(status, 200)
        self.assertEqual(data["service"], "sologsb-monitor")
        self.assertEqual(data["version"], APP_VERSION)
        self.assertEqual(data["displayVersion"], f"v{APP_VERSION}")
        self.assertIn("gitCommit", data)
        self.assertIn("gitBranch", data)

    def test_task_list_is_slim_and_detail_is_heavy(self):
        status, data = self._get("/api/tasks")
        self.assertEqual(status, 200)
        self.assertEqual(len(data["tasks"]), 1)
        card = data["tasks"][0]
        self.assertNotIn("promptText", card)
        self.assertNotIn("sides", card)

        status, detail = self._get(f"/api/tasks/{card['id']}")
        self.assertEqual(status, 200)
        self.assertIn("sides", detail)
        self.assertIn("workflow", detail)
        self.assertEqual(detail["promptText"], "题目提示词")

    def test_task_log_reads_from_the_tail(self):
        card = self._get("/api/tasks")[1]["tasks"][0]
        status, data = self._get(f"/api/tasks/{card['id']}/log?side=A&lines=10")
        self.assertEqual(status, 200)
        self.assertIn("lines", data)

    def test_settings_round_trip_without_leaking_credentials(self):
        status, data = self._post("/api/settings", {"settings": {"ui": {"density": "compact"}}})
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertEqual(data["settings"]["ui"]["density"], "compact")
        # The saved marker is read live from the Keychain.
        with mock.patch("api.service.keychain_read", return_value=""):
            status, data = self._get("/api/settings")
        self.assertEqual(status, 200)
        # Only the saved/unsaved marker may come back, never the secret itself.
        self.assertEqual(data["settings"]["manager"]["passwordSaved"], False)
        with mock.patch("api.service.keychain_read", return_value="s3cret"):
            status, data = self._get("/api/settings")
        self.assertTrue(data["settings"]["manager"]["passwordSaved"])
        self.assertNotIn("s3cret", json.dumps(data))
        self.assertNotIn("managerPassword", json.dumps(data["settings"]))

    def test_settings_answer_without_probing_the_manager(self):
        # The login probe is slow; the settings form must not wait on it.
        status, data = self._get("/api/settings")
        self.assertEqual(status, 200)
        self.assertNotIn("connection", data)
        status, data = self._get("/api/settings/connection")
        self.assertEqual(status, 200)
        self.assertIn("ok", data["connection"])

    def test_unknown_folder_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError):
            self._post("/api/settings", {"settings": {"defaultFolderId": "not-a-real-folder"}})

    def test_snapshot_payload_is_small(self):
        status, data = self._get("/api/snapshot")
        self.assertEqual(status, 200)
        self.assertIn("tasks", data)
        self.assertIn("queue", data)
        self.assertIn("containers", data)
        self.assertNotIn("scheduleMode", data["queue"])
        self.assertIn("maxTasks", data["queue"])

    def test_task_events_are_structured(self):
        card = self._get("/api/tasks")[1]["tasks"][0]
        status, data = self._get(f"/api/tasks/{card['id']}/events?side=A&limit=50")
        self.assertEqual(status, 200)
        self.assertIn("events", data)
        self.assertIn("path", data)
        for event in data["events"]:
            self.assertIn("kind", event)
            self.assertIn("title", event)

    def test_logs_endpoint_returns_entries(self):
        status, data = self._get("/api/logs?limit=50")
        self.assertEqual(status, 200)
        self.assertTrue(any(entry["event"] == "service.started" for entry in data["entries"]))

    def test_automation_mode_switch_is_gone(self):
        with self.assertRaises(urllib.error.HTTPError):
            self._post("/api/automation", {"action": "set-schedule-mode", "mode": "tasks"})

    def test_parallel_resume_is_gone(self):
        """There is no "run both sides at once" action any more.

        Validated against a real task so the assertion proves the rejection came
        from the side/mode check rather than from a missing task.
        """
        card = self._get("/api/tasks")[1]["tasks"][0]
        self.assertIsNotNone(card)
        for side, mode, expected in (
            ("BOTH", "both", "side 只能为 A 或 B"),
            ("A", "both", "mode 只能为 resume 或 rerun"),
            ("both", "resume", "side 只能为 A 或 B"),
        ):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self._post("/api/action", {"taskId": card["id"], "side": side, "mode": mode})
            self.assertEqual(ctx.exception.code, 400)
            body = json.loads(ctx.exception.read().decode("utf-8"))
            self.assertEqual(body["error"], expected)

    def test_sse_stream_delivers_a_snapshot(self):
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/stream?logs=1")
        with urllib.request.urlopen(request, timeout=15) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("text/event-stream", response.headers.get("Content-Type", ""))
            buffer = b""
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                chunk = response.read(1024)
                if not chunk:
                    break
                buffer += chunk
                if b"event: snapshot" in buffer:
                    break
        self.assertIn(b"event: snapshot", buffer)
        self.assertIn(b'"tasks"', buffer)


class PauseOnStartTests(unittest.TestCase):
    """What a process start does with a queue that was running."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.root / "tasks").mkdir(parents=True, exist_ok=True)

    def _service(self, *, stop=True, **automation):
        config = make_config(self.root)
        config["automation"].update(automation)
        service = SchedulerService(config)
        if stop:
            self.addCleanup(service.stop)
        return service, config

    def _crash(self, **automation):
        """A start whose process dies without SchedulerService.stop()."""
        service, _config = self._service(stop=False, **automation)
        service.log.close()
        return service

    def _saved(self):
        return json.loads((self.root / "config.json").read_text(encoding="utf-8"))

    def _events(self):
        return [json.loads(line)["event"] for line in
                (self.root / ".state" / "scheduler.jsonl").read_text(encoding="utf-8").splitlines()]

    def test_default_resumes_a_queue_that_was_running(self):
        service, config = self._service(paused=False)
        self.assertFalse(config["automation"]["paused"])
        self.assertFalse(service.queue.snapshot()["paused"])
        self.assertIn("config.resumed_on_start", self._events())

    def test_one_crash_restart_still_resumes(self):
        self._crash(paused=False)
        service, config = self._service(paused=False)
        self.assertFalse(config["automation"]["paused"])

    def test_crash_loop_pauses_and_persists(self):
        self._crash(paused=False)
        self._crash(paused=False)
        service, config = self._service(paused=False)
        self.assertTrue(config["automation"]["paused"])
        self.assertIn("异常退出 2 次", service._pause_reason)
        self.assertTrue(self._saved()["automation"]["paused"])
        self.assertIn("config.paused_on_start", self._events())

    def test_clean_restarts_are_not_a_crash_loop(self):
        for _ in range(4):
            service, _config = self._service(stop=False, paused=False)
            service.stop()
        service, config = self._service(paused=False)
        self.assertFalse(config["automation"]["paused"])

    def test_old_crashes_age_out(self):
        self._crash(paused=False)
        self._crash(paused=False)
        path = self.root / ".state" / "starts.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data["starts"]:
            entry["at"] -= 3600
        path.write_text(json.dumps(data), encoding="utf-8")
        _service, config = self._service(paused=False)
        self.assertFalse(config["automation"]["paused"])

    def test_always_pauses_and_legacy_true_means_always(self):
        for value in ("always", True):
            with self.subTest(value=value):
                service, config = self._service(paused=False, pauseOnStart=value)
                self.assertTrue(config["automation"]["paused"])
                self.assertEqual(service.queue.snapshot()["pauseOnStart"], "always")

    def test_never_resumes_even_in_a_crash_loop(self):
        self._crash(paused=False, pauseOnStart="never")
        self._crash(paused=False, pauseOnStart="never")
        _service, config = self._service(paused=False, pauseOnStart=False)
        self.assertFalse(config["automation"]["paused"])

    def test_already_paused_queue_is_not_rewritten(self):
        service, config = self._service(paused=True)
        self.assertTrue(config["automation"]["paused"])
        self.assertFalse(service._paused_on_start)
        self.assertFalse((self.root / "config.json").exists())


class ManualEnqueueGateTests(unittest.TestCase):
    """queue-add-platform must refuse projects the skill would refuse at init."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.root / "tasks").mkdir(parents=True, exist_ok=True)
        self.service = SchedulerService(make_config(self.root))
        self.addCleanup(self.service.stop)

    def _add(self, code: str):
        return self.service.automation_action(
            "queue-add-platform", {"project": {"id": "p-1", "code": code, "name": "项目"}}
        )

    def test_claimed_project_is_rejected(self):
        with mock.patch.object(self.service.platform, "occupied_project_codes", return_value={"gb-14-1"}):
            with self.assertRaises(MonitorError) as raised:
                self._add("GB-14-1")
        self.assertIn("仍被占用", str(raised.exception))
        self.assertEqual(self.service.queue.snapshot()["items"], [])

    def test_free_project_is_queued(self):
        with mock.patch.object(self.service.platform, "occupied_project_codes", return_value={"gb-15-1"}):
            self._add("gb-14-1")
        codes = [item.get("projectCode") for item in self.service.queue.snapshot()["items"]]
        self.assertEqual(codes, ["gb-14-1"])

    def test_unreadable_claims_block_enqueue(self):
        with mock.patch.object(self.service.platform, "occupied_project_codes", side_effect=OSError("boom")):
            with self.assertRaises(MonitorError):
                self._add("gb-14-1")


class SetLimitsTests(unittest.TestCase):
    """A limit saved on the page applies at once."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.root / "tasks").mkdir(parents=True, exist_ok=True)
        config = make_config(self.root)
        config["_configPath"] = str(self.root / "config.json")
        config["automation"].update({"maxContainers": 5, "capacity": 2})
        self.service = SchedulerService(config)
        self.addCleanup(self.service.stop)

    def _save(self, **payload):
        form = {"maxTasks": 3, "maxContainers": 5, "candidatesPerTask": 2, "cooldownSeconds": 210}
        return self.service.automation_action("set-limits", {**form, **payload})

    def test_saved_limits_apply_at_once(self):
        queue = self.service.queue
        snapshot = self._save(maxTasks=4, maxContainers=6)
        self.assertEqual(snapshot["maxTasks"], 4)
        self.assertEqual(snapshot["maxContainers"], 6)
        self.assertEqual(queue._max_tasks(), 4)
        self.assertEqual(json.loads(queue.slots.limit_path.read_text(encoding="utf-8"))["maxContainers"], 6)

    def test_legacy_knobs_are_dropped_on_save(self):
        self.service.config["automation"]["elasticContainers"] = {"enabled": True}
        self.service.config["automation"]["maxActiveTasks"] = 7
        self.service.config["automation"]["waitTimeoutSeconds"] = 50400
        self._save()
        self.assertNotIn("elasticContainers", self.service.config["automation"])
        self.assertNotIn("maxActiveTasks", self.service.config["automation"])
        self.assertNotIn("waitTimeoutSeconds", self.service.config["automation"])


class BackgroundLoopTests(unittest.TestCase):
    """The launch tick must not queue behind the slow Manager refill.

    A refill can spend minutes in the Keychain and in paginated Manager HTTP.
    While it shared the auto loop, a gate that had already opened (disk space
    freed, a container slot released) was only re-checked once the refill
    finished, so the queue sat on "磁盘不足" with plenty of free space.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        (self.root / "tasks").mkdir(parents=True, exist_ok=True)
        self.service = SchedulerService(make_config(self.root))
        self.addCleanup(self.service.stop)

    def test_tick_keeps_running_while_the_refill_is_blocked(self):
        service = self.service
        service._auto_interval = 0.05
        service._tick_interval = 0.05
        service.health.register("auto-loop", 0.05)
        service.health.register("queue-tick", 0.05)
        refill_started = threading.Event()
        release_refill = threading.Event()
        ticks: list[dict] = []

        def blocked_refill():
            refill_started.set()
            release_refill.wait(10)
            return {"status": "blocked"}

        def counting_tick(**kwargs):
            ticks.append(kwargs)
            return []

        with mock.patch.object(service, "maybe_refill_queue", side_effect=blocked_refill), \
                mock.patch.object(service.queue, "tick", side_effect=counting_tick):
            service._start_tick_loop()
            service._start_auto_loop()
            try:
                self.assertTrue(refill_started.wait(5), "补队没有开始")
                deadline = time.monotonic() + 5
                while not ticks and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ticks, "补队卡住时启动门禁没有继续跑")
            finally:
                release_refill.set()
                service._stop.set()

    def test_auto_iteration_no_longer_runs_the_tick(self):
        service = self.service
        calls: list[dict] = []
        with mock.patch.object(service.queue, "tick", side_effect=lambda **kwargs: calls.append(kwargs)), \
                mock.patch.object(service, "enforce_stop_tasks", return_value=[]), \
                mock.patch.object(service, "maybe_refill_queue", return_value={"status": "disabled"}), \
                mock.patch.object(service, "maybe_auto_resume", return_value=[]):
            service._auto_iteration([])
        self.assertEqual(calls, [])


class LockTests(unittest.TestCase):
    def test_rejects_second_monitor_instance_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "monitor.lock"
            first = MonitorInstanceLock(path)
            second = MonitorInstanceLock(path)
            first.acquire()
            try:
                with self.assertRaises(RuntimeError):
                    second.acquire()
            finally:
                first.release()
            second.acquire()
            second.release()


if __name__ == "__main__":
    unittest.main()
