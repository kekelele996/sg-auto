"""Fallback policies: candidate timeout, no progress, leaked processes."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from api.common import MonitorError, iso_from_timestamp  # noqa: E402
from api.guard import TaskGuard, guard_settings, worker_wait_timeout_seconds  # noqa: E402
from api.service import SchedulerService  # noqa: E402
from tests.support import SchedulerTestCase, write_task  # noqa: E402
from tests.test_scheduler import build_queue, platform_item  # noqa: E402

HOUR = 3600


class _Log:
    def __init__(self):
        self.events = []

    def emit(self, event, **fields):
        self.events.append((event, fields))

    def names(self):
        return [event for event, _ in self.events]


class GuardTestCase(SchedulerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.now = time.time()
        self.processes: list[dict] = []
        self.killed: list[int] = []
        self.removed: list[str] = []
        self.log = _Log()
        self.stop_path = self.root / "skill-stop-tasks.json"

    def _guard(self, queue, mode="observe", **settings):
        self.config["automation"]["guard"] = {"mode": mode, **settings}
        return TaskGuard(
            queue, log=self.log, stop_path=self.stop_path,
            processes=lambda: list(self.processes),
            kill=lambda process: self.killed.append(process["pid"]) or True,
            remove=lambda prefix: self.removed.append(prefix) or [f"{prefix}candidate-1"],
            clock=lambda: self.now,
        )

    def _task(self, name, *, status="candidates_running", race_age=None, race_done=False, quiet=0.0,
              post_age=None):
        task_root = write_task(self.root, name, status=status, sides={
            "A": {"status": "running", "runPid": 999999}, "B": {"status": "idle"},
        })
        state_path = task_root / "monitor" / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if race_age is not None:
            state["candidateRaceStartedAt"] = iso_from_timestamp(self.now - race_age)
        if race_done:
            state["candidateRaceFinishedAt"] = iso_from_timestamp(self.now - (post_age or 60))
        state["candidates"] = {"candidate-1": {"status": "running"}}
        state_path.write_text(json.dumps(state), encoding="utf-8")
        self._age(task_root, quiet)
        return task_root

    @staticmethod
    def _age(task_root, seconds):
        stamp = time.time() - seconds
        for current, _dirs, files in os.walk(task_root):
            for name in files:
                os.utime(Path(current) / name, (stamp, stamp))

    def _queue(self, task_root, **overrides):
        item = platform_item(**{"status": "running", "taskRoot": str(task_root), "capacityHeld": True,
                                "quota": {"state": "claimed"}, **overrides})
        return build_queue(self.config, items=[item])

    def _state(self, task_root):
        return json.loads((task_root / "monitor" / "state.json").read_text(encoding="utf-8"))


class CandidateTimeoutTests(GuardTestCase):
    def test_observe_flags_once_and_touches_nothing(self):
        task_root = self._task("gb-1-slow", race_age=7 * HOUR)
        queue = self._queue(task_root)
        guard = self._guard(queue)
        self.processes = [{"pid": 4242, "pgid": 4242,
                           "command": f"python3 sologsb.py run --task-root {task_root} --side both"}]

        guard.run_once()
        guard.run_once()

        item = queue._items[0]
        self.assertEqual(item["status"], "running")
        self.assertEqual(item["guardFlag"]["policy"], "candidate-timeout")
        self.assertEqual(queue.guard_status["flags"][0]["itemId"], item["id"])
        self.assertEqual(self.log.names().count("guard.would_stop"), 1)
        self.assertEqual((self.killed, self.removed), ([], []))
        self.assertFalse(self.stop_path.exists())
        self.assertEqual(self._state(task_root)["status"], "candidates_running")

    def test_enforce_stops_the_task_and_keeps_its_directory(self):
        task_root = self._task("gb-1-slow", race_age=7 * HOUR)
        queue = self._queue(task_root)
        self.stop_path.write_text(json.dumps({"tasks": ["cy-306-old"]}), encoding="utf-8")
        guard = self._guard(queue, mode="enforce")
        self.processes = [
            {"pid": 4242, "pgid": 4242, "command": f"python3 sologsb.py run --task-root {task_root} --side both"},
            {"pid": 4243, "pgid": 4100, "command": f"node {task_root}/source/candidates/candidate-1/node_modules/.bin/vite"},
            {"pid": 4244, "pgid": 4244, "command": f"/opt/python3 queue_worker.py --task-name {task_root.name} --workdir {task_root.parent}"},
            {"pid": 4245, "pgid": 4245, "command": f"/usr/bin/login -qflp me /bin/zsh -fc cd '{task_root}/source'"},
            {"pid": 4246, "pgid": 4246, "command": f"python3 sologsb.py run --task-root {task_root}-other --side A"},
        ]

        actions = guard.run_once()

        self.assertEqual(actions[0]["kind"], "guard-stopped")
        self.assertEqual(sorted(self.killed), [4242, 4243])
        self.assertEqual(self.removed, [f"sologsb-{task_root.name}-"])
        stop = json.loads(self.stop_path.read_text(encoding="utf-8"))
        self.assertEqual(stop["tasks"], ["cy-306-old", task_root.name])
        self.assertEqual(stop["guard"][task_root.name]["policy"], "candidate-timeout")
        state = self._state(task_root)
        self.assertEqual((state["status"], state["previousStatus"], state["closedPolicy"]),
                         ("failed", "candidates_running", "candidate-timeout"))
        self.assertEqual(state["candidates"]["candidate-1"]["status"], "invalidated")
        self.assertTrue(task_root.is_dir())
        item = queue._items[0]
        self.assertEqual(item["status"], "failed")
        self.assertFalse(item["capacityHeld"])
        self.assertIn("候选阶段超时", item["error"])
        self.assertEqual(item["guardStopped"]["killedProcesses"], 2)
        self.assertEqual(item["quota"]["state"], "claimed")  # no platform attached in this test
        self.assertIn("guard.stopped", self.log.names())

    def test_enforce_refunds_the_quota(self):
        task_root = self._task("gb-1-slow", race_age=7 * HOUR)
        queue = self._queue(task_root)
        guard = self._guard(queue, mode="enforce")

        class _Platform:
            released = []

            def release_task(self, task_id):
                self.released.append(task_id)
                return {"ok": False, "mode": "local"}

        guard.platform = _Platform()
        queue._items[0]["quota"] = {"state": "claimed", "platformTaskId": "t-9"}
        guard.run_once()
        self.assertEqual(queue._items[0]["quota"]["state"], "refunded")
        self.assertEqual(_Platform.released, ["t-9"])

    def test_race_within_the_limit_is_left_alone(self):
        queue = self._queue(self._task("gb-1-ok", race_age=5 * HOUR))
        self._guard(queue, mode="enforce").run_once()
        self.assertEqual(queue._items[0]["status"], "running")
        self.assertFalse(queue._items[0].get("guardFlag"))

    def test_finished_race_does_not_count(self):
        queue = self._queue(self._task("gb-1-reviewing", race_age=9 * HOUR, race_done=True))
        self._guard(queue).run_once()
        self.assertFalse(queue._items[0].get("guardFlag"))

    def test_limit_is_configurable(self):
        queue = self._queue(self._task("gb-1-slow", race_age=3 * HOUR))
        self._guard(queue, candidatePhaseHours=2).run_once()
        self.assertEqual(queue._items[0]["guardFlag"]["policy"], "candidate-timeout")

    def test_excluded_project_is_skipped(self):
        self.config["automation"]["excludedProjectCodes"] = ["gb-1"]
        queue = self._queue(self._task("gb-1-slow", race_age=9 * HOUR))
        self._guard(queue, mode="enforce").run_once()
        self.assertEqual(queue._items[0]["status"], "running")

    def test_finished_queue_item_is_not_touched(self):
        task_root = self._task("gb-1-slow", race_age=9 * HOUR)
        queue = self._queue(task_root, status="done", capacityHeld=False)
        self._guard(queue, mode="enforce").run_once()
        self.assertEqual(queue._items[0]["status"], "done")
        self.assertEqual(self._state(task_root)["status"], "candidates_running")


class PostTimeoutTests(GuardTestCase):
    def test_long_review_is_stopped_after_the_post_limit(self):
        task_root = self._task("gb-2-post", status="semantic_review_required", race_age=6 * HOUR,
                               race_done=True, post_age=5 * HOUR)
        queue = self._queue(task_root)
        self._guard(queue, mode="enforce").run_once()
        item = queue._items[0]
        self.assertEqual(item["status"], "failed")
        self.assertIn("评审/录屏阶段超时", item["error"])
        self.assertEqual(self._state(task_root)["closedPolicy"], "post-timeout")

    def test_within_the_post_limit_is_left_alone(self):
        task_root = self._task("gb-2-post", status="semantic_review_required", race_age=6 * HOUR,
                               race_done=True, post_age=3 * HOUR)
        queue = self._queue(task_root)
        self._guard(queue, mode="enforce").run_once()
        self.assertEqual(queue._items[0]["status"], "running")
        self.assertFalse(queue._items[0].get("guardFlag"))

    def test_post_limit_is_configurable(self):
        task_root = self._task("gb-2-post", status="semantic_review_required", race_age=6 * HOUR,
                               race_done=True, post_age=3 * HOUR)
        queue = self._queue(task_root)
        self._guard(queue, postPhaseHours=2).run_once()
        self.assertEqual(queue._items[0]["guardFlag"]["policy"], "post-timeout")

    def test_worker_wait_outlasts_both_phase_limits(self):
        self.assertEqual(worker_wait_timeout_seconds({}), (6 + 4 + 1) * HOUR)
        config = {"automation": {"guard": {"candidatePhaseHours": 8, "postPhaseHours": 2}}}
        self.assertEqual(worker_wait_timeout_seconds(config), 11 * HOUR)


class QuietTaskTests(GuardTestCase):
    def test_quiet_task_is_left_to_the_queue_lease(self):
        queue = self._queue(self._task("gb-1-quiet", status="semantic_review_required", quiet=2 * HOUR))
        self._guard(queue, mode="enforce").run_once()
        self.assertFalse(queue._items[0].get("guardFlag"))


class LeakedProcessTests(GuardTestCase):
    def _leaky(self, name="ld-9-finished", status="complete", quiet=3 * HOUR):
        task_root = self._task(name, status=status, quiet=quiet)
        self.processes = [
            {"pid": 5001, "pgid": 5000, "command": f"node {task_root}/monitor/verify/a/frontend/node_modules/.bin/vite --port 5610"},
            {"pid": 5002, "pgid": 5002, "command": f"/usr/bin/login -qflp me /bin/zsh -fc cd '{task_root}/source'"},
        ]
        return task_root

    def test_observe_lists_the_leak(self):
        task_root = self._leaky()
        queue = build_queue(self.config, items=[])
        guard = self._guard(queue)
        guard.run_once()
        guard.run_once()
        leaks = queue.guard_status["leaks"]
        self.assertEqual([leak["name"] for leak in leaks], [task_root.name])
        self.assertEqual([p["pid"] for p in leaks[0]["processes"]], [5001])
        self.assertEqual(self.log.names().count("guard.leak_found"), 1)
        self.assertEqual(self.killed, [])

    def test_enforce_kills_only_non_shell_processes(self):
        self._leaky()
        queue = build_queue(self.config, items=[])
        actions = self._guard(queue, mode="enforce").run_once()
        self.assertEqual(self.killed, [5001])
        self.assertEqual(actions[0]["kind"], "guard-leak-killed")

    def test_recently_finished_task_is_left_alone(self):
        self._leaky(quiet=10 * 60)
        queue = build_queue(self.config, items=[])
        self._guard(queue, mode="enforce").run_once()
        self.assertEqual(self.killed, [])

    def test_unfinished_task_is_not_a_leak(self):
        self._leaky(status="gsb_ready")
        queue = build_queue(self.config, items=[])
        self._guard(queue, mode="enforce").run_once()
        self.assertEqual(self.killed, [])

    def test_task_owned_by_an_active_item_is_not_a_leak(self):
        task_root = self._leaky(status="failed")
        queue = self._queue(task_root)
        self._guard(queue, mode="enforce").run_once()
        self.assertEqual(self.killed, [])


class GuardModeTests(GuardTestCase):
    def test_off_clears_flags_and_checks_nothing(self):
        queue = self._queue(self._task("gb-1-slow", race_age=9 * HOUR))
        guard = self._guard(queue)
        guard.run_once()
        self.assertTrue(queue._items[0]["guardFlag"])
        self.config["automation"]["guard"]["mode"] = "off"
        guard.run_once()
        self.assertFalse(queue._items[0]["guardFlag"])
        self.assertEqual(queue.guard_status["mode"], "off")

    def test_settings_default_to_observe_and_are_clamped(self):
        self.assertEqual(guard_settings({})["mode"], "observe")
        clamped = guard_settings({"automation": {"guard": {"mode": "nuke", "candidatePhaseHours": 0.1,
                                                           "leakedProcessMinutes": 99999}}})
        self.assertEqual((clamped["mode"], clamped["candidatePhaseHours"], clamped["leakedProcessMinutes"]),
                         ("observe", 1, 1440))
        self.assertNotIn("noProgressMinutes", clamped)

    def test_snapshot_carries_the_guard_status(self):
        queue = self._queue(self._task("gb-1-slow", race_age=9 * HOUR))
        self._guard(queue).run_once()
        self.assertEqual(queue.fast_snapshot()["guard"]["mode"], "observe")
        self.assertEqual(len(queue.snapshot()["guard"]["flags"]), 1)

    def test_stopped_item_stays_failed_after_a_state_machine_pass(self):
        task_root = self._task("gb-1-slow", race_age=9 * HOUR)
        queue = self._queue(task_root)
        self._guard(queue, mode="enforce").run_once()
        queue.snapshot()
        self.assertEqual(queue._items[0]["status"], "failed")
        self.assertFalse(queue._items[0]["capacityHeld"])


class SetGuardActionTests(SchedulerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.service = SchedulerService(self.config)
        self.addCleanup(self.service.stop)

    def test_mode_and_thresholds_persist(self):
        snapshot = self.service.automation_action("set-guard", {"mode": "enforce", "leakedProcessMinutes": 90,
                                                                "candidatePhaseHours": 7.5, "postPhaseHours": 3})
        self.assertEqual(snapshot["guard"]["mode"], "enforce")
        self.assertEqual(snapshot["guard"]["leakedProcessMinutes"], 90)
        saved = json.loads((self.root / "config.json").read_text(encoding="utf-8"))["automation"]["guard"]
        self.assertEqual((saved["mode"], saved["leakedProcessMinutes"], saved["candidatePhaseHours"],
                          saved["postPhaseHours"]), ("enforce", 90, 7.5, 3))
        self.assertEqual(snapshot["guard"]["postPhaseHours"], 3)

    def test_bad_values_are_rejected(self):
        for payload in ({"mode": "nuke"}, {"leakedProcessMinutes": 5}, {"candidatePhaseHours": "x"},
                        {"postPhaseHours": 0.5}, {}):
            with self.assertRaises(MonitorError):
                self.service.automation_action("set-guard", payload)

    def test_reconcile_runs_the_guard(self):
        self.assertIs(self.service.reconcile.guard, self.service.guard)
        self.service.reconcile.run_once()
        self.assertEqual(self.service.queue.guard_status["mode"], "observe")
