"""Auto refill: the on/off switch, its status line and the stable preview head."""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from api.common import AUTO_REFILL_PREVIEW_SIZE, SettingsStore  # noqa: E402
from api.service import SchedulerService  # noqa: E402
from tests.support import SchedulerTestCase  # noqa: E402


class _Platform:
    """Solo Manager stand-in: every task type offers the same fresh projects."""

    def __init__(self, codes):
        self.codes = codes

    def candidates(self, *, task_type, force=False, include_pool=False):
        return {"items": [{"code": code, "name": f"项目 {code}", "variantId": f"v-{code}"} for code in self.codes],
                "excluded": []}

    def set_blocklist(self, codes):
        pass


class AutoRefillSwitchTests(SchedulerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.service = SchedulerService(self.config)
        self.service.settings = SettingsStore(self.root / ".state" / "settings.json")
        self.addCleanup(self.service.stop)

    def _saved(self):
        return json.loads((self.root / "config.json").read_text(encoding="utf-8"))

    def _events(self):
        path = self.root / ".state" / "scheduler.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def _add_pending(self, count, prefix="gb"):
        for index in range(count):
            self.service.queue.add_platform({"code": f"{prefix}-{index}", "name": f"项目 {index}",
                                             "variantId": f"v-{index}"})

    def test_switch_persists_and_logs(self):
        snapshot = self.service.automation_action("set-auto-refill", {"enabled": True})
        self.assertTrue(snapshot["autoRefill"]["enabled"])
        self.assertTrue(self._saved()["automation"]["autoRefill"]["enabled"])

        snapshot = self.service.automation_action("set-auto-refill", {"enabled": False})
        self.assertFalse(snapshot["autoRefill"]["enabled"])
        self.assertFalse(self._saved()["automation"]["autoRefill"]["enabled"])
        self.assertEqual(snapshot["autoRefill"]["lastRun"]["status"], "disabled")
        details = [event["detail"] for event in self._events() if event["event"] == "config.auto_refill"]
        self.assertEqual(details, ["自动补队 → 开启", "自动补队 → 关闭"])

    def test_switch_keeps_the_other_refill_settings(self):
        self.config["automation"]["autoRefill"].update({"targetPending": 7, "taskTypes": ["feature迭代"]})
        self.service.automation_action("set-auto-refill", {"enabled": True})
        saved = self._saved()["automation"]["autoRefill"]
        self.assertEqual(saved["targetPending"], 7)
        self.assertEqual(saved["taskTypes"], ["feature迭代"])

    def test_snapshot_exposes_interval_and_preview_size(self):
        refill = self.service.queue.fast_snapshot()["autoRefill"]
        self.assertEqual(refill["previewSize"], AUTO_REFILL_PREVIEW_SIZE)
        self.assertGreaterEqual(refill["intervalSeconds"], 30)
        self.assertEqual(refill["lastRun"]["status"], "idle")

    def test_turning_on_refills_without_waiting_out_the_interval(self):
        self.service._last_queue_refill_at = time.time()
        self.service.automation_action("set-auto-refill", {"enabled": True})
        self.assertEqual(self.service._last_queue_refill_at, 0.0)

    def test_refill_round_is_reported_and_keeps_the_preview_head(self):
        self._add_pending(AUTO_REFILL_PREVIEW_SIZE + 5)
        head = [item["id"] for item in self.service.queue._items[:AUTO_REFILL_PREVIEW_SIZE]]
        self.config["automation"]["autoRefill"].update({
            "enabled": True, "targetPending": AUTO_REFILL_PREVIEW_SIZE + 8, "taskTypes": ["feature迭代"],
            "taskTypeWeights": {"feature迭代": 100}, "randomize": True, "shuffleExisting": True,
        })
        self.service.platform = _Platform([f"new-{index}" for index in range(10)])

        result = self.service.maybe_refill_queue()

        self.assertEqual(result["added"], 3)
        self.assertEqual([item["id"] for item in self.service.queue._items[:AUTO_REFILL_PREVIEW_SIZE]], head)
        last = self.service.queue.fast_snapshot()["autoRefill"]["lastRun"]
        self.assertEqual(last["added"], 3)
        self.assertIn("随机补队 3 项", last["message"])

    def test_completed_history_can_refill_but_current_queue_stays_unique(self):
        self.config["automation"]["autoRefill"].update({
            "enabled": True,
            "targetPending": 1,
            "batchSize": 1,
            "taskTypes": ["feature迭代"],
            "taskTypeWeights": {"feature迭代": 100},
            "randomize": False,
            "shuffleExisting": False,
        })
        self.service.platform = _Platform(["old-1"])
        self.service.queue._triggered = [{
            "id": "old-run",
            "source": "platform",
            "projectCode": "old-1",
            "status": "done",
            "capacityHeld": False,
        }]
        self.service._last_queue_refill_at = 0.0

        first = self.service.maybe_refill_queue()

        self.assertEqual(first["added"], 1)
        self.assertEqual(
            [item["projectCode"] for item in self.service.queue._items],
            ["old-1"],
        )

        self.service._last_queue_refill_at = 0.0
        second = self.service.maybe_refill_queue()

        self.assertEqual(second["added"], 0)
        self.assertEqual(len(self.service.queue._items), 1)


    def _refill_one(self, codes):
        self.config["automation"]["autoRefill"].update({
            "enabled": True, "targetPending": 1, "batchSize": 1, "taskTypes": ["feature迭代"],
            "taskTypeWeights": {"feature迭代": 100}, "randomize": False, "shuffleExisting": False,
        })
        self.service.platform = _Platform(codes)
        self.service._last_queue_refill_at = 0.0
        return self.service.maybe_refill_queue()

    def test_reuse_is_on_by_default(self):
        self.assertTrue(self.service.queue.fast_snapshot()["projectReuse"])

    def test_without_reuse_finished_projects_are_not_refilled(self):
        self.service.automation_action("set-project-reuse", {"enabled": False})
        self.service.queue._triggered = [{"id": "old-run", "source": "platform", "projectCode": "old-1",
                                          "status": "done", "capacityHeld": False}]
        result = self._refill_one(["old-1"])
        self.assertEqual(result["added"], 0)
        self.assertEqual(self.service.queue._items, [])

    def test_without_reuse_manual_enqueue_of_a_finished_project_is_refused(self):
        from api.common import MonitorError

        self.service.automation_action("set-project-reuse", {"enabled": False})
        self.service.queue._items = [{"id": "old", "source": "platform", "projectCode": "old-1", "status": "done"}]
        with self.assertRaises(MonitorError):
            self.service.queue.add_platform({"code": "OLD-1", "name": "x", "variantId": "v"})
        self.service.automation_action("set-project-reuse", {"enabled": True})
        self.service.queue.add_platform({"code": "OLD-1", "name": "x", "variantId": "v"})

    def test_pending_projects_are_tracked_so_refill_never_doubles_them(self):
        self._add_pending(1, prefix="dup")
        self.assertIn("dup-0", self.service.queue.tracked_project_codes())


class ShufflePendingTests(SchedulerTestCase):
    def test_keep_head_leaves_the_first_items_in_place(self):
        service = SchedulerService(self.config)
        self.addCleanup(service.stop)
        for index in range(15):
            service.queue.add_platform({"code": f"gb-{index}", "name": f"项目 {index}", "variantId": f"v-{index}"})
        before = [item["id"] for item in service.queue._items]
        rng = random.Random(7)
        tails = set()
        for _ in range(20):
            self.assertTrue(service.queue.shuffle_pending(rng, keep_head=10))
            after = [item["id"] for item in service.queue._items]
            self.assertEqual(after[:10], before[:10])
            self.assertEqual(sorted(after[10:]), sorted(before[10:]))
            tails.add(tuple(after[10:]))
        self.assertGreater(len(tails), 1)

    def test_nothing_to_shuffle_behind_a_short_queue(self):
        service = SchedulerService(self.config)
        self.addCleanup(service.stop)
        for index in range(10):
            service.queue.add_platform({"code": f"gb-{index}", "name": f"项目 {index}", "variantId": f"v-{index}"})
        self.assertFalse(service.queue.shuffle_pending(random.Random(1), keep_head=10))


class PriorityPrefixTests(SchedulerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.config["automation"]["priorityProjects"] = {"enabled": True, "prefixes": ["SoloGSB"]}
        self.service = SchedulerService(self.config)
        self.service.settings = SettingsStore(self.root / ".state" / "settings.json")
        self.addCleanup(self.service.stop)

    def _codes(self):
        return [item["projectCode"] for item in self.service.queue._items]

    def test_priority_projects_jump_ahead_of_pending_items(self):
        queue = self.service.queue
        for code in ["gb-1", "gb-2", "sologsb-1", "gb-3", "sologsb-2"]:
            queue.add_platform({"code": code, "name": code, "variantId": f"v-{code}"})
        self.assertEqual(self._codes(), ["sologsb-1", "sologsb-2", "gb-1", "gb-2", "gb-3"])

    def test_priority_projects_stay_ahead_after_a_shuffle(self):
        queue = self.service.queue
        for index in range(12):
            queue.add_platform({"code": f"gb-{index}", "name": "x", "variantId": f"v-{index}"})
        queue.add_platform({"code": "sologsb-9", "name": "x", "variantId": "v-s"})
        queue.shuffle_pending(random.Random(3), keep_head=AUTO_REFILL_PREVIEW_SIZE)
        self.assertEqual(self._codes()[0], "sologsb-9")

    def test_refill_takes_priority_projects_before_the_rest(self):
        self.config["automation"]["autoRefill"].update({
            "enabled": True, "targetPending": 2, "batchSize": 2, "taskTypes": ["feature迭代"],
            "taskTypeWeights": {"feature迭代": 100}, "randomize": True, "shuffleExisting": False,
        })
        self.service.platform = _Platform(["gb-1", "gb-2", "gb-3", "sologsb-1", "sologsb-2"])
        self.service._last_queue_refill_at = 0.0
        result = self.service.maybe_refill_queue()
        self.assertEqual(result["added"], 2)
        self.assertEqual(sorted(self._codes()), ["sologsb-1", "sologsb-2"])

    def test_full_queue_still_takes_priority_projects_only(self):
        for index in range(3):
            self.service.queue.add_platform({"code": f"gb-{index}", "name": "x", "variantId": f"v-{index}"})
        self.config["automation"]["autoRefill"].update({
            "enabled": True, "targetPending": 3, "batchSize": 10, "taskTypes": ["feature迭代"],
            "taskTypeWeights": {"feature迭代": 100}, "randomize": False, "shuffleExisting": False,
        })
        self.service.platform = _Platform(["gb-7", "gb-8", "sologsb-1", "sologsb-2"])
        self.service._last_queue_refill_at = 0.0
        result = self.service.maybe_refill_queue()
        self.assertEqual(result["added"], 2)
        self.assertEqual(self._codes(), ["sologsb-1", "sologsb-2", "gb-0", "gb-1", "gb-2"])

    def test_a_finished_priority_project_requeues_behind_waiting_priority_projects(self):
        queue = self.service.queue
        for code in ["sologsb-1", "sologsb-2", "sologsb-3", "gb-1"]:
            queue.add_platform({"code": code, "name": code, "variantId": f"v-{code}"})
        queue._items[0]["status"] = "done"
        self.config["automation"]["autoRefill"].update({
            "enabled": True, "targetPending": 3, "batchSize": 5, "taskTypes": ["feature迭代"],
            "taskTypeWeights": {"feature迭代": 100}, "randomize": True, "shuffleExisting": True,
        })
        self.service.platform = _Platform(["gb-1", "sologsb-1", "sologsb-2", "sologsb-3"])
        self.service._last_queue_refill_at = 0.0
        self.assertEqual(self.service.maybe_refill_queue()["added"], 1)
        pending = [item["projectCode"] for item in queue._items if item["status"] == "pending"]
        self.assertEqual(pending, ["sologsb-2", "sologsb-3", "sologsb-1", "gb-1"])

    def test_shuffle_keeps_priority_projects_in_queue_order(self):
        queue = self.service.queue
        codes = [f"sologsb-{index}" for index in range(14)]
        for code in [*codes, "gb-1", "gb-2", "gb-3"]:
            queue.add_platform({"code": code, "name": code, "variantId": f"v-{code}"})
        for seed in range(5):
            queue.shuffle_pending(random.Random(seed), keep_head=AUTO_REFILL_PREVIEW_SIZE)
            self.assertEqual(self._codes()[:14], codes)

    def test_switched_off_the_queue_keeps_its_usual_order(self):
        self.config["automation"]["priorityProjects"]["enabled"] = False
        for code in ["gb-1", "sologsb-1"]:
            self.service.queue.add_platform({"code": code, "name": code, "variantId": f"v-{code}"})
        self.assertEqual(self._codes(), ["gb-1", "sologsb-1"])


class PriorityProjectsActionTests(SchedulerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.service = SchedulerService(self.config)
        self.service.settings = SettingsStore(self.root / ".state" / "settings.json")
        self.addCleanup(self.service.stop)

    def _saved(self):
        return json.loads((self.root / "config.json").read_text(encoding="utf-8"))["automation"]["priorityProjects"]

    def test_off_by_default(self):
        self.assertEqual(self.service.queue.fast_snapshot()["priorityProjects"], {"enabled": False, "prefixes": []})

    def test_saving_prefixes_and_switching_on_reorders_the_queue(self):
        for code in ["gb-1", "sologsb-1", "abc-1"]:
            self.service.queue.add_platform({"code": code, "name": code, "variantId": f"v-{code}"})
        self.service._last_queue_refill_at = time.time()
        snapshot = self.service.automation_action(
            "set-priority-projects", {"enabled": True, "prefixes": "sologsb， ABC, sologsb"})
        self.assertEqual(snapshot["priorityProjects"], {"enabled": True, "prefixes": ["sologsb", "ABC"]})
        self.assertEqual(self._saved(), {"enabled": True, "prefixes": ["sologsb", "ABC"]})
        self.assertEqual([item["projectCode"] for item in self.service.queue._items], ["sologsb-1", "abc-1", "gb-1"])
        self.assertEqual(self.service._last_queue_refill_at, 0.0)

    def test_switching_on_without_a_prefix_is_refused(self):
        from api.common import MonitorError

        with self.assertRaises(MonitorError):
            self.service.automation_action("set-priority-projects", {"enabled": True})

    def test_clearing_the_prefixes_switches_it_off(self):
        self.service.automation_action("set-priority-projects", {"enabled": True, "prefixes": ["sologsb"]})
        snapshot = self.service.automation_action("set-priority-projects", {"prefixes": ""})
        self.assertEqual(snapshot["priorityProjects"], {"enabled": False, "prefixes": []})
