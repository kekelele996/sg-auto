"""LLM outage guard: auto pause, auto resume, and never touching a manual pause."""
from __future__ import annotations

import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from api.llm_guard import LlmGuard, probe_llm  # noqa: E402
from api.service import SchedulerService  # noqa: E402
from tests.support import SchedulerTestCase  # noqa: E402


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _guard(config, results):
    clock = _Clock()
    events = []
    outcomes = iter(results)
    guard = LlmGuard(
        config,
        persist=lambda: None,
        emit=lambda event, **kw: events.append((event, kw.get("detail", ""))),
        probe=lambda endpoint, timeout: {"ok": next(outcomes), "error": "down", "latencyMs": 5},
        resolve=lambda config: {"baseUrl": "https://llm.example", "key": "k", "model": "m"},
        clock=clock,
    )
    return guard, clock, events


class LlmGuardTests(SchedulerTestCase):
    def test_pauses_after_threshold_and_resumes_when_back(self):
        self.config["automation"]["paused"] = False
        guard, clock, events = _guard(self.config, [False, False, False, True])

        guard.step()
        self.assertFalse(self.config["automation"]["paused"])
        clock.now += 60
        guard.step()
        self.assertTrue(self.config["automation"]["paused"])
        self.assertTrue(guard.settings()["pausedByGuard"])

        # While paused by the guard, the next probe waits the paused interval.
        clock.now += 60
        self.assertIsNone(guard.step())
        clock.now += 240
        guard.step()
        self.assertTrue(self.config["automation"]["paused"])
        clock.now += 300
        guard.step()
        self.assertFalse(self.config["automation"]["paused"])
        self.assertFalse(guard.settings()["pausedByGuard"])
        self.assertEqual([name for name, _ in events if name in {"llm_guard.paused", "llm_guard.resumed"}],
                         ["llm_guard.paused", "llm_guard.resumed"])

    def test_manual_pause_is_never_lifted(self):
        self.config["automation"]["paused"] = True
        guard, clock, _ = _guard(self.config, [True])
        guard.step()
        self.assertTrue(self.config["automation"]["paused"])
        # A test probe that succeeds does not start a manually paused queue either.
        guard.test()
        self.assertTrue(self.config["automation"]["paused"])

    def test_manual_action_takes_the_pause_back(self):
        self.config["automation"]["paused"] = False
        guard, clock, _ = _guard(self.config, [False, False, True])
        guard.step(); clock.now += 60; guard.step()
        self.assertTrue(guard.settings()["pausedByGuard"])
        guard.note_manual_pause()
        clock.now += 300
        guard.step()
        self.assertTrue(self.config["automation"]["paused"])

    def test_disabled_guard_does_not_pause(self):
        self.config["automation"]["paused"] = False
        self.config["automation"]["llmGuard"] = {"enabled": False}
        guard, clock, _ = _guard(self.config, [False, False])
        guard.step(); clock.now += 60; guard.step()
        self.assertFalse(self.config["automation"]["paused"])

    def test_probe_without_key_reports_instead_of_calling(self):
        result = probe_llm({"baseUrl": "https://llm.example", "key": "", "model": "m"})
        self.assertFalse(result["ok"])
        self.assertIn("Key", result["error"])


class LlmGuardActionTests(SchedulerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.service = SchedulerService(self.config)
        self.addCleanup(self.service.stop)
        self.service.llm_guard._probe = lambda endpoint, timeout: {"ok": True, "latencyMs": 7, "error": "",
                                                                 "at": "2026-09-25T00:00:00Z"}
        self.service.llm_guard._resolve = lambda config: {"baseUrl": "https://llm.example", "key": "k", "model": "m"}

    def test_switch_and_test_are_in_the_snapshot(self):
        snapshot = self.service.automation_action("set-llm-guard", {"enabled": False})
        self.assertFalse(snapshot["llmGuard"]["enabled"])
        saved = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        self.assertFalse(saved["automation"]["llmGuard"]["enabled"])
        snapshot = self.service.automation_action("test-llm")
        self.assertTrue(snapshot["llmGuard"]["lastProbe"]["ok"])
        self.assertEqual(snapshot["llmGuard"]["lastProbe"]["latencyMs"], 7)

    def test_manual_start_clears_the_guard_pause(self):
        self.config["automation"]["llmGuard"] = {"pausedByGuard": True}
        self.service.automation_action("set-paused", {"paused": False})
        self.assertFalse(self.service.llm_guard.settings()["pausedByGuard"])
