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

    def test_default_probe_interval_is_thirty_minutes(self):
        guard, _clock, _events = _guard({"automation": {}}, [])
        self.assertEqual(guard.settings()["probeSeconds"], 1800)

    def test_intervals_are_validated_and_applied(self):
        guard, _clock, _events = _guard(self.config, [])
        with self.assertRaises(ValueError):
            guard.set_intervals(probe_seconds=10)
        settings = guard.set_intervals(probe_seconds=2700, paused_probe_seconds=600)
        self.assertEqual((settings["probeSeconds"], settings["pausedProbeSeconds"]), (2700, 600))

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

    def test_interval_action_validates_and_persists(self):
        from api.common import MonitorError
        for bad in (0, 2000, "x"):
            with self.assertRaises(MonitorError):
                self.service.automation_action("set-llm-guard-interval", {"probeMinutes": bad})
        snapshot = self.service.automation_action("set-llm-guard-interval", {"probeMinutes": 45})
        self.assertEqual(snapshot["llmGuard"]["probeSeconds"], 2700)
        saved = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["automation"]["llmGuard"]["probeSeconds"], 2700)

    def test_manual_start_clears_the_guard_pause(self):
        self.config["automation"]["llmGuard"] = {"pausedByGuard": True}
        self.service.automation_action("set-paused", {"paused": False})
        self.assertFalse(self.service.llm_guard.settings()["pausedByGuard"])


class ProbeResponseTests(SchedulerTestCase):
    """probe_llm only passes when the model actually answers with text."""

    ENDPOINT = {"baseUrl": "https://llm.example", "key": "secret-key", "model": "m"}

    def _probe(self, body: str):
        from unittest import mock

        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return body.encode("utf-8")

        with mock.patch("api.llm_guard.urllib.request.urlopen", return_value=Response()):
            return probe_llm(self.ENDPOINT, 5)

    def test_real_text_reply_passes_and_is_reported(self):
        result = self._probe(json.dumps({
            "type": "message", "model": "auto_model/urm", "stop_reason": "end_turn",
            "usage": {"output_tokens": 26},
            "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "OK"}],
        }))
        self.assertTrue(result["ok"])
        self.assertEqual(result["reply"], "OK")
        self.assertEqual(result["replyModel"], "auto_model/urm")
        self.assertEqual(result["outputTokens"], 26)

    def test_200_without_text_fails(self):
        result = self._probe(json.dumps({"type": "message", "content": [{"type": "thinking", "thinking": "x"}],
                                         "stop_reason": "max_tokens"}))
        self.assertFalse(result["ok"])
        self.assertIn("没有返回文本", result["error"])

    def test_200_with_non_json_body_fails(self):
        result = self._probe("<html>502 Bad Gateway</html>")
        self.assertFalse(result["ok"])
        self.assertIn("不是 JSON", result["error"])

    def test_error_body_fails(self):
        result = self._probe(json.dumps({"type": "error", "error": {"message": "overloaded"}}))
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "overloaded")


class StatusStabilityTests(SchedulerTestCase):
    def test_status_does_not_change_as_time_passes(self):
        # The snapshot hub pushes the page whenever the queue payload changes;
        # a ticking countdown here re-rendered the page every build.
        self.config["automation"]["paused"] = False
        guard, clock, _ = _guard(self.config, [True])
        guard.step()
        before = guard.status()
        clock.now += 7
        self.assertEqual(guard.status(), before)
        self.assertTrue(before["nextProbeAt"])
