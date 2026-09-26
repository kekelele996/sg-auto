"""Tests for the container-start probe and the launch gate it drives."""
from __future__ import annotations

import sys
import threading
import unittest
import unittest.mock
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from api.docker_probe import FALLBACK_IMAGE, DockerProbe, resolve_image  # noqa: E402
from tests.support import SchedulerTestCase, make_config  # noqa: E402
from tests.test_scheduler import build_queue, fake_docker, platform_item  # noqa: E402


class FakeClock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def make_probe(config, results, clock=None):
    """A probe whose docker runs return ``results`` in order, synchronously."""
    calls = []

    def probe(image, timeout):
        calls.append((image, timeout))
        return results.pop(0)

    guard = DockerProbe(config, emit=lambda event, **kw: events.append(event), probe=probe, clock=clock or FakeClock())
    events: list[str] = []
    guard.events = events
    guard.calls = calls
    return guard


def settle(guard):
    thread = guard._thread
    if isinstance(thread, threading.Thread):
        thread.join(5)


class DockerProbeTests(SchedulerTestCase):
    def test_two_slow_probes_hold_and_two_fast_ones_release(self):
        clock = FakeClock()
        ok, slow = {"ok": True, "seconds": 3.0}, {"ok": True, "seconds": 75.0}
        guard = make_probe(self.config, [slow, slow, ok, ok], clock)
        guard.maybe_probe(); settle(guard)
        self.assertTrue(guard.healthy, "one slow probe is not an outage")
        self.assertTrue(guard.blocking(), "but launches wait for the next probe")
        self.assertIn("超过 60 秒", guard.hold_reason())
        clock.now += 60
        guard.maybe_probe(); settle(guard)
        self.assertFalse(guard.healthy)
        self.assertIn("超过 60 秒", guard.hold_reason())
        self.assertEqual(guard.events, ["queue.docker_slow"])
        clock.now += 60
        guard.maybe_probe(); settle(guard)
        self.assertFalse(guard.healthy, "one fast probe does not end an outage")
        clock.now += 60
        guard.maybe_probe(); settle(guard)
        self.assertTrue(guard.healthy)
        self.assertEqual(guard.events, ["queue.docker_slow", "queue.docker_probe_recovered"])

    def test_probe_waits_out_its_interval(self):
        clock = FakeClock()
        guard = make_probe(self.config, [{"ok": True, "seconds": 2.0}], clock)
        self.assertTrue(guard.maybe_probe()); settle(guard)
        clock.now += 100
        self.assertFalse(guard.maybe_probe())
        self.assertEqual(len(guard.calls), 1)

    def test_first_launch_waits_for_a_result(self):
        clock = FakeClock()
        guard = make_probe(self.config, [], clock)
        started = threading.Event()
        release = threading.Event()

        def slow_probe(image, timeout):
            started.set()
            release.wait(5)
            return {"ok": True, "seconds": 2.0}

        guard._probe = slow_probe
        self.assertTrue(guard.blocking(), "no result yet: hold")
        started.wait(5)
        self.assertIn("正在检测", guard.hold_reason())
        release.set(); settle(guard)
        self.assertFalse(guard.blocking())

    def test_disabled_probe_never_holds(self):
        self.config["automation"]["dockerProbe"] = {"enabled": False}
        guard = make_probe(self.config, [])
        self.assertFalse(guard.blocking())
        self.assertEqual(guard.calls, [])

    def test_image_follows_the_skill_runner(self):
        runner = self.root / "side_runner.py"
        runner.write_text('DEFAULT_IMAGE = os.environ.get(\n    "SOLOSB_DOCKER_IMAGE",\n    "example/image:1",\n)\n', encoding="utf-8")
        config = make_config(self.root, skillScript=str(self.root / "sologsb.py"))
        with unittest.mock.patch.dict("os.environ", {}, clear=False) as env:
            env.pop("SOLOSB_DOCKER_IMAGE", None)
            self.assertEqual(resolve_image(config), "example/image:1")
            runner.unlink()
            self.assertEqual(resolve_image(config), FALLBACK_IMAGE)


class DockerGateTests(SchedulerTestCase):
    def test_unhealthy_docker_holds_launches_without_claiming_quota(self):
        queue = build_queue(self.config, docker=fake_docker([]))
        guard = make_probe(self.config, [])
        guard.healthy = False
        guard.last_result_at = guard._clock()
        guard.last_probe_at = guard._clock()
        guard.last_probe = {"ok": False, "error": "180 秒内未能启动容器"}
        queue.docker_probe = guard
        item = platform_item()
        queue._items = [item]
        queue._save()
        queue.tick()
        self.assertEqual(item["status"], "pending")
        self.assertIn("Docker 创建容器过慢", str(item.get("containerWait") or ""))
        self.assertEqual(item.get("containerWaitKind"), "gate")
        self.assertEqual(item["quota"]["state"], "pending")
        self.assertFalse(queue.snapshot()["dockerProbe"]["healthy"])


if __name__ == "__main__":
    unittest.main()
