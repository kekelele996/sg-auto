"""Project use counts from the QC platform and the enqueue gate built on them."""
from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from api.common import MonitorError, SettingsStore  # noqa: E402
from api.platform import PlatformProvider  # noqa: E402
from api.qc_usage import (  # noqa: E402
    ProjectUsage,
    count_submissions,
    fetch_submissions,
    project_usage_settings,
    repo_project_code,
)
from api.service import SchedulerService  # noqa: E402
from tests.support import SchedulerTestCase  # noqa: E402


def _submission(repo, task_type="feature迭代", stage="QC_PASSED"):
    return {"repo_id": repo, "question_type": task_type, "stage": stage}


def _usage(items, config=None):
    usage = ProjectUsage(config if config is not None else {}, fetch=lambda _config: {"items": items, "source": "test"})
    usage.refresh()
    return usage


class RepoCodeTests(unittest.TestCase):
    def test_codes_are_read_from_repo_names(self):
        self.assertEqual(repo_project_code("kekelele996/gb-532-qq2k"), "gb-532")
        self.assertEqual(repo_project_code("kekelele996/cy-178---0-1-qpqfo7"), "cy-178")
        self.assertEqual(repo_project_code("kekelele996/cy402-hearing-schedule-wt1qpi"), "cy-402")
        self.assertEqual(repo_project_code("kekelele996/tripweaver-yw181a"), "")

    def test_discarded_and_unparsed_submissions_do_not_count(self):
        counted = count_submissions([
            _submission("o/cy-302-a"),
            _submission("o/cy-302-b", "0-1代码生成"),
            _submission("o/cy-302-c", stage="DISCARDED"),
            _submission("o/reswap-v649i0"),
        ])
        self.assertEqual(counted["codes"]["cy-302"], {"total": 2, "byType": {"feature迭代": 1, "0-1代码生成": 1}})
        self.assertEqual(counted["counted"], 2)
        self.assertEqual(counted["unparsed"], ["o/reswap-v649i0"])

    def test_settings_are_clamped(self):
        settings = project_usage_settings({"automation": {"projectUsage": {"limit": 0, "scope": "x"}}})
        self.assertEqual(settings["limit"], 1)
        self.assertEqual(settings["scope"], "total")
        self.assertEqual(project_usage_settings({})["limit"], 10)


class GateTests(unittest.TestCase):
    def test_total_scope_blocks_at_the_limit(self):
        items = [_submission(f"o/cy-1-{n}", "feature迭代" if n % 2 else "0-1代码生成") for n in range(10)]
        usage = _usage(items)
        self.assertEqual(usage.usage("CY-1")["used"], 10)
        self.assertIn("达到上限 10", usage.blocked_reason("cy-1", "feature迭代"))
        self.assertEqual(usage.blocked_reason("cy-2"), "")

    def test_queued_items_count_as_uses(self):
        usage = _usage([_submission(f"o/cy-1-{n}") for n in range(9)])
        self.assertEqual(usage.blocked_reason("cy-1"), "")
        usage.inflight = lambda: {"cy-1": {"feature迭代": 1}}
        self.assertIn("队列中 1", usage.blocked_reason("cy-1"))
        self.assertEqual(usage.blocked_reason("cy-1", include_queued=False), "")
        self.assertEqual(usage.usage("cy-1")["queuedByType"], {"feature迭代": 1})

    def test_status_totals_uses_by_type(self):
        usage = _usage([_submission("o/cy-1-a"), _submission("o/cy-2-a"), _submission("o/cy-2-b", "Bug修复")])
        self.assertEqual(usage.status()["byType"], {"feature迭代": 2, "Bug修复": 1})

    def test_per_type_scope_counts_each_type_alone(self):
        config = {"automation": {"projectUsage": {"scope": "perType", "limit": 3}}}
        usage = _usage([_submission(f"o/cy-1-{n}") for n in range(3)], config)
        self.assertIn("feature迭代", usage.blocked_reason("cy-1", "feature迭代"))
        self.assertEqual(usage.blocked_reason("cy-1", "0-1代码生成"), "")

    def test_disabled_never_blocks(self):
        config = {"automation": {"projectUsage": {"enabled": False, "limit": 1}}}
        usage = _usage([_submission("o/cy-1-a")], config)
        self.assertEqual(usage.blocked_reason("cy-1"), "")

    def test_nothing_is_blocked_before_the_first_fetch(self):
        def fail(_config):
            raise RuntimeError("HTTP 401")

        usage = ProjectUsage({"automation": {"projectUsage": {"limit": 1}}}, fetch=fail, retry_delay=0)
        status = usage.refresh()
        self.assertIn("401", status["error"])
        self.assertFalse(status["loaded"])
        self.assertEqual(usage.blocked_reason("cy-1"), "")

    def test_a_failed_refresh_keeps_the_last_counts(self):
        calls = {"n": 0}

        def fetch(_config):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("HTTP 401")
            return {"items": [_submission("o/cy-1-a")]}

        usage = ProjectUsage({"automation": {"projectUsage": {"limit": 1}}}, fetch=fetch, retry_delay=0)
        usage.refresh()
        status = usage.refresh()
        self.assertTrue(status["error"])
        self.assertTrue(usage.blocked_reason("cy-1"))

    def test_a_failed_fetch_is_retried_once(self):
        calls = {"n": 0}

        def fetch(_config):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("timed out")
            return {"items": [_submission("o/cy-1-a")]}

        usage = ProjectUsage({}, fetch=fetch, retry_delay=0)
        failed = []
        usage.on_failed = failed.append
        status = usage.refresh()
        self.assertEqual(calls["n"], 2)
        self.assertEqual(status["error"], "")
        self.assertEqual(failed, [])

    def test_two_failures_ask_for_one_llm_probe_per_streak(self):
        calls = {"n": 0}

        def fetch(_config):
            calls["n"] += 1
            raise RuntimeError("timed out")

        usage = ProjectUsage({}, fetch=fetch, retry_delay=0)
        failed = []
        usage.on_failed = failed.append
        usage.refresh()
        usage.refresh()
        self.assertEqual(calls["n"], 4)
        self.assertEqual(failed, ["timed out"])


class FetchTests(unittest.TestCase):
    def test_all_pages_are_fetched_and_no_cookie_leaks(self):
        pages = {
            1: {"items": [_submission("o/cy-1-a")], "meta": {"page": 1, "total_pages": 2}},
            2: {"items": [_submission("o/cy-2-a")], "meta": {"page": 2, "total_pages": 2}},
        }
        seen = []

        def urlopen(request, timeout=0):
            seen.append(request.full_url)
            page = int(request.full_url.split("page=")[1].split("&")[0])
            return io.BytesIO(json.dumps(pages[page]).encode("utf-8"))

        with mock.patch("api.qc_usage._solo2_credentials", return_value=[("test", "c=1", "t")]), \
                mock.patch("api.qc_usage.urllib.request.urlopen", side_effect=urlopen):
            result = fetch_submissions({})
        self.assertEqual(len(result["items"]), 2)
        self.assertEqual(len(seen), 2)

        import urllib.error

        def denied(request, timeout=0):
            raise urllib.error.HTTPError(request.full_url, 401, "no", {}, io.BytesIO(b""))

        with mock.patch("api.qc_usage._solo2_credentials", return_value=[("test", "secret-cookie", "t")]), \
                mock.patch("api.qc_usage.urllib.request.urlopen", side_effect=denied):
            with self.assertRaises(RuntimeError) as caught:
                fetch_submissions({})
        self.assertIn("HTTP 401", str(caught.exception))
        self.assertNotIn("secret-cookie", str(caught.exception))


class CandidateFilterTests(unittest.TestCase):
    def test_candidates_at_the_limit_are_excluded_and_others_annotated(self):
        provider = PlatformProvider({})
        provider.project_usage = _usage([_submission(f"o/cy-1-{n}") for n in range(10)])
        result = provider._apply_usage({
            "items": [{"code": "cy-1"}, {"code": "cy-2"}], "excluded": [], "taskType": "feature迭代",
        })
        self.assertEqual([item["code"] for item in result["items"]], ["cy-2"])
        self.assertEqual(result["items"][0]["qcUsage"]["used"], 0)
        self.assertIn("质检已用 10", result["excluded"][0]["reason"])


class ServiceUsageTests(SchedulerTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.service = SchedulerService(self.config)
        self.service.settings = SettingsStore(self.root / ".state" / "settings.json")
        self.addCleanup(self.service.stop)
        self.service.project_usage._fetch = lambda _config: {
            "items": [_submission(f"o/cy-1-{n}") for n in range(10)]}
        self.service.project_usage.refresh()

    def test_enqueue_is_refused_at_the_limit(self):
        with self.assertRaises(MonitorError) as caught:
            self.service.queue.add_platform({"code": "CY-1", "name": "x", "variantId": "v"}, task_type="feature迭代")
        self.assertIn("达到上限", str(caught.exception))
        self.service.automation_action("set-project-usage", {"limit": 11})
        self.service.queue.add_platform({"code": "CY-1", "name": "x", "variantId": "v"}, task_type="feature迭代")

    def test_queue_items_and_snapshot_carry_usage(self):
        self.service.queue.add_platform({"code": "cy-2", "name": "x", "variantId": "v"}, task_type="feature迭代")
        snapshot = self.service.queue.fast_snapshot()
        self.assertEqual(snapshot["items"][0]["qcUsage"]["queued"], 1)
        self.assertEqual(snapshot["projectUsage"]["limit"], 10)
        self.assertEqual(self.service.queue.inflight_project_types(), {"cy-2": {"feature迭代": 1}})

    def test_settings_action_validates_and_persists(self):
        with self.assertRaises(MonitorError):
            self.service.automation_action("set-project-usage", {"scope": "weekly"})
        snapshot = self.service.automation_action("set-project-usage", {"scope": "perType", "enabled": False})
        self.assertEqual(snapshot["projectUsage"]["scope"], "perType")
        self.assertFalse(snapshot["projectUsage"]["enabled"])
        saved = json.loads((self.root / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["automation"]["projectUsage"]["scope"], "perType")


if __name__ == "__main__":
    unittest.main()
