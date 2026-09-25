"""Tests for the task scanner: slim cards, full details, trace parsing."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from api.tasks import (  # noqa: E402
    DockerCache,
    TaskScanner,
    TraceCache,
    _new_trace_stats,
    container_group_name,
    discover_task_roots,
    process_trace_event,
    read_trace_stats,
    task_container_names,
)
from tests.support import SchedulerTestCase, write_task  # noqa: E402


class TraceTests(unittest.TestCase):
    def test_processes_stream_json_events(self):
        stats = _new_trace_stats()
        stats["recentLimit"] = 10
        events = [
            {"type": "system", "subtype": "init", "session_id": "sess-1", "model": "claude"},
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Bash", "input": {"command": "ls -la"}},
            ]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "content": "ok", "is_error": False},
            ]}},
        ]
        for event in events:
            process_trace_event(stats, event)
        self.assertEqual(stats["sessionId"], "sess-1")
        self.assertEqual(stats["model"], "claude")
        self.assertEqual(stats["toolCalls"], 1)
        self.assertEqual(stats["toolResults"], 1)
        self.assertEqual(stats["lastResult"], "ok")
        self.assertEqual(len(stats["recent"]), 3)

    def test_todo_write_is_captured(self):
        stats = _new_trace_stats()
        process_trace_event(stats, {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "TodoWrite", "input": {"todos": [
                {"content": "接入源码", "status": "completed"},
                {"content": "跑测试", "status": "in_progress", "activeForm": "正在跑测试"},
            ]}},
        ]}})
        self.assertEqual(len(stats["todos"]), 2)
        self.assertEqual(stats["todos"][1]["status"], "in_progress")

    def test_read_trace_stats_from_file(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stdout.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in [
                {"type": "system", "subtype": "init", "session_id": "abc"},
                {"type": "result", "is_error": False, "stop_reason": "end_turn"},
            ]), encoding="utf-8")
            stats = read_trace_stats(path)
        self.assertEqual(stats["sessionId"], "abc")
        self.assertEqual(stats["lastPhase"], "done")

    def test_trace_cache_evicts_old_entries(self):
        cache = TraceCache(max_entries=2)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(4):
                path = Path(tmp) / f"{index}.jsonl"
                path.write_text(json.dumps({"type": "system", "subtype": "init", "session_id": f"s{index}"}), encoding="utf-8")
                paths.append(path)
            for path in paths:
                cache.stats(path)
            self.assertLessEqual(len(cache._entries), 2)


class DiscoveryTests(SchedulerTestCase):
    def test_discovers_task_roots_and_prunes(self):
        write_task(self.root, "gb-1-20260920")
        (self.root / "tasks" / "gb-1-20260920" / "node_modules").mkdir()
        (self.root / "tasks" / "not-a-task").mkdir()
        found = discover_task_roots([self.root / "tasks"])
        self.assertEqual([item.name for item in found], ["gb-1-20260920"])

    def test_missing_root_is_ignored(self):
        self.assertEqual(discover_task_roots([self.root / "nope"]), [])


class DockerListingFallbackTests(unittest.TestCase):
    """One corrupted container must not blind the whole scheduler.

    ``docker ps -a`` fails as a single command ("rw layer snapshot not found for
    container ...") when any container on the host is broken.  Reading that as
    "Docker unavailable" stopped every launch even though the daemon and the
    running containers were fine.
    """

    @staticmethod
    def _completed(code: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(["docker", "ps"], code, stdout, stderr)

    def test_broken_full_listing_falls_back_to_running_containers(self):
        broken = self._completed(1, "", "Error response from daemon: rw layer snapshot not found for container abc")
        running = self._completed(0, json.dumps({"ID": "1", "Names": "cand-1", "State": "running", "Status": "Up"}) + "\n")
        calls: list[list[str]] = []

        def fake_run(args, **_kwargs):
            calls.append(list(args))
            return broken if "-a" in args else running

        with mock.patch("api.tasks.subprocess.run", side_effect=fake_run):
            data = DockerCache(ttl=0).get()

        self.assertEqual(data.get("error"), "")
        self.assertTrue(data.get("fetchedOk"))
        self.assertEqual([item["name"] for item in data["items"]], ["cand-1"])
        self.assertEqual(len(calls), 2)
        self.assertNotIn("-a", calls[1])

    def test_two_broken_listings_still_report_docker_unavailable(self):
        broken = self._completed(1, "", "Error response from daemon: rw layer snapshot not found for container abc")
        with mock.patch("api.tasks.subprocess.run", return_value=broken):
            data = DockerCache(ttl=0).get()
        self.assertIn("rw layer snapshot not found", str(data.get("error") or ""))
        self.assertFalse(data.get("fetchedOk"))


class SnapshotTests(SchedulerTestCase):
    def _scanner(self):
        docker = DockerCache(ttl=0)
        docker._data = {"items": [], "byName": {}, "error": "", "fetchedOk": True}
        return TaskScanner(self.config, docker_cache=docker)

    def test_card_is_slim(self):
        write_task(self.root, "gb-2-20260920", status="running")
        cards = self._scanner().cards([self.root / "tasks"])
        self.assertEqual(len(cards), 1)
        card = cards[0]
        # The heavy fields must not be in the list payload.
        for absent in ("promptText", "sides", "candidates", "workflow", "localHead"):
            self.assertNotIn(absent, card)
        self.assertEqual(card["name"], "gb-2-20260920")
        self.assertEqual(card["bucket"], "running")
        self.assertIn("id", card)
        self.assertIn("badges", card)

    def test_detail_has_heavy_fields(self):
        write_task(self.root, "gb-3-20260920", status="running")
        detail = self._scanner().detail_for_root(self.root / "tasks" / "gb-3-20260920")
        self.assertIn("sides", detail)
        self.assertIn("candidates", detail)
        self.assertIn("workflow", detail)
        self.assertEqual(detail["promptText"], "题目提示词")
        self.assertEqual(detail["workflow"]["total"], 9)

    def test_bucket_classification(self):
        write_task(self.root, "gb-done", status="complete")
        write_task(self.root, "gb-blocked", status="blocked")
        write_task(self.root, "gb-idle", status="prepared", sides={"A": {"status": "idle"}, "B": {"status": "idle"}})
        cards = {card["name"]: card for card in self._scanner().cards([self.root / "tasks"])}
        self.assertEqual(cards["gb-done"]["bucket"], "finished")
        self.assertEqual(cards["gb-blocked"]["bucket"], "failed")
        self.assertEqual(cards["gb-idle"]["bucket"], "waiting")

    def test_summary_counts_buckets(self):
        write_task(self.root, "gb-a", status="complete")
        write_task(self.root, "gb-b", status="running")
        cards = self._scanner().cards([self.root / "tasks"])
        summary = self._scanner().summary(cards)
        self.assertEqual(summary["tasks"], 2)
        self.assertEqual(summary["finished"], 1)
        self.assertEqual(summary["active"], 1)

    def test_container_count_uses_the_directory_name(self):
        """state.taskName can hold a placeholder; the directory is the identity.

        Containers are named from safe_slug(task_root.name), so trusting a
        recorded taskName of "task" made the container count silently read zero.
        """
        task_root = write_task(self.root, "gb-1-20260920-120000-abc", status="running")
        state_path = task_root / "monitor" / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["taskName"] = "task"
        state_path.write_text(json.dumps(state), encoding="utf-8")

        docker = {"items": [
            {"name": "sologsb-gb-1-20260920-120000-abc-candidate-1-1790000000-abc", "state": "running"},
            {"name": "sologsb-gb-1-20260920-120000-abc-candidate-2-1790000000-def", "state": "running"},
            {"name": "sologsb-task-candidate-1-1790000000-zzz", "state": "running"},
            {"name": "sologsb-gb-1-20260920-120000-abc-candidate-1-1790000000-old", "state": "exited"},
        ]}
        scanner = self._scanner()
        card = scanner.card(task_root, docker=docker)
        self.assertEqual(card["name"], "gb-1-20260920-120000-abc")
        self.assertEqual(card["activeContainers"], 2)

    def test_container_group_name(self):
        self.assertEqual(container_group_name("sologsb-gb-1-20260920-120000-abc-candidate-1"), "gb-1-20260920-120000-abc")
        self.assertEqual(container_group_name("postgres"), "")

    def test_task_container_names_filters_by_prefix_and_state(self):
        docker = {"items": [
            {"name": "sologsb-gb-1-20260920-a-1", "state": "running"},
            {"name": "sologsb-gb-1-20260920-a-2", "state": "exited"},
            {"name": "postgres", "state": "running"},
        ]}
        self.assertEqual(task_container_names("gb-1-20260920", docker), ["sologsb-gb-1-20260920-a-1"])


class UnitTests(SchedulerTestCase):
    """Candidates are the entities; A/B are markers on them."""

    def _scanner(self):
        docker = DockerCache(ttl=0)
        docker._data = {"items": [], "byName": {}, "error": "", "fetchedOk": True}
        return TaskScanner(self.config, docker_cache=docker)

    def _task(self, name, candidates, sides=None):
        task_root = write_task(self.root, name, status="running")
        state_path = task_root / "monitor" / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["candidateIds"] = list(candidates)
        state["candidates"] = {
            key: {"candidateId": key, "status": value["status"],
                  "mappedSide": value.get("mappedSide", ""),
                  "completionOrder": value.get("completionOrder")}
            for key, value in candidates.items()
        }
        if sides is not None:
            state["sides"] = sides
        state_path.write_text(json.dumps(state), encoding="utf-8")
        return task_root

    def test_two_candidates_become_a_and_b(self):
        self._task("gb-1-20260920", {
            "candidate-1": {"status": "staged", "mappedSide": "B", "completionOrder": 2},
            "candidate-2": {"status": "staged", "mappedSide": "A", "completionOrder": 1},
        })
        detail = self._scanner().detail_for_root(self.root / "tasks" / "gb-1-20260920")
        self.assertEqual([unit["label"] for unit in detail["units"]], ["B", "A"])
        self.assertEqual([unit["mappedSide"] for unit in detail["units"]], ["B", "A"])
        # The card badge carries the mapped letter too, so A and B are never
        # rendered as a separate list alongside the candidates.
        self.assertEqual([badge["label"] for badge in detail["badges"]], ["B", "A"])

    def test_unmapped_candidates_show_their_number(self):
        self._task("gb-2-20260920", {
            "candidate-1": {"status": "running"},
            "candidate-2": {"status": "running"},
            "candidate-3": {"status": "idle"},
        })
        detail = self._scanner().detail_for_root(self.root / "tasks" / "gb-2-20260920")
        self.assertEqual([unit["label"] for unit in detail["units"]], ["1", "2", "3"])
        self.assertEqual([badge["label"] for badge in detail["badges"]], ["1", "2", "3"])

    def test_four_candidates_all_render_on_one_page(self):
        self._task("gb-3-20260920", {
            f"candidate-{index}": {"status": "running"} for index in range(1, 5)
        })
        detail = self._scanner().detail_for_root(self.root / "tasks" / "gb-3-20260920")
        self.assertEqual(len(detail["units"]), 4)
        self.assertEqual(len(detail["badges"]), 4)

    def test_no_candidates_falls_back_to_the_two_sides(self):
        self._task("gb-4-20260920", {}, sides={
            "A": {"status": "running", "attempt": 1, "runPid": os.getpid()},
            "B": {"status": "idle"},
        })
        detail = self._scanner().detail_for_root(self.root / "tasks" / "gb-4-20260920")
        self.assertEqual([unit["key"] for unit in detail["units"]], ["A"])
        self.assertEqual([unit["mappedSide"] for unit in detail["units"]], ["A"])


if __name__ == "__main__":
    unittest.main()
