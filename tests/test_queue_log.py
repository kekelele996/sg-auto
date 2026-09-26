import json
import sys
import tempfile
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from queue_log import LogWriter, RolloutLogFollower, format_rollout_event, redact_log_text  # noqa: E402


class QueueLogTests(unittest.TestCase):
    def test_formats_real_execution_events_and_skips_system_messages(self):
        task_name = "cy-999-20260918-120000"
        events = [
            {
                "timestamp": "2026-09-18T01:00:00Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "system"}]},
            },
            {
                "timestamp": "2026-09-18T01:00:01Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": f"任务 {task_name}"}]},
            },
            {
                "timestamp": "2026-09-18T01:00:02Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "开始执行"}]},
            },
            {
                "timestamp": "2026-09-18T01:00:03Z",
                "type": "response_item",
                "payload": {"type": "function_call", "name": "exec_command", "arguments": json.dumps({"cmd": "echo ok"})},
            },
            {
                "timestamp": "2026-09-18T01:00:04Z",
                "type": "response_item",
                "payload": {"type": "function_call_output", "output": "ok"},
            },
        ]
        lines = [line for event in events for line in format_rollout_event(event, task_name)]
        text = "\n".join(lines)
        self.assertNotIn("system", text)
        self.assertIn("[用户] 任务 cy-999", text)
        self.assertIn("[助手] 开始执行", text)
        self.assertIn("[命令] echo ok", text)
        self.assertIn("[输出] ok", text)

    def test_redacts_api_keys(self):
        text = redact_log_text("apikey=test-keychain-service token=sk-abcdefghijk")
        self.assertNotIn("test-keychain-service", text)
        self.assertNotIn("sk-abcdefghijk", text)
        self.assertIn("***", text)

    def test_follower_streams_appended_events(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rollout = root / "rollout.jsonl"
            output = root / "queue.log"
            event = {
                "timestamp": "2026-09-18T01:00:02Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "第一段"}]},
            }
            rollout.write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")
            follower = RolloutLogFollower(rollout, LogWriter(output))
            self.assertEqual(follower.pump(), 1)
            with rollout.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event | {"timestamp": "2026-09-18T01:00:03Z"}, ensure_ascii=False) + "\n")
            self.assertEqual(follower.pump(), 1)
            self.assertEqual(output.read_text(encoding="utf-8").count("第一段"), 2)


if __name__ == "__main__":
    unittest.main()
