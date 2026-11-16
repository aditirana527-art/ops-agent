import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agent.llm import OpenAIClient

ROOT = Path(__file__).resolve().parent.parent


class ConversionTests(unittest.TestCase):
    def test_openai_conversion_of_tool_round_trip(self):
        msgs = [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "thinking"},
                {"type": "tool_use", "id": "t1", "name": "query_database", "input": {"sql": "SELECT 1"}},
                {"type": "tool_use", "id": "t2", "name": "search_policies", "input": {"query": "x"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "{}"},
                {"type": "tool_result", "tool_use_id": "t2", "content": "{}"},
                {"type": "text", "text": "budget reached"}]},
        ]
        out = OpenAIClient._convert(msgs)
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "tool", "tool", "user"])
        self.assertEqual(len(out[1]["tool_calls"]), 2)
        self.assertEqual(out[2]["tool_call_id"], "t1")


class CliTests(unittest.TestCase):
    def run_cli(self, *args, env_extra=None):
        with tempfile.TemporaryDirectory() as d:
            env = {**os.environ, "AGENT_RUNTIME_DIR": d, "ANTHROPIC_API_KEY": "", "OPENAI_API_KEY": "", **(env_extra or {})}
            return subprocess.run([sys.executable, "main.py", *args], cwd=ROOT, env=env, capture_output=True, text=True)

    def test_verify_audit_on_empty_log(self):
        r = self.run_cli("--verify-audit")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("intact", r.stdout)

    def test_missing_key_gives_clear_error(self):
        r = self.run_cli("hello")
        self.assertEqual(r.returncode, 2)
        self.assertRegex(r.stderr, "ANTHROPIC_API_KEY|anthropic")

    def test_reset_all(self):
        r = self.run_cli("--reset-all", "--show-memory")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("rebuilt", r.stdout)


if __name__ == "__main__":
    unittest.main()
