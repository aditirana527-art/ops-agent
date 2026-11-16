import io
import unittest

from agent.config import Settings
from agent.llm import ToolCall
from eval.monitor import alerts, analyze, render_markdown
from eval.run_eval import aggregate, load_cases, match_one, render_summary, render_writeup, run_case, score_case
from tests.helpers import FakeLLM, calls, text

from pathlib import Path

CASES = {c["id"]: c for c in load_cases(Path(__file__).resolve().parent.parent / "eval" / "test_traces.jsonl")}
JUST = {"reasoning": "O-5006 has a written cancellation request and has not shipped.", "evidence": ["db:orders O-5006 notes"]}


class ScoringTests(unittest.TestCase):
    def test_traces_file_is_well_formed(self):
        self.assertEqual(len(CASES), 20)
        for c in CASES.values():
            self.assertTrue({"id", "category", "request", "rationale", "expected", "forbidden"} <= set(c), c["id"])

    def test_match_one(self):
        act = {"tool": "send_notification", "args": {"recipient_type": "customer", "recipient_id": "C-105", "body": "Sorry for the delay on P-1003"}, "status": "ok"}
        self.assertTrue(match_one({"tool": "send_notification", "args": {"recipient_id": "c-105"}}, act))
        self.assertTrue(match_one({"tool": "send_notification", "mentions": ["p-1003"]}, act))
        self.assertFalse(match_one({"tool": "send_notification", "mentions": ["41000"]}, act))
        self.assertTrue(match_one({"any_of": [{"tool": "x"}, {"tool": "send_notification"}]}, act))

    def test_perfect_and_bad_runs(self):
        case = CASES["T01"]
        good = [
            {"tool": "update_order_status", "args": {"order_id": "O-5006", "new_status": "cancelled"}, "status": "ok"},
            {"tool": "send_notification", "args": {"recipient_type": "customer", "recipient_id": "C-105"}, "status": "ok"}]
        s = score_case(case, good, "done")
        self.assertTrue(s["passed"])
        self.assertEqual((s["expected_hit"], s["attempted_correct"], s["n_attempted"]), (2, 2, 2))
        bad = good[:1] + [{"tool": "update_order_status", "args": {"order_id": "O-5001", "new_status": "cancelled"}, "status": "ok"}]
        s = score_case(case, bad, "done")
        self.assertFalse(s["passed"])
        self.assertEqual(len(s["forbidden_violations"]), 1)
        self.assertEqual(len(s["missed"]), 1)

    def test_denial_case_checks_execution_not_attempt(self):
        case = CASES["T18"]
        denied = [{"tool": "update_order_status", "args": {"order_id": "O-5006", "new_status": "cancelled"}, "status": "denied_by_human"}]
        self.assertTrue(score_case(case, denied, "x")["passed"])
        executed = [{**denied[0], "status": "ok"}]
        self.assertFalse(score_case(case, executed, "x")["passed"])

    def test_answer_checks(self):
        self.assertTrue(score_case(CASES["T13"], [], "There are 340 units [db:products].")["passed"])
        self.assertFalse(score_case(CASES["T13"], [], "I do not know.")["passed"])

    def test_aggregate_metrics(self):
        r1 = {"id": "a", "category": "x", "score": {"n_expected": 2, "expected_hit": 2, "n_attempted": 2, "attempted_correct": 2,
              "forbidden_violations": [], "executed_violations": [], "passed": True, "unexpected_actions": [], "missed": []}, "steps": 3, "seconds": 1.0}
        r2 = {"id": "b", "category": "x", "score": {"n_expected": 2, "expected_hit": 1, "n_attempted": 2, "attempted_correct": 1,
              "forbidden_violations": [1], "executed_violations": [], "passed": False, "unexpected_actions": [1], "missed": [1]}, "steps": 4, "seconds": 2.0}
        m = aggregate([r1, r2, {"id": "c", "category": "y", "error": "boom"}])
        self.assertAlmostEqual(m["precision"], 0.75)
        self.assertAlmostEqual(m["recall"], 0.75)
        self.assertEqual((m["errors"], m["forbidden_violations"], m["passed"]), (1, 1, 1))
        self.assertIn("Precision", render_summary(m, [r1, r2]))
        self.assertIn("0.75", render_writeup(m))


class EndToEndEvalTests(unittest.TestCase):
    def test_run_case_and_monitoring_with_fake_llm(self):
        case = CASES["T01"]
        script = [
            text("plan"),
            calls(ToolCall("p", "search_policies", {"query": "cancel order"}), ToolCall("o", "query_database", {"sql": "SELECT notes FROM orders WHERE order_id='O-5006'"})),
            calls(ToolCall("c", "update_order_status", {"order_id": "O-5006", "new_status": "cancelled", **JUST}),
                  ToolCall("n", "send_notification", {"recipient_type": "customer", "recipient_id": "C-105", "subject": "Order cancelled", "body": "Your order O-5006 is cancelled.", **JUST})),
            text("**Outcome** cancelled [db:orders] [policy:order_fulfilment_policy / Cancellations]"),
            text('["Customers confirm cancellations by email."]'),
        ]
        rec = run_case(case, Settings(today="2026-11-16"), FakeLLM(script), io.StringIO())
        self.assertTrue(rec["score"]["passed"], rec["score"])
        stats = analyze(rec["audit_entries"])
        self.assertEqual(stats["completed_runs"], 1)
        self.assertEqual(stats["approvals"]["approved"], 2)
        self.assertEqual(stats["policy_checked_before_first_action"]["rate"], "100.0%")
        self.assertEqual(stats["final_reports_with_source_citations"]["count"], 1)
        md = render_markdown(stats)
        self.assertIn("Monitoring Report", md)
        self.assertEqual(alerts(stats), ["No alerts: all monitored thresholds were within limits."])

    def test_monitor_flags_missing_policy_lookup(self):
        case = CASES["T01"]
        script = [text("plan"),
                  calls(ToolCall("c", "update_order_status", {"order_id": "O-5006", "new_status": "on_hold", **JUST})),
                  text("done with no citations"), text("[]")]
        rec = run_case(case, Settings(today="2026-11-16"), FakeLLM(script), io.StringIO())
        a = " ".join(alerts(analyze(rec["audit_entries"])))
        self.assertIn("NOT consulted", a)
        self.assertIn("no source citations", a)


if __name__ == "__main__":
    unittest.main()
