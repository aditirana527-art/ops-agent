import json
import unittest

from agent.llm import ToolCall
from agent.approval import Approver
from tests.helpers import FakeLLM, calls, make_agent, text

PO_ARGS = {"supplier_id": "S01", "sku": "P-1001", "quantity": 500, "unit_cost": 4.5,
           "reasoning": "Available stock 140 is below reorder level 150 under policy v2.",
           "evidence": ["db:products P-1001 stock=340 reserved=200", "policy:procurement_policy_v2 Reorder trigger"]}


def po_count(agent):
    return agent.kb.sql_readonly("SELECT COUNT(*) AS n FROM purchase_orders")["rows"][0]["n"]


class AgentFlowTests(unittest.TestCase):
    def script(self):
        return [
            text("1. Check stock\n2. Check policy"),                                   # plan
            calls(ToolCall("a", "query_database", {"sql": "SELECT * FROM products WHERE sku='P-1001'"}),
                  ToolCall("b", "search_policies", {"query": "reorder trigger"}),
                  thought="Gathering facts in parallel."),
            calls(ToolCall("c", "create_purchase_order", PO_ARGS), thought="Reordering."),
            text("**Outcome** done"),                                                  # final
            text('["Operator approved reorders for P-1001 at contract cost."]'),       # reflection
        ]

    def test_approved_action_executes_and_is_audited(self):
        tmp, s, agent = make_agent(self.script(), "approve_all")
        with tmp:
            res = agent.run("Check P-1001 and reorder if needed")
            self.assertEqual(po_count(agent), 3)
            self.assertEqual(res.actions[0]["status"], "ok")
            events = [e["event"] for e in agent.audit.entries()]
            for ev in ["run_start", "plan", "llm_turn", "tool_call", "tool_result",
                       "approval_requested", "approval_decision", "run_end", "memory_saved"]:
                self.assertIn(ev, events)
            self.assertTrue(agent.audit.verify()[0])
            self.assertEqual(len(agent.memory.all()), 1)

    def test_denied_action_does_not_execute_and_llm_sees_denial(self):
        tmp, s, agent = make_agent(self.script(), "deny_all")
        with tmp:
            res = agent.run("Check P-1001 and reorder if needed")
            self.assertEqual(po_count(agent), 2)  # unchanged
            self.assertEqual(res.actions[0]["status"], "denied_by_human")
            last_results = agent.llm.seen[-2]["messages"][-1]["content"][0]["content"]  # fed back before final turn
            self.assertIn("denied_by_human", last_results)

    def test_dry_run_never_mutates(self):
        tmp, s, agent = make_agent(self.script(), "approve_all", dry_run=True)
        with tmp:
            res = agent.run("x")
            self.assertEqual(po_count(agent), 2)
            self.assertEqual(res.actions[0]["status"], "dry_run")

    def test_parallel_reads_and_error_recovery(self):
        script = [
            text("plan"),
            calls(ToolCall("1", "query_database", {"sql": "DROP TABLE products"}),
                  ToolCall("2", "no_such_tool", {}),
                  ToolCall("3", "update_order_status", {"order_id": "O-5005", "new_status": "on_hold"})),  # missing justification
            text("final"), text("[]"),
        ]
        tmp, s, agent = make_agent(script)
        with tmp:
            agent.run("x")
            fed_back = agent.llm.seen[2]["messages"][-1]["content"]
            self.assertEqual(len(fed_back), 3)
            self.assertTrue(all("error" in b["content"] for b in fed_back))
            self.assertEqual(agent.kb.sql_readonly("SELECT COUNT(*) n FROM products")["rows"][0]["n"], 12)

    def test_standard_actions_skip_approval_but_cancel_needs_it(self):
        just = {"reasoning": "Customer is on credit hold, so the order must not be fulfilled.", "evidence": ["json:customers C-104"]}
        script = [
            text("plan"),
            calls(ToolCall("1", "update_order_status", {"order_id": "O-5005", "new_status": "on_hold", **just}),
                  ToolCall("2", "update_order_status", {"order_id": "O-5006", "new_status": "cancelled", **just})),
            text("final"), text("[]"),
        ]
        tmp, s, agent = make_agent(script, "deny_all")
        with tmp:
            res = agent.run("x")
            status = {a["args"]["order_id"]: a["status"] for a in res.actions}
            self.assertEqual(status, {"O-5005": "ok", "O-5006": "denied_by_human"})
            rows = agent.kb.sql_readonly("SELECT order_id, status FROM orders WHERE order_id IN ('O-5005','O-5006')")["rows"]
            self.assertEqual({r["order_id"]: r["status"] for r in rows}, {"O-5005": "on_hold", "O-5006": "pending"})

    def test_step_budget_forces_final_report(self):
        loop = [calls(ToolCall(str(i), "search_policies", {"query": "cancel"})) for i in range(6)]
        script = [text("plan")] + loop + [text("budget final"), text("[]")]
        tmp, s, agent = make_agent(script)
        with tmp:
            res = agent.run("x")
            self.assertEqual(res.final_text, "budget final")
            self.assertEqual(agent.llm.seen[-2]["tool_choice"], "none")

    def test_interactive_denial_reason_reaches_agent(self):
        answers = iter(["n", "Budget freeze this week"])
        tmp, s, agent = make_agent(self.script(), "interactive")
        agent.approver = Approver("interactive", input_fn=lambda _: next(answers), output_fn=lambda *_: None)
        with tmp:
            agent.run("x")
            fed = json.dumps(agent.llm.seen[-2]["messages"][-1]["content"])
            self.assertIn("Budget freeze this week", fed)

    def test_audit_tamper_detection(self):
        tmp, s, agent = make_agent(self.script())
        with tmp:
            agent.run("x")
            self.assertTrue(agent.audit.verify()[0])
            lines = s.audit_path.read_text().splitlines()
            lines[3] = lines[3].replace("run_start", "run_stArt").replace("llm_turn", "llm_Turn").replace("tool_call", "tool_Call")
            s.audit_path.write_text("\n".join(lines) + "\n")
            self.assertFalse(agent.audit.verify()[0])


if __name__ == "__main__":
    unittest.main()
