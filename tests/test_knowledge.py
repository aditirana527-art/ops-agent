import sqlite3
import unittest
from contextlib import closing

from agent.actions import ToolError
from agent.retrieval import TfidfIndex
from tests.helpers import make_env


class KnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp, self.s, self.kb, self.mem, self.audit, self.reg, _ = make_env()
        from agent.actions import ActionExecutor
        self.act = ActionExecutor(self.s, self.kb)

    def tearDown(self):
        self.tmp.cleanup()

    def test_sql_read_works_and_blocks_writes(self):
        out = self.kb.sql_readonly("SELECT sku, stock_on_hand FROM products WHERE sku='P-1001'")
        self.assertEqual(out["rows"][0]["stock_on_hand"], 340)
        for bad in ["DELETE FROM products", "SELECT 1; DROP TABLE products",
                    "WITH x AS (SELECT 1) DELETE FROM products", "UPDATE products SET stock_on_hand=0"]:
            with self.assertRaises(Exception, msg=bad):
                self.kb.sql_readonly(bad)
        with closing(sqlite3.connect(self.s.db_path)) as c:
            self.assertEqual(c.execute("SELECT COUNT(*) FROM products").fetchone()[0], 12)

    def test_csv_and_customers(self):
        r = self.kb.search_csv("supplier_price_list", "supplier_id", "s02")
        self.assertEqual(len(r["rows"]), 3)
        self.assertEqual(self.kb.lookup_customer("C-104")["matches"][0]["credit_status"], "hold")
        self.assertIn("note", self.kb.lookup_customer("C-999"))
        with self.assertRaises(ValueError):
            self.kb.search_csv("supplier_price_list", "nope", "x")

    def test_policy_retrieval_finds_relevant_section(self):
        r = self.kb.search_policies("when can a customer cancel an order", 3)["results"]
        self.assertEqual(r[0]["section"], "Cancellations")
        r = self.kb.search_policies("purchase order approval threshold", 4)["results"]
        docs = {x["doc"] for x in r}
        self.assertTrue({"procurement_policy_v1", "procurement_policy_v2"} <= docs)  # both versions surface

    def test_cancel_releases_reservation(self):
        before = self.kb.sql_readonly("SELECT reserved_qty FROM products WHERE sku='P-1009'")["rows"][0]["reserved_qty"]
        res = self.act.update_order_status("O-5006", "cancelled", "Customer requested cancellation in writing.", ["x"])
        after = self.kb.sql_readonly("SELECT reserved_qty FROM products WHERE sku='P-1009'")["rows"][0]["reserved_qty"]
        self.assertEqual(before - after, 20)
        self.assertEqual(len(res["reservations_released"]), 2)
        with self.assertRaises(ToolError):  # cannot cancel twice
            self.act.update_order_status("O-5006", "cancelled", "Customer requested cancellation again.", ["x"])
        with self.assertRaises(ToolError):  # shipped orders are final
            self.act.update_order_status("O-5008", "cancelled", "Trying to cancel a shipped order.", ["x"])

    def test_purchase_order_and_review_and_notification(self):
        po = self.act.create_purchase_order("S01", "P-1001", 500, "Reorder below threshold.", ["x"], 4.5)
        self.assertEqual(po["po_id"], "PO-9003")
        self.assertEqual(po["est_cost"], 2250.0)
        with self.assertRaises(ToolError):
            self.act.create_purchase_order("S99", "P-1001", 5, "Unknown supplier test.", ["x"])
        self.assertEqual(self.act.flag_for_human_review("t", "d", "high", "Some reasoning here.", ["x"])["review_id"], 1)
        n = self.act.send_notification("customer", "C-103", "Delay", "body", "Late order notice.", ["x"])
        self.assertTrue((self.s.outbox_dir / n["file"]).exists())
        with self.assertRaises(ToolError):
            self.act.send_notification("customer", "C-999", "s", "b", "Unknown customer test.", ["x"])

    def test_registry_validation_requires_justification(self):
        tool = self.reg.get("create_purchase_order")
        err = self.reg.validate(tool, {"supplier_id": "S01", "sku": "P-1001", "quantity": 5,
                                       "reasoning": "short", "evidence": ["x"]})
        self.assertIn("reasoning", err)
        ok = self.reg.validate(tool, {"supplier_id": "S01", "sku": "P-1001", "quantity": 5,
                                      "reasoning": "A sufficiently long justification.", "evidence": ["db"]})
        self.assertIsNone(ok)

    def test_memory_roundtrip(self):
        self.mem.add("Operator prefers partial shipments for Gold customers.")
        self.mem.add("Supplier S04 is suspended pending a quality audit.")
        self.assertFalse(self.mem.add("supplier s04 is suspended pending a quality audit."))  # dedupe
        hit = self.mem.search("which suppliers are suspended", 1)[0]["text"]
        self.assertIn("S04", hit)

    def test_tfidf_empty_query(self):
        self.assertEqual(TfidfIndex([("hello world", {})]).search("zzz"), [])


if __name__ == "__main__":
    unittest.main()
