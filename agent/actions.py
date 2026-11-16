"""Write-side operations. Each method is exposed to the LLM as an *action* tool.

Only referential-integrity checks live here (does the SKU exist? is the quantity positive?).
Business decisions (should we order? from whom? how much?) are left entirely to the LLM,
which reads the policy documents.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from .config import Settings
from .knowledge import KnowledgeBase


class ToolError(Exception):
    """A tool call failed in a way the LLM should see and adapt to."""


FINAL_STATUSES = {"shipped", "delivered", "cancelled"}
SETTABLE_STATUSES = {"confirmed", "on_hold", "backordered", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ActionExecutor:
    def __init__(self, settings: Settings, kb: KnowledgeBase):
        self.s = settings
        self.kb = kb

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.s.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # --- create_purchase_order ------------------------------------------
    def create_purchase_order(
        self, supplier_id: str, sku: str, quantity: int, reasoning: str, evidence: list[str],
        unit_cost: float | None = None,
    ) -> dict[str, Any]:
        try:
            qty = int(quantity)
        except (TypeError, ValueError):
            raise ToolError("quantity must be an integer")
        if qty <= 0:
            raise ToolError("quantity must be positive")
        with self._tx() as c:
            if not c.execute("SELECT 1 FROM suppliers WHERE supplier_id=?", (supplier_id,)).fetchone():
                raise ToolError(f"Unknown supplier_id '{supplier_id}'")
            prod = c.execute("SELECT * FROM products WHERE sku=?", (sku,)).fetchone()
            if not prod:
                raise ToolError(f"Unknown sku '{sku}'")
            cost = float(unit_cost) if unit_cost is not None else float(prod["unit_cost"])
            n = c.execute("SELECT COUNT(*) FROM purchase_orders").fetchone()[0]
            po_id = f"PO-{9001 + n}"
            est = round(cost * qty, 2)
            c.execute(
                "INSERT INTO purchase_orders VALUES (?,?,?,?,?,?,?,?,?,?)",
                (po_id, supplier_id, sku, qty, cost, est, "sent", self.s.today, "ops-agent", reasoning),
            )
        return {"po_id": po_id, "supplier_id": supplier_id, "sku": sku, "qty": qty,
                "unit_cost": cost, "est_cost": est, "status": "sent", "created_at": self.s.today}

    # --- update_order_status --------------------------------------------
    def update_order_status(
        self, order_id: str, new_status: str, reasoning: str, evidence: list[str]
    ) -> dict[str, Any]:
        if new_status not in SETTABLE_STATUSES:
            raise ToolError(f"new_status must be one of {sorted(SETTABLE_STATUSES)}")
        released: list[dict[str, Any]] = []
        with self._tx() as c:
            order = c.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone()
            if not order:
                raise ToolError(f"Unknown order_id '{order_id}'")
            old = order["status"]
            if old in FINAL_STATUSES:
                raise ToolError(f"Order {order_id} is already '{old}' and cannot be changed")
            if old == new_status:
                raise ToolError(f"Order {order_id} already has status '{old}'")
            if new_status == "cancelled":
                for it in c.execute("SELECT sku, qty FROM order_items WHERE order_id=?", (order_id,)).fetchall():
                    c.execute(
                        "UPDATE products SET reserved_qty = MAX(0, reserved_qty - ?) WHERE sku=?",
                        (it["qty"], it["sku"]),
                    )
                    released.append({"sku": it["sku"], "qty_released": it["qty"]})
            note = f" | [ops-agent {self.s.today}] {old}->{new_status}: {reasoning[:200]}"
            c.execute(
                "UPDATE orders SET status=?, notes=COALESCE(notes,'') || ? WHERE order_id=?",
                (new_status, note, order_id),
            )
        return {"order_id": order_id, "old_status": old, "new_status": new_status,
                "reservations_released": released}

    # --- send_notification ----------------------------------------------
    def send_notification(
        self, recipient_type: str, recipient_id: str, subject: str, body: str,
        reasoning: str, evidence: list[str],
    ) -> dict[str, Any]:
        if recipient_type == "internal":
            if not re.fullmatch(r"[a-z0-9\-]+", recipient_id or ""):
                raise ToolError("internal recipient_id must be a team id such as 'procurement' or 'finance'")
            to = f"{recipient_id}@northstar-av.example"
        elif recipient_type == "customer":
            cust = self.kb.customers.get(recipient_id)
            if not cust:
                raise ToolError(f"Unknown customer_id '{recipient_id}'")
            to = cust["contact_email"]
        elif recipient_type == "supplier":
            with self._tx() as c:
                row = c.execute("SELECT email FROM suppliers WHERE supplier_id=?", (recipient_id,)).fetchone()
            if not row:
                raise ToolError(f"Unknown supplier_id '{recipient_id}'")
            to = row["email"]
        else:
            raise ToolError("recipient_type must be internal, customer or supplier")
        nid = uuid.uuid4().hex[:8]
        self.s.outbox_dir.mkdir(parents=True, exist_ok=True)
        path = self.s.outbox_dir / f"{self.s.today}_{nid}.json"
        path.write_text(
            json.dumps(
                {"notification_id": nid, "created_at": _now(), "to": to, "recipient_type": recipient_type,
                 "recipient_id": recipient_id, "subject": subject, "body": body,
                 "reasoning": reasoning, "evidence": evidence},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {"notification_id": nid, "to": to, "status": "queued_in_outbox", "file": path.name}

    # --- flag_for_human_review ------------------------------------------
    def flag_for_human_review(
        self, topic: str, details: str, severity: str, reasoning: str, evidence: list[str],
        related_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        if severity not in {"low", "medium", "high"}:
            raise ToolError("severity must be low, medium or high")
        full = f"{details}\n\nReasoning: {reasoning}\nEvidence: {json.dumps(evidence, ensure_ascii=False)}"
        with self._tx() as c:
            cur = c.execute(
                "INSERT INTO review_queue(created_at, topic, severity, details, related_ids, status) VALUES (?,?,?,?,?,?)",
                (_now(), topic, severity, full, json.dumps(related_ids or []), "open"),
            )
            rid = cur.lastrowid
        return {"review_id": rid, "status": "open", "severity": severity}
