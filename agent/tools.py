"""Tool registry: the interface between the LLM and the knowledge / action layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .actions import ActionExecutor
from .knowledge import CSV_SOURCES, KnowledgeBase
from .memory import LongTermMemory

READ, ACTION = "read", "action"
NONE, STANDARD, CRITICAL = "none", "standard", "critical"


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    kind: str = READ
    risk: Callable[[dict[str, Any]], str] = lambda args: NONE

    def risk_for(self, args: dict[str, Any]) -> str:
        return self.risk(args)

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.input_schema}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    @staticmethod
    def validate(tool: Tool, args: Any) -> str | None:
        """Return an error message if the arguments are unusable, else None."""
        if not isinstance(args, dict):
            return "Tool input must be a JSON object."
        for key in tool.input_schema.get("required", []):
            if args.get(key) in (None, "", []):
                return f"Missing required argument '{key}'."
        if tool.kind == ACTION:
            if len(str(args.get("reasoning", "")).strip()) < 20:
                return "Actions require a substantive 'reasoning' (why this action, why now, alternatives rejected)."
            ev = args.get("evidence")
            if not isinstance(ev, list) or not ev or not all(isinstance(e, str) and e.strip() for e in ev):
                return "Actions require 'evidence': a non-empty list of strings citing the data/policy relied on."
        return None


_JUSTIFY = {
    "reasoning": {
        "type": "string",
        "description": "Why this action, why now, and which alternatives you rejected. Written BEFORE execution.",
    },
    "evidence": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Citations of the facts/policies relied on, e.g. 'db:products P-1001 stock_on_hand=340', 'policy:procurement_policy_v2 Reorder trigger'.",
    },
}


def build_registry(kb: KnowledgeBase, memory: LongTermMemory, actions: ActionExecutor) -> ToolRegistry:
    reg = ToolRegistry()

    # ---- read tools ------------------------------------------------------
    reg.register(Tool(
        "query_database",
        "Run ONE read-only SQL SELECT against the operations SQLite database (tables: suppliers, products, "
        "orders, order_items, purchase_orders, review_queue). Max 50 rows. Use JOINs and aggregates to get "
        "what you need in one call.",
        {"type": "object", "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
         "required": ["sql"]},
        lambda sql: kb.sql_readonly(sql),
    ))
    reg.register(Tool(
        "search_csv",
        "Look up rows in a CSV knowledge file. Sources: " + ", ".join(sorted(CSV_SOURCES)) + ". "
        "Optionally filter with an exact (case-insensitive) match on one column; omit column/value to list rows.",
        {"type": "object", "properties": {
            "source": {"type": "string", "enum": sorted(CSV_SOURCES)},
            "column": {"type": "string"}, "value": {"type": "string"}},
         "required": ["source"]},
        lambda source, column=None, value=None: kb.search_csv(source, column, value),
    ))
    reg.register(Tool(
        "lookup_customer",
        "Look up a customer in the JSON customer master by customer_id or by (partial) name. "
        "Returns tier, credit status, contact email and account manager.",
        {"type": "object", "properties": {"customer_id": {"type": "string"}, "name": {"type": "string"}}},
        lambda customer_id=None, name=None: kb.lookup_customer(customer_id, name),
    ))
    reg.register(Tool(
        "search_policies",
        "Semantic-style keyword search over the company policy documents. Returns the most relevant policy "
        "sections with their effective date and status. Several policy versions may exist; check dates/status.",
        {"type": "object", "properties": {
            "query": {"type": "string"}, "top_k": {"type": "integer", "description": "1-8, default 4"}},
         "required": ["query"]},
        lambda query, top_k=4: kb.search_policies(query, top_k),
    ))
    reg.register(Tool(
        "recall_memory",
        "Search long-term memory: lessons, data quirks and operator decisions saved from earlier runs.",
        {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        lambda query: {"source": "memory", "results": memory.search(query, 5)},
    ))

    # ---- action tools ----------------------------------------------------
    reg.register(Tool(
        "create_purchase_order",
        "Create (send) a purchase order to a supplier. Financial commitment: ALWAYS needs human approval.",
        {"type": "object", "properties": {
            "supplier_id": {"type": "string"}, "sku": {"type": "string"},
            "quantity": {"type": "integer"},
            "unit_cost": {"type": "number", "description": "Contract unit cost to use; defaults to the product's unit_cost."},
            **_JUSTIFY},
         "required": ["supplier_id", "sku", "quantity", "reasoning", "evidence"]},
        actions.create_purchase_order, ACTION, lambda a: CRITICAL,
    ))
    reg.register(Tool(
        "update_order_status",
        "Change a customer order's status to confirmed, on_hold, backordered or cancelled. Cancelling is "
        "irreversible (releases reserved stock) and needs human approval; other changes execute immediately.",
        {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "new_status": {"type": "string", "enum": ["confirmed", "on_hold", "backordered", "cancelled"]},
            **_JUSTIFY},
         "required": ["order_id", "new_status", "reasoning", "evidence"]},
        actions.update_order_status, ACTION,
        lambda a: CRITICAL if a.get("new_status") == "cancelled" else STANDARD,
    ))
    reg.register(Tool(
        "send_notification",
        "Send a message. recipient_type 'internal' (recipient_id = team id e.g. procurement, finance, warehouse, "
        "sales-ops, management) executes immediately; 'customer' (customer_id) and 'supplier' (supplier_id) "
        "are external and irreversible, so they need human approval.",
        {"type": "object", "properties": {
            "recipient_type": {"type": "string", "enum": ["internal", "customer", "supplier"]},
            "recipient_id": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"},
            **_JUSTIFY},
         "required": ["recipient_type", "recipient_id", "subject", "body", "reasoning", "evidence"]},
        actions.send_notification, ACTION,
        lambda a: STANDARD if a.get("recipient_type") == "internal" else CRITICAL,
    ))
    reg.register(Tool(
        "flag_for_human_review",
        "Add an item to the human review queue: unresolved data conflicts, policy gaps, supply risks, or "
        "decisions that need a person (e.g. a draft PO above the agent's authority). Executes immediately.",
        {"type": "object", "properties": {
            "topic": {"type": "string"}, "details": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
            "related_ids": {"type": "array", "items": {"type": "string"}},
            **_JUSTIFY},
         "required": ["topic", "details", "severity", "reasoning", "evidence"]},
        actions.flag_for_human_review, ACTION, lambda a: STANDARD,
    ))
    return reg
