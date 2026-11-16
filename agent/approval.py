"""Human-in-the-loop approval gate for critical / irreversible actions."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

INTERACTIVE, APPROVE_ALL, DENY_ALL = "interactive", "approve_all", "deny_all"


@dataclass
class Decision:
    approved: bool
    note: str = ""
    mode: str = INTERACTIVE


class Approver:
    def __init__(self, mode: str = INTERACTIVE, input_fn: Callable[[str], str] = input,
                 output_fn: Callable[[str], None] = print):
        self.mode, self.input_fn, self.out = mode, input_fn, output_fn

    def request(self, tool_name: str, args: dict[str, Any]) -> Decision:
        shown = {k: v for k, v in args.items() if k not in {"reasoning", "evidence"}}
        self.out("\n" + "=" * 70)
        self.out(f"  APPROVAL REQUIRED - {tool_name}")
        self.out("=" * 70)
        self.out("  Arguments: " + json.dumps(shown, ensure_ascii=False, indent=2).replace("\n", "\n  "))
        self.out(f"  Agent's reasoning: {args.get('reasoning', '')}")
        for e in args.get("evidence", []):
            self.out(f"    - evidence: {e}")
        if self.mode == APPROVE_ALL:
            self.out("  -> auto-approved (--auto-approve)")
            return Decision(True, "auto-approved", self.mode)
        if self.mode == DENY_ALL:
            self.out("  -> auto-denied (--auto-deny)")
            return Decision(False, "auto-denied", self.mode)
        try:
            answer = self.input_fn("  Approve this action? [y/N]: ").strip().lower()
            if answer in {"y", "yes"}:
                return Decision(True, "approved by operator", self.mode)
            note = self.input_fn("  Reason for denial (optional, the agent will see it): ").strip()
        except EOFError:
            return Decision(False, "no operator input available", self.mode)
        return Decision(False, note or "denied by operator", self.mode)
