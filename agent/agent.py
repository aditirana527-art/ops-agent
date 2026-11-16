"""The autonomous agent: plan -> (parallel) tool use -> approval gating -> report -> reflect."""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .actions import ToolError
from .approval import Approver
from .audit import AuditLog
from .config import Settings
from .knowledge import KnowledgeBase
from .llm import LLMClient, ToolCall
from .memory import LongTermMemory
from .prompts import PLANNER_SYSTEM, REFLECT_SYSTEM, SYSTEM_PROMPT
from .tools import ACTION, CRITICAL, NONE, Tool, ToolRegistry

MAX_RESULT_CHARS = 6000


@dataclass
class RunResult:
    run_id: str
    request: str
    plan: str
    final_text: str
    actions: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0


def _parse_json_list(text: str) -> list[Any]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


class Agent:
    def __init__(
        self, llm: LLMClient, registry: ToolRegistry, audit: AuditLog, memory: LongTermMemory,
        approver: Approver, settings: Settings, kb: KnowledgeBase,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        self.llm, self.registry, self.audit, self.memory = llm, registry, audit, memory
        self.approver, self.s, self.kb = approver, settings, kb
        self.on_event = on_event or (lambda kind, data: None)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def run(self, request: str) -> RunResult:
        run_id = uuid.uuid4().hex[:8]
        self.audit.record(run_id, "run_start", {
            "request": request, "provider": self.s.provider, "model": self.s.model,
            "dry_run": self.s.dry_run, "today": self.s.today,
        })
        sources = self.kb.describe_sources()
        memories = self.memory.search(request, k=4)
        mem_text = "\n".join(f"- {m['text']}" for m in memories) or "(none)"

        plan = self._make_plan(run_id, request, mem_text, sources)
        system = SYSTEM_PROMPT.replace("{today}", self.s.today).replace("{sources}", sources)
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": (f"## Request\n{request}\n\n## Long-term memory notes\n{mem_text}\n\n"
                        f"## Initial plan (revise if the facts require it)\n{plan}"),
        }]
        specs = self.registry.specs()
        actions: list[dict[str, Any]] = []
        final_text, steps = "", 0

        for steps in range(1, self.s.max_steps + 1):
            resp = self.llm.chat(system, messages, specs, self.s.max_tokens)
            self.audit.record(run_id, "llm_turn", {
                "step": steps, "reasoning_text": resp.text, "usage": resp.usage,
                "tool_calls": [{"id": c.id, "name": c.name, "input": c.input} for c in resp.tool_calls],
            })
            messages.append(resp.assistant_message)
            if not resp.tool_calls:
                final_text = resp.text
                break
            if resp.text:
                self.on_event("thought", {"text": resp.text})
            results = self._execute_batch(run_id, resp.tool_calls, actions)
            messages.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tc.id, "content": self._to_text(res)}
                for tc, res in results
            ]})
        else:
            messages[-1]["content"].append({
                "type": "text",
                "text": "Step budget reached. Write the final report now from what you have. "
                        "Clearly list anything left undone. Do not call tools.",
            })
            resp = self.llm.chat(system, messages, specs, self.s.max_tokens, tool_choice="none")
            final_text = resp.text
            self.audit.record(run_id, "step_budget_exhausted", {"steps": steps})

        self.on_event("final", {"text": final_text})
        self.audit.record(run_id, "run_end", {"steps": steps, "final_report": final_text,
                                              "actions": [{k: a[k] for k in ("tool", "status")} for a in actions]})
        self._reflect(run_id, request, final_text, actions)
        return RunResult(run_id, request, plan, final_text, actions, steps)

    # ------------------------------------------------------------------
    def _make_plan(self, run_id: str, request: str, mem_text: str, sources: str) -> str:
        prompt = f"Request:\n{request}\n\nMemory notes:\n{mem_text}"
        resp = self.llm.chat(PLANNER_SYSTEM.replace("{sources}", sources),
                             [{"role": "user", "content": prompt}], None, 1024)
        plan = resp.text or "(no plan produced)"
        self.audit.record(run_id, "plan", {"plan": plan})
        self.on_event("plan", {"plan": plan})
        return plan

    @staticmethod
    def _to_text(result: dict[str, Any]) -> str:
        s = json.dumps(result, ensure_ascii=False, default=str)
        if len(s) > MAX_RESULT_CHARS:
            s = s[:MAX_RESULT_CHARS] + '... [truncated; narrow the query]'
        return s

    # ------------------------------------------------------------------
    def _execute_batch(self, run_id: str, calls: list[ToolCall], actions: list[dict[str, Any]]):
        results: dict[str, dict[str, Any]] = {}
        parallel: list[tuple[ToolCall, Tool, str]] = []
        gated: list[tuple[ToolCall, Tool, str]] = []
        for tc in calls:
            tool = self.registry.get(tc.name)
            if tool is None:
                results[tc.id] = {"error": f"Unknown tool '{tc.name}'. Available: {self.registry.names()}"}
                self.audit.record(run_id, "tool_rejected", {"name": tc.name, "reason": "unknown tool"})
                continue
            err = self.registry.validate(tool, tc.input)
            if err:
                results[tc.id] = {"error": err}
                self.audit.record(run_id, "tool_rejected", {"name": tc.name, "input": tc.input, "reason": err})
                continue
            risk = tool.risk_for(tc.input)
            (gated if risk == CRITICAL else parallel).append((tc, tool, risk))

        if parallel:  # independent reads and low-risk actions run concurrently
            with ThreadPoolExecutor(max_workers=min(8, len(parallel))) as pool:
                futures = {tc.id: pool.submit(self._run_tool, run_id, tc, tool, risk, actions)
                           for tc, tool, risk in parallel}
                for tid, fut in futures.items():
                    results[tid] = fut.result()

        for tc, tool, risk in gated:  # critical actions: one at a time, each needs a human
            results[tc.id] = self._run_gated(run_id, tc, tool, risk, actions)
        return [(tc, results[tc.id]) for tc in calls]

    def _run_gated(self, run_id, tc: ToolCall, tool: Tool, risk: str, actions) -> dict[str, Any]:
        self.audit.record(run_id, "approval_requested", {"tool": tool.name, "input": tc.input, "risk": risk})
        decision = self.approver.request(tool.name, tc.input)
        self.audit.record(run_id, "approval_decision", {
            "tool": tool.name, "approved": decision.approved, "note": decision.note, "mode": decision.mode})
        if not decision.approved:
            result = {"status": "denied_by_human", "note": decision.note,
                      "instruction": "Do NOT retry this action. Adapt (e.g. flag for human review) and report the denial."}
            with self._lock:
                actions.append({"tool": tool.name, "args": tc.input, "status": "denied_by_human", "result": result})
            self.on_event("tool_end", {"status": "DENIED", "ms": 0, "result": result})
            return result
        return self._run_tool(run_id, tc, tool, risk, actions, approved_by=decision.mode)

    def _run_tool(self, run_id, tc: ToolCall, tool: Tool, risk: str, actions, approved_by: str | None = None):
        self.on_event("tool_start", {"name": tool.name, "args": tc.input, "risk": risk})
        self.audit.record(run_id, "tool_call", {"id": tc.id, "name": tool.name, "input": tc.input,
                                                "kind": tool.kind, "risk": risk, "approved_by": approved_by})
        t0 = time.perf_counter()
        if tool.kind == ACTION and self.s.dry_run:
            result, status = {"status": "dry_run", "note": "Not executed (dry-run mode)."}, "dry_run"
        else:
            try:
                result, status = tool.fn(**tc.input), "ok"
            except (ToolError, ValueError) as e:
                result, status = {"error": str(e)}, "error"
            except TypeError as e:
                result, status = {"error": f"Invalid arguments: {e}"}, "error"
            except Exception as e:  # noqa: BLE001 - surfaced to the LLM, which can adapt
                result, status = {"error": f"{type(e).__name__}: {e}"}, "error"
        ms = int((time.perf_counter() - t0) * 1000)
        self.audit.record(run_id, "tool_result", {"id": tc.id, "name": tool.name, "status": status, "ms": ms,
                                                  "result": result if tool.kind == ACTION else self._to_text(result)[:1500]})
        if tool.kind == ACTION:
            with self._lock:
                actions.append({"tool": tool.name, "args": tc.input, "status": status, "result": result})
        self.on_event("tool_end", {"status": status, "ms": ms, "result": result})
        return result

    # ------------------------------------------------------------------
    def _reflect(self, run_id: str, request: str, final_text: str, actions: list[dict[str, Any]]) -> None:
        """Distil durable lessons into long-term memory."""
        try:
            summary = {
                "request": request,
                "actions": [{"tool": a["tool"], "args": {k: v for k, v in a["args"].items() if k != "evidence"},
                             "status": a["status"], "note": a["result"].get("note") if isinstance(a["result"], dict) else None}
                            for a in actions],
                "final_report": final_text[:3000],
            }
            resp = self.llm.chat(REFLECT_SYSTEM, [{"role": "user", "content": json.dumps(summary, ensure_ascii=False)}],
                                 None, 512)
            for item in _parse_json_list(resp.text)[:3]:
                if isinstance(item, str) and self.memory.add(item, run_id):
                    self.audit.record(run_id, "memory_saved", {"text": item})
                    self.on_event("memory", {"text": item})
        except Exception as e:  # noqa: BLE001 - memory is best-effort
            self.audit.record(run_id, "reflection_failed", {"error": f"{type(e).__name__}: {e}"})
