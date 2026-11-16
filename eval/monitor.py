"""Monitoring report generated from the agent's audit log.

Usage:  python -m eval.monitor [runtime/audit_log.jsonl] [-o monitoring_report.md]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CITATION = re.compile(r"\[(db|csv|json|policy|memory)[:\]/ ]", re.I)


def _pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "n/a"


def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def analyze(entries: list[dict[str, Any]]) -> dict[str, Any]:
    runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in entries:
        runs[e["run_id"]].append(e)

    kind_of: dict[tuple[str, str], str] = {}
    calls, results, approvals = Counter(), Counter(), Counter()
    by_tool_status: dict[str, Counter] = defaultdict(Counter)
    latencies: list[float] = []
    tok_in = tok_out = rejected = budget = reflect_fail = mem_saved = 0
    steps: list[int] = []
    approval_requested = approved = denied = 0
    approval_by_tool: dict[str, Counter] = defaultdict(Counter)
    action_runs = policy_first = completed = cited = 0
    action_kinds = {"action"}

    for rid, evs in runs.items():
        evs.sort(key=lambda x: x["seq"])
        first_policy = first_action = None
        for e in evs:
            d, ev = e["data"], e["event"]
            if ev == "tool_call":
                kind_of[(rid, d["id"])] = d["kind"]
                calls[d["name"]] += 1
                if d["name"] == "search_policies" and first_policy is None:
                    first_policy = e["seq"]
                if d["kind"] in action_kinds and first_action is None:
                    first_action = e["seq"]
            elif ev == "tool_result":
                results[d["status"]] += 1
                by_tool_status[d["name"]][d["status"]] += 1
                latencies.append(d["ms"])
            elif ev == "tool_rejected":
                rejected += 1
            elif ev == "llm_turn":
                tok_in += d.get("usage", {}).get("input_tokens", 0)
                tok_out += d.get("usage", {}).get("output_tokens", 0)
            elif ev == "approval_requested":
                approval_requested += 1
            elif ev == "approval_decision":
                (approval_by_tool[d["tool"]]["approved" if d["approved"] else "denied"]) += 1
                approved += d["approved"]
                denied += not d["approved"]
            elif ev == "step_budget_exhausted":
                budget += 1
            elif ev == "reflection_failed":
                reflect_fail += 1
            elif ev == "memory_saved":
                mem_saved += 1
            elif ev == "run_end":
                completed += 1
                steps.append(d["steps"])
                cited += bool(CITATION.search(d.get("final_report", "")))
        if first_action is not None:
            action_runs += 1
            policy_first += first_policy is not None and first_policy < first_action

    total_results = sum(results.values())
    action_tools = {"create_purchase_order", "update_order_status", "send_notification", "flag_for_human_review"}
    n_actions_ok = sum(c["ok"] for t, c in by_tool_status.items() if t in action_tools)
    return {
        "runs": len(runs), "completed_runs": completed, "audit_entries": len(entries),
        "steps_mean": round(statistics.mean(steps), 2) if steps else 0, "steps_max": max(steps, default=0),
        "tool_calls": dict(calls), "tool_results": dict(results),
        "tool_error_rate": _pct(results["error"], total_results),
        "tool_status_by_name": {k: dict(v) for k, v in by_tool_status.items()},
        "rejected_calls": rejected,
        "latency_ms": {"p50": _quantile(latencies, .5), "p95": _quantile(latencies, .95), "max": max(latencies, default=0)},
        "tokens": {"input": tok_in, "output": tok_out, "per_run_avg": round((tok_in + tok_out) / max(len(runs), 1))},
        "approvals": {"requested": approval_requested, "approved": approved, "denied": denied,
                      "by_tool": {k: dict(v) for k, v in approval_by_tool.items()}},
        "actions_executed_ok": n_actions_ok,
        "policy_checked_before_first_action": {"runs_with_actions": action_runs, "compliant": policy_first,
                                               "rate": _pct(policy_first, action_runs)},
        "final_reports_with_source_citations": {"count": cited, "of": completed, "rate": _pct(cited, completed)},
        "step_budget_exhausted": budget, "reflection_failures": reflect_fail, "memories_saved": mem_saved,
    }


def alerts(s: dict[str, Any]) -> list[str]:
    out = []
    tr = s["tool_results"]
    total = sum(tr.values())
    if total and tr.get("error", 0) / total > 0.10:
        out.append(f"Tool error rate is {s['tool_error_rate']} (> 10%).")
    if s["rejected_calls"]:
        out.append(f"{s['rejected_calls']} tool call(s) were rejected for missing/invalid arguments (e.g. no reasoning/evidence).")
    p = s["policy_checked_before_first_action"]
    if p["runs_with_actions"] and p["compliant"] < p["runs_with_actions"]:
        out.append(f"Policy was NOT consulted before acting in {p['runs_with_actions'] - p['compliant']} run(s).")
    if s["step_budget_exhausted"]:
        out.append(f"{s['step_budget_exhausted']} run(s) hit the step budget.")
    c = s["final_reports_with_source_citations"]
    if c["of"] and c["count"] < c["of"]:
        out.append(f"{c['of'] - c['count']} final report(s) contain no source citations (grounding risk).")
    if s["reflection_failures"]:
        out.append(f"{s['reflection_failures']} memory-reflection failure(s).")
    return out or ["No alerts: all monitored thresholds were within limits."]


def render_markdown(s: dict[str, Any], title: str = "Monitoring Report") -> str:
    L = [f"# {title}", "", f"Source: agent audit log ({s['audit_entries']} entries, {s['runs']} runs, {s['completed_runs']} completed).", "",
         "## Alerts", ""] + [f"- {a}" for a in alerts(s)] + ["", "## Key indicators", "",
         "| Indicator | Value |", "|---|---|",
         f"| Runs completed | {s['completed_runs']} / {s['runs']} |",
         f"| Steps per run (mean / max) | {s['steps_mean']} / {s['steps_max']} |",
         f"| Tool error rate | {s['tool_error_rate']} |",
         f"| Rejected tool calls (validation) | {s['rejected_calls']} |",
         f"| Tool latency p50 / p95 / max (ms) | {s['latency_ms']['p50']} / {s['latency_ms']['p95']} / {s['latency_ms']['max']} |",
         f"| Tokens in / out (total) | {s['tokens']['input']} / {s['tokens']['output']} |",
         f"| Tokens per run (avg) | {s['tokens']['per_run_avg']} |",
         f"| Human approvals requested / approved / denied | {s['approvals']['requested']} / {s['approvals']['approved']} / {s['approvals']['denied']} |",
         f"| Policy consulted before first action | {s['policy_checked_before_first_action']['compliant']} / {s['policy_checked_before_first_action']['runs_with_actions']} runs ({s['policy_checked_before_first_action']['rate']}) |",
         f"| Final reports citing sources | {s['final_reports_with_source_citations']['count']} / {s['final_reports_with_source_citations']['of']} ({s['final_reports_with_source_citations']['rate']}) |",
         f"| Step-budget exhaustions | {s['step_budget_exhausted']} |",
         f"| Memories saved | {s['memories_saved']} |", "", "## Tool usage", "",
         "| Tool | Calls | ok | error | dry_run |", "|---|---|---|---|---|"]
    for name, n in sorted(s["tool_calls"].items(), key=lambda x: -x[1]):
        st = s["tool_status_by_name"].get(name, {})
        L.append(f"| {name} | {n} | {st.get('ok', 0)} | {st.get('error', 0)} | {st.get('dry_run', 0)} |")
    if s["approvals"]["by_tool"]:
        L += ["", "## Approvals by tool", "", "| Tool | Approved | Denied |", "|---|---|---|"]
        for t, c in s["approvals"]["by_tool"].items():
            L.append(f"| {t} | {c.get('approved', 0)} | {c.get('denied', 0)} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit", nargs="?", default="runtime/audit_log.jsonl")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    entries = [json.loads(l) for l in Path(a.audit).read_text(encoding="utf-8").splitlines() if l.strip()]
    md = render_markdown(analyze(entries))
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
