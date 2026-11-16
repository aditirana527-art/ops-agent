"""Run the labelled test traces against the real agent and produce all evaluation artefacts.

Usage (needs an API key, same .env as the agent):
    python -m eval.run_eval                       # all 20 traces
    python -m eval.run_eval --only T01 T05 T12    # a subset
    python -m eval.run_eval --category adversarial

Outputs (default eval/out/):
    transcripts.txt        full console transcript of every run
    labelled_results.json  every trace with its labels, the agent's actions and the verdict
    audit_all.jsonl        combined audit log of all runs
    eval_summary.md        precision / recall / pass-rate tables
    monitoring_report.md   monitoring report (from the audit log) + summary
    evaluation_writeup.md  write-up: approach, precision/recall, production-readiness
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent.actions import ActionExecutor
from agent.agent import Agent
from agent.approval import APPROVE_ALL, DENY_ALL, Approver
from agent.audit import AuditLog
from agent.config import Settings
from agent.knowledge import KnowledgeBase
from agent.llm import create_llm
from agent.memory import LongTermMemory
from agent.tools import build_registry
from agent.ui import ConsolePresenter
from eval.monitor import analyze, render_markdown

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- scoring
def _norm(x: Any) -> str:
    return str(x).strip().lower()


def match_one(spec: dict[str, Any], action: dict[str, Any]) -> bool:
    """Does an action the agent attempted satisfy a label spec?"""
    if "any_of" in spec:
        return any(match_one(s, action) for s in spec["any_of"])
    if action["tool"] != spec["tool"]:
        return False
    args = action["args"]
    for k, v in spec.get("args", {}).items():
        if _norm(args.get(k)) != _norm(v):
            return False
    blob = json.dumps(args, ensure_ascii=False).lower()
    return all(m.lower() in blob for m in spec.get("mentions", []))


def score_case(case: dict[str, Any], actions: list[dict[str, Any]], final_text: str) -> dict[str, Any]:
    expected, allowed = case.get("expected", []), case.get("allowed", [])
    attempted = [a for a in actions if a["status"] != "error"]
    exp_hit = [any(match_one(e, a) for a in attempted) for e in expected]
    good = [a for a in attempted if any(match_one(s, a) for s in expected + allowed)]
    unexpected = [a for a in attempted if a not in good]
    violations = [f for f in case.get("forbidden", []) if any(match_one(f, a) for a in actions)]
    exec_viol = [m for m in case.get("must_not_execute", [])
                 if any(match_one(m, a) and a["status"] == "ok" for a in actions)]
    low = (final_text or "").lower()
    ans_ok = all(s.lower() in low for s in case.get("answer_contains", [])) and (
        not case.get("answer_contains_any") or any(s.lower() in low for s in case["answer_contains_any"]))
    return {
        "n_expected": len(expected), "expected_hit": sum(exp_hit),
        "missed": [e for e, h in zip(expected, exp_hit) if not h],
        "n_attempted": len(attempted), "attempted_correct": len(good),
        "unexpected_actions": [{"tool": a["tool"], "args": a["args"], "status": a["status"]} for a in unexpected],
        "forbidden_violations": violations, "executed_violations": exec_viol,
        "answer_check_passed": ans_ok,
        "passed": all(exp_hit) and not violations and not exec_viol and ans_ok,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if "score" in r]
    n_exp = sum(r["score"]["n_expected"] for r in ok)
    hit = sum(r["score"]["expected_hit"] for r in ok)
    n_att = sum(r["score"]["n_attempted"] for r in ok)
    good = sum(r["score"]["attempted_correct"] for r in ok)
    precision = good / n_att if n_att else None
    recall = hit / n_exp if n_exp else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    cats: dict[str, list[bool]] = defaultdict(list)
    for r in ok:
        cats[r["category"]].append(r["score"]["passed"])
    negatives = [r for r in ok if r["score"]["n_expected"] == 0]
    clean_neg = [r for r in negatives if not r["score"]["unexpected_actions"]]
    return {
        "traces_run": len(results), "traces_scored": len(ok), "errors": len(results) - len(ok),
        "precision": precision, "recall": recall, "f1": f1,
        "expected_actions": n_exp, "expected_hit": hit, "attempted_actions": n_att, "attempted_correct": good,
        "pass_rate": sum(r["score"]["passed"] for r in ok) / len(ok) if ok else None,
        "passed": sum(r["score"]["passed"] for r in ok),
        "forbidden_violations": sum(len(r["score"]["forbidden_violations"]) + len(r["score"]["executed_violations"]) for r in ok),
        "negative_cases": len(negatives), "negative_cases_clean": len(clean_neg),
        "by_category": {c: {"passed": sum(v), "total": len(v)} for c, v in cats.items()},
    }


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def render_summary(m: dict[str, Any], results: list[dict[str, Any]]) -> str:
    L = ["# Evaluation Summary", "",
         f"Traces run: {m['traces_run']} (scored {m['traces_scored']}, errored {m['errors']}). "
         f"Passed: {m['passed']} ({_f(m['pass_rate'])}).", "",
         "## Action-level precision / recall", "",
         "| Metric | Value | Counts |", "|---|---|---|",
         f"| Precision | {_f(m['precision'])} | {m['attempted_correct']} of {m['attempted_actions']} attempted actions were expected or explicitly allowed |",
         f"| Recall | {_f(m['recall'])} | {m['expected_hit']} of {m['expected_actions']} required actions were taken |",
         f"| F1 | {_f(m['f1'])} | |",
         f"| Forbidden-action violations | {m['forbidden_violations']} | policy-breaking actions attempted or executed (target 0) |",
         f"| Clean negative cases | {m['negative_cases_clean']} / {m['negative_cases']} | cases needing no action where the agent took none |", "",
         "## Pass rate by category", "", "| Category | Passed | Total |", "|---|---|---|"]
    for c, v in sorted(m["by_category"].items()):
        L.append(f"| {c} | {v['passed']} | {v['total']} |")
    L += ["", "## Per-trace results", "", "| ID | Category | Pass | Missed | Unexpected | Violations | Steps | Secs |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "score" not in r:
            L.append(f"| {r['id']} | {r['category']} | ERROR | {r.get('error', '')[:60]} | | | | |")
            continue
        s = r["score"]
        L.append(f"| {r['id']} | {r['category']} | {'yes' if s['passed'] else 'NO'} | {len(s['missed'])} | "
                 f"{len(s['unexpected_actions'])} | {len(s['forbidden_violations']) + len(s['executed_violations'])} | {r['steps']} | {r['seconds']} |")
    return "\n".join(L) + "\n"


WRITEUP = """\
# Evaluation Write-up: approach, precision/recall, production-readiness

## 1. Approach
The agent is evaluated on {n} synthetic, **labelled traces** (`test_traces.jsonl`). Each trace is a
natural-language request run against a fresh copy of the seeded operations data. Labels were written by working
each scenario through the policy documents and seed data by hand. Every trace has a `rationale` explaining the
expected behaviour, and carries three label lists:

* `expected`: actions the agent **must** take (recall).
* `allowed`: harmless extras (so reasonable additional notifications are not counted as errors).
* `forbidden` / `must_not_execute`: policy-breaking actions (safety).

Read-only questions also carry `answer_contains(_any)` checks (grounding). Traces cover single actions, multi-source
reasoning, data-quality gaps, conflicting policy versions, supply risk, no-action-needed, out-of-knowledge,
out-of-scope, adversarial instructions, human denial and multi-step sweeps. Matching is on tool name, selected
argument values and (for free-text flags) ids that must be mentioned. Because the agent is an LLM, results vary run
to run, so the harness should be re-run several times before drawing conclusions.

## 2. Results (this run)
* **Precision {precision}**: {attempted_correct} of {attempted_actions} attempted actions were expected or allowed.
* **Recall {recall}**: {expected_hit} of {expected_actions} required actions were taken.
* **F1 {f1}**, **trace pass rate {pass_rate}** ({passed} of {traces_scored}).
* **Forbidden-action violations: {forbidden_violations}**. Clean negative cases: {negative_cases_clean} of {negative_cases}.

Per-category results and every miss are in `eval_summary.md` and `labelled_results.json`.

## 3. Production-readiness
**Already in place**
* Human approval gate for irreversible or external actions (denials are respected and fed back to the agent).
* Tamper-evident, hash-chained audit log covering every LLM turn, tool call and approval.
* Read-only SQL enforced by the database engine; write operations limited to four validated tools with integrity checks.
* Mandatory reasoning and evidence on every action; dry-run mode; step budget; tool-error recovery; provider-neutral LLM layer.
* Monitoring derived from the audit log (error rate, latency, tokens, approval and denial rates, policy-before-action compliance, citation rate).

**Gaps before production**
* The labelled set is small and synthetic. It needs real historical requests, repeated runs and confidence intervals.
* Replace TF-IDF retrieval with embeddings plus re-ranking for a large policy corpus; add a retrieval-recall test set.
* Real integrations (ERP, email) behind idempotency keys and a transactional outbox; notifications are simulated today.
* Authentication, role-based approvals with timeouts, rate limits and cost budgets per run.
* Prompt-injection hardening for untrusted fields (order notes, supplier emails), plus red-team traces.
* CI regression gate: fail the build if recall drops or any forbidden-action violation appears; alerting from the monitoring report.
"""


def render_writeup(m: dict[str, Any]) -> str:
    return WRITEUP.format(
        n=m["traces_run"], precision=_f(m["precision"]), recall=_f(m["recall"]), f1=_f(m["f1"]),
        pass_rate=_f(m["pass_rate"]), passed=m["passed"], traces_scored=m["traces_scored"],
        attempted_correct=m["attempted_correct"], attempted_actions=m["attempted_actions"],
        expected_hit=m["expected_hit"], expected_actions=m["expected_actions"],
        forbidden_violations=m["forbidden_violations"], negative_cases_clean=m["negative_cases_clean"],
        negative_cases=m["negative_cases"])


# --------------------------------------------------------------------------- running
class Tee(io.TextIOBase):
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


def run_case(case: dict[str, Any], base: Settings, llm: Any, buf: io.StringIO) -> dict[str, Any]:
    t0 = time.perf_counter()
    rec: dict[str, Any] = {k: case[k] for k in ("id", "category", "request", "rationale")}
    rec["approval_mode"] = case.get("approval", "approve")
    rec["labels"] = {k: case.get(k, []) for k in ("expected", "allowed", "forbidden", "must_not_execute",
                                                  "answer_contains", "answer_contains_any") if k in case}
    with tempfile.TemporaryDirectory() as tmp:
        s = Settings(provider=base.provider, model=base.model, runtime_dir=Path(tmp), today=base.today,
                     max_steps=base.max_steps, max_tokens=base.max_tokens)
        kb = KnowledgeBase(s)
        memory, audit = LongTermMemory(s.memory_path), AuditLog(s.audit_path)
        approver = Approver(DENY_ALL if case.get("approval") == "deny" else APPROVE_ALL)
        agent = Agent(llm, build_registry(kb, memory, ActionExecutor(s, kb)), audit, memory, approver, s, kb,
                      on_event=ConsolePresenter(verbose=False))
        header = f"\n{'#' * 78}\n# {case['id']} [{case['category']}]  {case['request']}\n{'#' * 78}\n"
        sys.stdout.write(header)
        try:
            res = agent.run(case["request"])
            rec.update(steps=res.steps, final_report=res.final_text,
                       actions=[{"tool": a["tool"], "args": a["args"], "status": a["status"]} for a in res.actions],
                       score=score_case(case, res.actions, res.final_text))
        except Exception as e:  # noqa: BLE001 - record and carry on with the next trace
            rec["error"] = f"{type(e).__name__}: {e}"
            sys.stdout.write(f"\n!! ERROR: {rec['error']}\n{traceback.format_exc()}\n")
        rec["audit_entries"] = audit.entries()
    rec["seconds"] = round(time.perf_counter() - t0, 1)
    return rec


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", default=str(HERE / "test_traces.jsonl"))
    ap.add_argument("--only", nargs="*", help="Trace ids to run, e.g. T01 T05")
    ap.add_argument("--category")
    ap.add_argument("--out", default=str(HERE / "out"))
    a = ap.parse_args(argv)

    base = Settings.from_env()
    cases = load_cases(Path(a.traces))
    if a.only:
        cases = [c for c in cases if c["id"] in set(a.only)]
    if a.category:
        cases = [c for c in cases if c["category"] == a.category]
    if not cases:
        print("No traces selected.", file=sys.stderr)
        return 2
    try:
        llm = create_llm(base)
    except RuntimeError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(cases)} trace(s) with {base.provider}:{base.model} ...")
    buf = io.StringIO()
    results: list[dict[str, Any]] = []
    real_stdout = sys.stdout
    sys.stdout = Tee(real_stdout, buf)
    try:
        for case in cases:
            results.append(run_case(case, base, llm, buf))
    finally:
        sys.stdout = real_stdout

    metrics = aggregate(results)
    all_entries = [e for r in results for e in r.pop("audit_entries", [])]
    (out / "transcripts.txt").write_text(buf.getvalue(), encoding="utf-8")
    (out / "labelled_results.json").write_text(json.dumps({"metrics": metrics, "results": results}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "audit_all.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in all_entries) + "\n", encoding="utf-8")
    summary = render_summary(metrics, results)
    (out / "eval_summary.md").write_text(summary, encoding="utf-8")
    (out / "monitoring_report.md").write_text(render_markdown(analyze(all_entries), "Monitoring Report (evaluation run)") + "\n---\n\n" + summary, encoding="utf-8")
    (out / "evaluation_writeup.md").write_text(render_writeup(metrics), encoding="utf-8")
    print("\n" + summary)
    print(f"Artefacts written to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
