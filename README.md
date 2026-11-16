# OpsAgent: Autonomous Knowledge Execution Agent

An autonomous AI agent for a (fictional) AV-equipment distributor, **NorthStar AV Distribution**.
Staff give it a plain-English request. It then:

1. **retrieves** facts from several internal knowledge sources,
2. **reasons** over them (including conflicting or missing data),
3. **decides** which actions are appropriate according to company policy,
4. **executes** those actions through tools (with a human approving the critical ones), and
5. **explains** every decision, before and after execution.

All decisions come from an LLM using native tool calling. There are **no hard-coded responses, no
if/else business rules and no static outputs**. Business rules (reorder logic, approval thresholds,
cancellation rules and so on) live in the *policy documents* that the agent reads and reasons over.

---

## 1. Quick start

**Requirements:** Python 3.10+ and an API key for Anthropic (default) or OpenAI.

```bash
# 1. get the code
unzip ops-agent.zip && cd ops-agent          # or: git clone <your-repo-url> && cd ops-agent

# 2. (recommended) virtual environment
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate

# 3. install dependencies
pip install -r requirements.txt

# 4. configure your LLM key
cp .env.example .env                         # Windows: copy .env.example .env
#    then edit .env and set ANTHROPIC_API_KEY (or LLM_PROVIDER=openai + OPENAI_API_KEY)

# 5. run
python main.py "Review stock levels across all active products and take whatever replenishment actions are needed."
```

The first run creates `runtime/` (SQLite database seeded from `data/*.csv`, audit log, memory file and
outbox). Nothing else needs to be set up.

Run with no arguments for an interactive session:

```bash
python main.py
request> Order O-5003 is late. Find out why and handle it.
```

### CLI options

| Flag | Purpose |
|---|---|
| `--auto-approve` / `--auto-deny` | Answer approval prompts automatically (demo / CI). Default is an interactive y/N prompt. |
| `--dry-run` | Full reasoning and planning, but actions are *simulated*, never executed. |
| `--provider openai --model gpt-4o` | Switch LLM provider/model (also via `LLM_PROVIDER`, `LLM_MODEL`). |
| `--reset-data` | Rebuild the operations DB from the seed CSVs. |
| `--reset-all` | Wipe `runtime/` (DB, memory, audit log, outbox) and rebuild. |
| `--show-audit N` | Print the last N audit entries. |
| `--verify-audit` | Verify the audit log's tamper-evident hash chain. |
| `--show-memory` | Print long-term memory. |
| `--verbose` | Longer tool-result previews. |

Run the tests (they need **no API key**; they use a scripted fake LLM to verify the plumbing):

```bash
python -m unittest discover -s tests -t . -v      # or: pytest
```

---

## 2. Architecture

```
                         user request
                              |
                              v
   +--------------------- Agent loop (agent/agent.py) ----------------------+
   |  recall memory --> PLAN (LLM) --> [ LLM turn <-> tool execution ] x N   |
   |                                          |                              |
   |                       +------------------+------------------+           |
   |                       v                                     v           |
   |               read tools + standard actions         CRITICAL actions    |
   |               run IN PARALLEL (thread pool)         -> human approval   |
   |                       |                              (one at a time)    |
   |                       +------------------+------------------+           |
   |                                          v                              |
   |                             results fed back to the LLM                 |
   |   ... final report --> REFLECT (LLM) --> long-term memory               |
   +----------------------------------------------------------------------+
        |                |                 |                   |
        v                v                 v                   v
  Knowledge sources   Action layer     Audit log          Long-term memory
  (agent/knowledge)   (agent/actions)  (hash-chained)     (TF-IDF recall)
  - SQLite DB         - purchase order  runtime/           runtime/memory.jsonl
  - CSV files         - order status    audit_log.jsonl
  - JSON customers    - notification
  - Policy docs       - human-review flag
```

### Knowledge sources (several, used simultaneously)

| Source | Format | Tool | Contents |
|---|---|---|---|
| Operations DB | SQLite | `query_database` (read-only SQL, enforced by an SQLite authorizer) | products, suppliers, orders, order_items, purchase_orders, review_queue |
| Supplier price list, cycle counts | CSV | `search_csv` | contracted costs/lead times/MOQs; physical stock counts |
| Customer master | JSON | `lookup_customer` | tier, credit status, contacts |
| Company policies | Markdown, chunked + TF-IDF index | `search_policies` | procurement (v1 *and* v2), suppliers, fulfilment, inventory accuracy, communication |
| Memory of past runs | JSONL + TF-IDF | `recall_memory` | lessons and operator decisions |

### Actions

| Tool | Effect | Approval |
|---|---|---|
| `create_purchase_order` | inserts a PO | **always** (financial commitment) |
| `update_order_status` | confirmed / on_hold / backordered / cancelled | only `cancelled` (irreversible, releases stock) |
| `send_notification` | writes a message to `runtime/outbox/` (simulated email) | external (customer/supplier) only |
| `flag_for_human_review` | adds to the `review_queue` table | no |

Every action requires the LLM to supply `reasoning` and `evidence`; calls without them are rejected.

---

## 3. How each requirement and bonus item is met

| Requirement / bonus | Where / how |
|---|---|
| Use internal knowledge sources | SQLite + CSV + JSON + policy docs + memory (`agent/knowledge.py`) |
| Answer **only** from internal knowledge | System-prompt grounding rules (rule 1 and 2); all facts come from tools; unknown things are reported as unknown; off-topic requests are declined; findings carry source tags |
| Reason before deciding | Planning call, then step-by-step tool use. The LLM's reasoning text is logged on each turn |
| Pick the action from request + knowledge | The LLM chooses among the action tools; policies are retrieved at run time rather than coded |
| Execute automatically | `agent/actions.py`, invoked straight from the LLM's tool calls |
| Explain why | Mandatory `reasoning` + `evidence` per action and a structured final report (Outcome / Findings / Actions + why / Data issues / Follow-up) |
| **Long-term memory** | After each run a reflection call stores up to 3 durable lessons (operator denials and reasons, data quirks). They are recalled by relevance at the start of later runs and via `recall_memory` |
| **Parallel actions** | Independent tool calls in one LLM turn run concurrently in a `ThreadPoolExecutor`. Critical actions are serialised because each needs a human |
| **Multi-step reasoning and planning** | Separate planner call, then an iterative observe-think-act loop (up to `AGENT_MAX_STEPS`), with a forced final report if the step budget runs out |
| **Human approval** | `agent/approval.py`: interactive prompt, with a **denial reason** that is fed back to the agent; the agent must adapt and not retry |
| **Audit logs** | `runtime/audit_log.jsonl`: run start, plan, every LLM turn, every tool call/result, approval requests/decisions, memory writes. Entries are **hash-chained**, so `--verify-audit` detects edits or deletions |
| **Multiple sources simultaneously** | The agent queries DB + CSV + JSON + policies in the same turn |
| **Incomplete or conflicting information** | The seed data contains deliberate conflicts (see below). The prompt requires naming the conflict, resolving it by policy or recency, stating confidence, and escalating via `flag_for_human_review` when unresolved |
| **Reasoning before execution** | Reasoning is a required argument of every action tool and is shown in the approval prompt |

### Designed-in data problems (to exercise the agent)

* **Conflicting policy versions**: v1 (superseded) says POs above 5,000 USD need manager sign-off. v2 (active) says 2,500, and also switches to *available* stock.
* **Supplier lead time conflict**: supplier master says 7 days for SonicSource; the contracted price list says 14. A policy says the price list wins.
* **Stock discrepancy**: the system says 6 projectors, the newer cycle count says 4.
* **Suspended supplier** with no alternative for two SKUs.
* **Blank `reorder_level`** (P-1009) and an **expired price-list line**.
* **Unknown customer** (C-999), a customer on **credit hold** (C-104), a **discontinued** product, a **late order** and a **written cancellation request**.

---

## 4. Suggested demo requests

| Request | What good behaviour looks like |
|---|---|
| `Review stock levels across all active products and take whatever replenishment actions are needed.` | Uses *available* stock and open POs, applies policy v2 (not v1), raises a PO only where within authority (with approval), flags large POs, suspended-supplier risks, the stock discrepancy and the missing reorder level |
| `Order O-5003 is late. Find out why and handle it.` | Finds the short stock and the overdue PO, spots the 7 vs 14 day lead-time conflict, notifies the customer (approval) and procurement, offers partial shipment (Silver) |
| `The customer on order O-5006 asked to cancel. Process it.` | Verifies the note, cancels (approval required), releases stock, notifies the customer |
| `Can we fulfil order O-5005?` | Finds the credit hold, puts the order on hold, notifies finance, and tells the customer only "pending account review" |
| `Run an end-of-day sweep of all open orders and fix what's wrong.` | Multi-step: late order, credit hold, unknown customer, discontinued product |
| `What is our refund policy for enterprise customers?` | Says it is not in the internal knowledge base and takes no action |
| `What is the capital of France?` | Declines as outside company operations |

Try `--auto-deny` on the first request to see the agent adapt after refusals (for example by flagging
for review). Inspect results afterwards:

```bash
python main.py --show-audit 20
python main.py --verify-audit
ls runtime/outbox/                      # simulated outgoing messages
sqlite3 runtime/ops.db "select * from review_queue;"      # if the sqlite3 CLI is installed
```

---

## 5. Design decisions

* **No agent framework, on purpose.** A transparent loop over native tool calling is easy to audit
  and has few moving parts. It also makes the planning, parallelism, approval and audit mechanics
  explicit rather than hidden inside a library.
* **Rules live in documents, not code.** The only code-level guards are *integrity* checks (the SKU exists,
  quantity > 0, you cannot cancel a shipped order) and the fail-safe approval gate. Whether to order,
  from whom and how much is always the LLM's decision, based on retrieved policy.
* **Fail-safe approval.** Which actions are gated is a property of the *tool* (irreversible or external),
  not something the LLM can opt out of.
* **Read-only SQL is enforced by the database**, not by string filtering alone (read-only connection plus an
  SQLite authorizer that denies everything except SELECT/READ).
* **Retrieval is a dependency-free TF-IDF** index. It is enough for a small policy corpus and keeps
  installation trivial; swap in embeddings or a vector DB behind `TfidfIndex.search` if the corpus grows.
* **Provider-neutral.** One internal message format; adapters for Anthropic and OpenAI.

## 6. Limitations and next steps

* Notifications are written to an outbox folder instead of being emailed.
* Parallel execution is per LLM turn. Dependent steps are sequenced by the model across turns.
* Only one operator approves actions. A production system would add roles and approval timeouts.
* For large corpora: embeddings plus a vector store, and SQL result paging.

## 8. Evaluation and monitoring

`eval/test_traces.jsonl` holds 20 labelled synthetic traces (expected / allowed / forbidden actions, with a rationale each).
Run them against the real agent (needs an API key):

```bash
python -m eval.run_eval                  # all traces
python -m eval.run_eval --only T01 T05   # subset
python -m eval.monitor runtime/audit_log.jsonl    # monitoring report for any audit log
```

Outputs in `eval/out/`: `transcripts.txt`, `labelled_results.json`, `audit_all.jsonl`, `eval_summary.md`
(precision / recall / F1, pass rate by category, forbidden-action violations), `monitoring_report.md` (error rate,
latency, tokens, approvals, policy-before-action compliance, citation rate, alerts) and `evaluation_writeup.md`.
See `SUBMISSION.md` for how these map to the submission form, and `docs/` for the architecture diagram and write-up.

## 7. Project layout

```
main.py                  CLI
agent/agent.py           plan -> act -> approve -> report -> reflect loop
agent/prompts.py         system / planner / reflection prompts
agent/tools.py           tool registry, schemas, risk classification
agent/knowledge.py       DB / CSV / JSON / policy access (read-only)
agent/actions.py         write operations (integrity checks only)
agent/approval.py        human-in-the-loop gate
agent/audit.py           hash-chained audit log
agent/memory.py          long-term memory
agent/retrieval.py       TF-IDF index
agent/llm.py             Anthropic + OpenAI adapters
data/                    seed CSV/JSON + policies/*.md
tests/                   offline tests (scripted fake LLM)
eval/                    labelled traces, evaluation runner, monitoring report
docs/                    architecture diagram, design write-up
SUBMISSION.md            form field-by-field cheat-sheet
```
