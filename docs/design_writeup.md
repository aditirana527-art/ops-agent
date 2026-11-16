---
title: "OpsAgent: Design Decisions and Limitations"
---

## 1. Domain and goal

**Domain:** operations for a fictional distributor of professional audio-video equipment (NorthStar AV Distribution).
**Goal:** an autonomous agent that takes a plain-English operations request (for example "Order O-5003 is late, handle it"),
retrieves facts from several internal knowledge sources, reasons over them under company policy, executes the appropriate
actions, and explains why. It covers replenishment, order fulfilment, cancellations, supplier follow-up and escalation.

## 2. Architecture in one paragraph

A request first triggers recall of relevant long-term memory, then a separate planning call, then an iterative
observe-think-act loop using native LLM tool calling. Independent tool calls in one turn run in parallel. Critical or
irreversible actions are routed to a human approval gate. The run ends with a structured report, and a reflection call stores durable lessons.
Every step is written to a hash-chained audit log. See `architecture_diagram.png`.

## 3. Key design decisions

1. **LLM decides, documents define the rules.** No business rule (reorder triggers, approval thresholds, cancellation
   conditions) is coded. They live in policy documents that the agent retrieves and applies. Code only enforces
   *integrity* (the SKU exists, quantity is positive, a shipped order cannot be cancelled) and the safety gate.
2. **No agent framework.** A transparent loop over native tool calling keeps planning, parallelism, approval and audit
   mechanics explicit and easy to inspect, with few dependencies.
3. **Multiple heterogeneous knowledge sources** (SQLite, CSV, JSON, policy documents, memory) queried together. This allows
   cross-source reasoning, and conflicts between sources can occur.
4. **Conflicts are handled by policy plus escalation.** The seed data deliberately contains conflicts (two procurement
   policy versions, a supplier lead time that differs between the database and the price list, a stock count mismatch, an
   expired price line, a blank reorder level). The agent must name the conflict, resolve it with the source-of-truth and
   recency rules, state its confidence, and flag for a human when it cannot resolve it.
5. **Fail-safe human approval.** Which actions need approval is a property of the tool (financial commitments, cancellations,
   external messages), so the model cannot opt out. A denial and its reason are fed back to the agent, which must adapt rather than retry.
6. **Reasoning before execution.** Every action tool requires `reasoning` and `evidence`; calls without them are rejected. The
   reasoning appears in the approval prompt and in the audit log.
7. **Defence in depth for data access.** SQL is read-only, enforced by a read-only connection plus an SQLite authorizer rather
   than string filtering. Writes go through four validated action tools only.
8. **Auditability.** Each audit entry contains the hash of the previous one, so edits or deletions are detectable
   (`--verify-audit`).
9. **Long-term memory by reflection.** After each run an LLM call extracts at most three durable lessons (operator
   preferences, denial reasons, data quirks) that are recalled by relevance in later runs. Transient numbers are excluded.
10. **Provider-neutral.** One internal message format with adapters for Anthropic and OpenAI.

## 4. Assumptions and mock data

* All data is **synthetic**: 12 products, 5 suppliers, 9 orders, 6 customers, 2 open purchase orders, 6 policy documents.
* "Sending" a notification writes a JSON file to `runtime/outbox/`. No real email, ERP or payment system is touched.
* Currency is USD. The 2,500 USD agent authority limit and other thresholds are *policy-document content*, not code.
* One human operator approves actions at the console. "Today" defaults to the real date and can be pinned with `AGENT_TODAY`.
* Seed data is reset for every evaluation trace so traces are independent.

## 5. Limitations

* **Non-determinism:** LLM behaviour varies between runs; evaluation numbers should be averaged over repeated runs.
* **Small synthetic evaluation set** (20 traces) labelled by hand, so it is not a statistically strong benchmark.
* **Retrieval** is TF-IDF keyword matching. It works for a small corpus but would need embeddings and re-ranking at scale.
* **Parallelism** is per LLM turn: dependent steps are sequenced by the model across turns, and critical actions are serialised.
* **Integrations are simulated**, with no idempotency keys, retries or transactional outbox.
* **Prompt injection:** free-text fields (order notes, supplier emails) are untrusted but not specially sanitised.
* **Approvals** are a single-operator console prompt, with no roles, timeouts or delegation.
* The step budget and token budget are fixed; very large queries are truncated and the agent must narrow them.

## 6. Next steps

Embedding-based retrieval, repeated-run evaluation with confidence intervals, real ERP/email integrations behind idempotency
keys, role-based approvals with timeouts, injection red-teaming, and a CI gate on recall and forbidden-action violations.
