#!/usr/bin/env python3
"""CLI for the Autonomous Knowledge Execution Agent."""
from __future__ import annotations

import argparse
import json
import shutil
import sys

from agent.actions import ActionExecutor
from agent.agent import Agent
from agent.approval import APPROVE_ALL, DENY_ALL, INTERACTIVE, Approver
from agent.audit import AuditLog
from agent.config import Settings
from agent.knowledge import KnowledgeBase, ensure_database
from agent.llm import create_llm
from agent.memory import LongTermMemory
from agent.tools import build_registry
from agent.ui import ConsolePresenter


def build_agent(settings: Settings, approval_mode: str, verbose: bool) -> Agent:
    kb = KnowledgeBase(settings)
    memory = LongTermMemory(settings.memory_path)
    audit = AuditLog(settings.audit_path)
    registry = build_registry(kb, memory, ActionExecutor(settings, kb))
    return Agent(create_llm(settings), registry, audit, memory, Approver(approval_mode), settings, kb,
                 on_event=ConsolePresenter(verbose))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Autonomous Knowledge Execution Agent (NorthStar AV operations)")
    p.add_argument("request", nargs="?", help="A request to run once. Omit for an interactive session.")
    p.add_argument("--auto-approve", action="store_true", help="Approve critical actions automatically (demo/testing).")
    p.add_argument("--auto-deny", action="store_true", help="Deny critical actions automatically (demo/testing).")
    p.add_argument("--dry-run", action="store_true", help="Reason and plan but never execute actions.")
    p.add_argument("--provider", choices=["anthropic", "openai"], help="Override LLM_PROVIDER.")
    p.add_argument("--model", help="Override LLM_MODEL.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--reset-data", action="store_true", help="Rebuild the operations database from the seed files.")
    p.add_argument("--reset-all", action="store_true", help="Delete the whole runtime folder (db, memory, audit log, outbox).")
    p.add_argument("--show-audit", type=int, metavar="N", help="Print the last N audit-log entries and exit.")
    p.add_argument("--verify-audit", action="store_true", help="Verify the audit log's hash chain and exit.")
    p.add_argument("--show-memory", action="store_true", help="Print long-term memory and exit.")
    args = p.parse_args(argv)

    settings = Settings.from_env()
    if args.provider:
        settings = Settings(**{**settings.__dict__, "provider": args.provider, "model": args.model or ""})
    elif args.model:
        settings.model = args.model
    if args.dry_run:
        settings.dry_run = True

    if args.reset_all and settings.runtime_dir.exists():
        shutil.rmtree(settings.runtime_dir)
        print("Runtime folder cleared.")
    if args.reset_data or args.reset_all:
        ensure_database(settings, force=True)
        print("Operations database rebuilt from seed data.")

    if args.verify_audit:
        ok, msg = AuditLog(settings.audit_path).verify()
        print(("OK: " if ok else "TAMPERED: ") + msg)
        return 0 if ok else 1
    if args.show_audit:
        for e in AuditLog(settings.audit_path).entries()[-args.show_audit:]:
            print(json.dumps(e, ensure_ascii=False))
        return 0
    if args.show_memory:
        for m in LongTermMemory(settings.memory_path).all():
            print(f"- [{m['ts']}] {m['text']}")
        return 0

    mode = APPROVE_ALL if args.auto_approve else DENY_ALL if args.auto_deny else INTERACTIVE
    try:
        agent = build_agent(settings, mode, args.verbose)
    except RuntimeError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2

    if args.request:
        agent.run(args.request)
        return 0

    print(f"OpsAgent ready ({settings.provider}:{settings.model}). Type a request, or 'exit'.")
    while True:
        try:
            request = input("\nrequest> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if request.lower() in {"exit", "quit", ""}:
            if request:
                return 0
            continue
        agent.run(request)


if __name__ == "__main__":
    raise SystemExit(main())
