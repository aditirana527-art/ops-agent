"""Test helpers: a scripted stand-in for the LLM so plumbing can be tested offline.

The fake LLM is used ONLY in tests. The real agent never uses scripted behaviour.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from agent.actions import ActionExecutor
from agent.agent import Agent
from agent.approval import Approver
from agent.audit import AuditLog
from agent.config import Settings
from agent.knowledge import KnowledgeBase
from agent.llm import LLMResponse, ToolCall
from agent.memory import LongTermMemory
from agent.tools import build_registry


def text(t: str) -> LLMResponse:
    return LLMResponse(t, [], {"role": "assistant", "content": [{"type": "text", "text": t}]})


def calls(*tcs: ToolCall, thought: str = "") -> LLMResponse:
    blocks = ([{"type": "text", "text": thought}] if thought else []) + [
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.input} for c in tcs
    ]
    return LLMResponse(thought, list(tcs), {"role": "assistant", "content": blocks})


class FakeLLM:
    def __init__(self, script: list[LLMResponse]):
        self.script, self.seen = list(script), []

    def chat(self, system, messages, tools=None, max_tokens=4096, tool_choice=None):
        self.seen.append({"messages": list(messages), "tools": tools, "tool_choice": tool_choice})
        return self.script.pop(0)


def make_env(mode: str = "approve_all", dry_run: bool = False):
    tmp = tempfile.TemporaryDirectory()
    s = Settings(runtime_dir=Path(tmp.name), today="2026-11-16", dry_run=dry_run, max_steps=6)
    kb = KnowledgeBase(s)
    mem = LongTermMemory(s.memory_path)
    audit = AuditLog(s.audit_path)
    reg = build_registry(kb, mem, ActionExecutor(s, kb))
    approver = Approver(mode, output_fn=lambda *_: None)
    return tmp, s, kb, mem, audit, reg, approver


def make_agent(script, mode="approve_all", dry_run=False):
    tmp, s, kb, mem, audit, reg, approver = make_env(mode, dry_run)
    agent = Agent(FakeLLM(script), reg, audit, mem, approver, s, kb)
    return tmp, s, agent
