"""Provider-neutral LLM client with native tool calling.

Internally, messages use Anthropic-style content blocks:
  assistant: {"role": "assistant", "content": [{"type": "text"|"tool_use", ...}]}
  user:      {"role": "user", "content": "text" | [{"type": "tool_result", ...}, {"type": "text", ...}]}
The OpenAI adapter converts to/from this format.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    assistant_message: dict[str, Any]
    stop_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    def chat(
        self, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096, tool_choice: str | None = None,
    ) -> LLMResponse: ...


class AnthropicClient:
    def __init__(self, model: str):
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install the 'anthropic' package: pip install anthropic") from e
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set (see .env.example)")
        self.client = anthropic.Anthropic()
        self.model = model

    def chat(self, system, messages, tools=None, max_tokens=4096, tool_choice=None) -> LLMResponse:
        kwargs: dict[str, Any] = dict(model=self.model, max_tokens=max_tokens, system=system, messages=messages)
        if tools:
            kwargs["tools"] = tools
            if tool_choice == "none":
                kwargs["tool_choice"] = {"type": "none"}
        resp = self.client.messages.create(**kwargs)
        blocks, texts, calls = [], [], []
        for b in resp.content:
            if b.type == "text":
                blocks.append({"type": "text", "text": b.text})
                texts.append(b.text)
            elif b.type == "tool_use":
                blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
                calls.append(ToolCall(b.id, b.name, dict(b.input)))
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        return LLMResponse(
            "\n".join(texts).strip(), calls, {"role": "assistant", "content": blocks},
            resp.stop_reason or "", usage,
        )


class OpenAIClient:
    def __init__(self, model: str):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install the 'openai' package: pip install openai") from e
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
        self.client = OpenAI()
        self.model = model

    @staticmethod
    def _convert(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            content = m["content"]
            if m["role"] == "assistant":
                if isinstance(content, str):
                    out.append({"role": "assistant", "content": content})
                    continue
                text = "\n".join(b["text"] for b in content if b["type"] == "text")
                calls = [
                    {"id": b["id"], "type": "function",
                     "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                    for b in content if b["type"] == "tool_use"
                ]
                msg: dict[str, Any] = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
            else:
                if isinstance(content, str):
                    out.append({"role": "user", "content": content})
                    continue
                for b in content:
                    if b["type"] == "tool_result":
                        out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]})
                text = "\n".join(b["text"] for b in content if b["type"] == "text")
                if text:
                    out.append({"role": "user", "content": text})
        return out

    def chat(self, system, messages, tools=None, max_tokens=4096, tool_choice=None) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}] + self._convert(messages),
            max_completion_tokens=max_tokens,
        )
        if tools:
            kwargs["tools"] = [
                {"type": "function",
                 "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in tools
            ]
            if tool_choice == "none":
                kwargs["tool_choice"] = "none"
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        blocks: list[dict[str, Any]] = []
        calls: list[ToolCall] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            blocks.append({"type": "tool_use", "id": tc.id, "name": tc.function.name, "input": args})
            calls.append(ToolCall(tc.id, tc.function.name, args))
        usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens} if resp.usage else {}
        return LLMResponse((msg.content or "").strip(), calls, {"role": "assistant", "content": blocks},
                           resp.choices[0].finish_reason or "", usage)


def create_llm(settings: Settings) -> LLMClient:
    if settings.provider == "anthropic":
        return AnthropicClient(settings.model)
    if settings.provider == "openai":
        return OpenAIClient(settings.model)
    raise RuntimeError(f"Unsupported LLM_PROVIDER '{settings.provider}' (use 'anthropic' or 'openai')")
