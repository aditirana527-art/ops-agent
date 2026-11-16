"""Console progress output."""
from __future__ import annotations

import json
import sys
from typing import Any


class ConsolePresenter:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.color = sys.stdout.isatty()

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    @staticmethod
    def _short(obj: Any, n: int = 170) -> str:
        s = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        return s if len(s) <= n else s[: n - 3] + "..."

    def __call__(self, kind: str, data: dict[str, Any]) -> None:
        if kind == "plan":
            print(self._c("1;36", "\n[plan]") + "\n" + data["plan"])
        elif kind == "thought" and data.get("text"):
            print(self._c("2", f"\n[reasoning] {self._short(data['text'], 600)}"))
        elif kind == "tool_start":
            tag = {"none": "read", "standard": "action", "critical": "CRITICAL action"}[data["risk"]]
            print(self._c("33", f"  -> {tag}: {data['name']}({self._short(data['args'])})"))
        elif kind == "tool_end":
            msg = self._short(data["result"], 220 if self.verbose else 110)
            print(self._c("2", f"     <- {data['status']} ({data['ms']} ms) {msg}"))
        elif kind == "memory":
            print(self._c("35", f"\n[memory] saved: {data['text']}"))
        elif kind == "final":
            print(self._c("1;32", "\n" + "-" * 70 + "\nFINAL REPORT\n" + "-" * 70))
            print(data["text"])
