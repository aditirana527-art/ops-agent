"""Long-term memory: durable lessons persisted across runs and recalled by relevance."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .retrieval import TfidfIndex


class LongTermMemory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def add(self, text: str, run_id: str = "") -> bool:
        text = " ".join(text.split())
        if not text:
            return False
        with self._lock:
            if any(m["text"].lower() == text.lower() for m in self.all()):
                return False
            item = {
                "id": uuid.uuid4().hex[:8],
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "run_id": run_id,
                "text": text,
            }
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return True

    def search(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        items = self.all()
        if not items:
            return []
        index = TfidfIndex([(m["text"], m) for m in items])
        return [meta for _, _, meta in index.search(query, k)]
