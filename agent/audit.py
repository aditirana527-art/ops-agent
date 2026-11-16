"""Append-only, hash-chained audit log (JSON Lines).

Every entry stores the hash of the previous entry, so any later edit or deletion
of a line breaks the chain and is detected by `verify()`.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _hash(entry: dict[str, Any]) -> str:
    payload = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0
        self._last = GENESIS
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._seq, self._last = entry["seq"], entry["hash"]

    def record(self, run_id: str, event: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        safe = json.loads(json.dumps(data or {}, default=str, ensure_ascii=False))
        with self._lock:
            self._seq += 1
            entry = {
                "seq": self._seq,
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "run_id": run_id,
                "event": event,
                "data": safe,
                "prev_hash": self._last,
            }
            entry["hash"] = _hash(entry)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._last = entry["hash"]
        return entry

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def verify(self) -> tuple[bool, str]:
        prev, expected_seq = GENESIS, 1
        for entry in self.entries():
            body = {k: v for k, v in entry.items() if k != "hash"}
            if entry["seq"] != expected_seq:
                return False, f"sequence gap or reorder at seq {entry['seq']} (expected {expected_seq})"
            if entry["prev_hash"] != prev:
                return False, f"broken chain at seq {entry['seq']}"
            if _hash(body) != entry["hash"]:
                return False, f"entry content modified at seq {entry['seq']}"
            prev, expected_seq = entry["hash"], expected_seq + 1
        return True, f"audit log intact ({expected_seq - 1} entries)"
