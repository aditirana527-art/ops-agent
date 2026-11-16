"""Internal knowledge sources: SQLite DB, CSV files, JSON customer master, policy documents."""
from __future__ import annotations

import csv
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .config import Settings
from .retrieval import TfidfIndex

DDL = """
CREATE TABLE suppliers(
  supplier_id TEXT PRIMARY KEY, name TEXT, email TEXT, lead_time_days INTEGER,
  payment_terms TEXT, rating REAL, status TEXT);
CREATE TABLE products(
  sku TEXT PRIMARY KEY, name TEXT, category TEXT, status TEXT, unit_cost REAL, unit_price REAL,
  stock_on_hand INTEGER, reserved_qty INTEGER, reorder_level INTEGER, reorder_qty INTEGER,
  supplier_id TEXT REFERENCES suppliers(supplier_id), last_updated TEXT);
CREATE TABLE orders(
  order_id TEXT PRIMARY KEY, customer_id TEXT, order_date TEXT, promised_date TEXT,
  status TEXT, notes TEXT);
CREATE TABLE order_items(order_id TEXT, sku TEXT, qty INTEGER, unit_price REAL);
CREATE TABLE purchase_orders(
  po_id TEXT PRIMARY KEY, supplier_id TEXT, sku TEXT, qty INTEGER, unit_cost REAL, est_cost REAL,
  status TEXT, created_at TEXT, created_by TEXT, reasoning TEXT);
CREATE TABLE review_queue(
  review_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, topic TEXT, severity TEXT,
  details TEXT, related_ids TEXT, status TEXT);
"""

# table -> (csv file, integer columns, real columns)
SEED_TABLES = {
    "suppliers": ("suppliers.csv", {"lead_time_days"}, {"rating"}),
    "products": (
        "products.csv",
        {"stock_on_hand", "reserved_qty", "reorder_level", "reorder_qty"},
        {"unit_cost", "unit_price"},
    ),
    "orders": ("orders.csv", set(), set()),
    "order_items": ("order_items.csv", {"qty"}, {"unit_price"}),
    "purchase_orders": ("purchase_orders.csv", {"qty"}, {"unit_cost", "est_cost"}),
}

CSV_SOURCES = {
    "supplier_price_list": "supplier_price_list.csv",
    "warehouse_cycle_counts": "warehouse_cycle_counts.csv",
}


def ensure_database(settings: Settings, force: bool = False) -> None:
    """Create ops.db from the seed CSVs if it does not exist yet."""
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    if settings.db_path.exists():
        if not force:
            return
        settings.db_path.unlink()
    with closing(sqlite3.connect(settings.db_path)) as conn:
        conn.executescript(DDL)
        for table, (fname, int_cols, real_cols) in SEED_TABLES.items():
            with (settings.data_dir / fname).open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            cols = list(rows[0].keys())
            for row in rows:
                for c in cols:
                    v = row[c]
                    if v == "":
                        row[c] = None
                    elif c in int_cols:
                        row[c] = int(v)
                    elif c in real_cols:
                        row[c] = float(v)
            conn.executemany(
                f"INSERT INTO {table}({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                [tuple(r[c] for c in cols) for r in rows],
            )
        conn.commit()


def _parse_policy(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Split a policy markdown file into (text, metadata) chunks, one per '## ' section."""
    raw = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    body_lines = []
    for line in raw.splitlines():
        m = re.match(r"^(Title|Effective|Status):\s*(.+)$", line)
        if m and not body_lines:
            meta[m.group(1).lower()] = m.group(2).strip()
        else:
            body_lines.append(line)
    chunks: list[tuple[str, dict[str, Any]]] = []
    section, buf = "Overview", []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            header = f"{meta.get('title', path.stem)} | Effective {meta.get('effective', '?')} | Status: {meta.get('status', '?')} | Section: {section}"
            chunks.append(
                (
                    f"{header}\n{text}",
                    {
                        "doc": path.stem,
                        "title": meta.get("title", path.stem),
                        "effective": meta.get("effective", ""),
                        "status": meta.get("status", ""),
                        "section": section,
                        "text": text,
                    },
                )
            )

    for line in body_lines:
        if line.startswith("## "):
            flush()
            section, buf = line[3:].strip(), []
        else:
            buf.append(line)
    flush()
    return chunks


class KnowledgeBase:
    """Read-side access to every internal knowledge source."""

    def __init__(self, settings: Settings):
        self.s = settings
        ensure_database(settings)
        self.customers: dict[str, dict[str, Any]] = {
            c["customer_id"]: c
            for c in json.loads((settings.data_dir / "customers.json").read_text(encoding="utf-8"))
        }
        chunks: list[tuple[str, dict[str, Any]]] = []
        for p in sorted(settings.policy_dir.glob("*.md")):
            chunks.extend(_parse_policy(p))
        self.policy_chunks = chunks
        self.policy_index = TfidfIndex(chunks)

    # --- SQL (read-only) -------------------------------------------------
    def sql_readonly(self, sql: str, limit: int = 50) -> dict[str, Any]:
        stmt = sql.strip().rstrip(";").strip()
        if not re.match(r"(?is)^(select|with)\b", stmt):
            raise ValueError("Only SELECT queries are allowed.")
        recursive = getattr(sqlite3, "SQLITE_RECURSIVE", 33)
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, recursive}

        def authorizer(action: int, *_: Any) -> int:
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

        uri = f"file:{self.s.db_path.as_posix()}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=10)) as conn:
            conn.row_factory = sqlite3.Row
            conn.set_authorizer(authorizer)
            cur = conn.execute(stmt)
            rows = cur.fetchmany(limit + 1)
            cols = [d[0] for d in cur.description or []]
        truncated = len(rows) > limit
        rows = rows[:limit]
        return {
            "source": "db:ops.db",
            "columns": cols,
            "rows": [dict(r) for r in rows],
            "row_count": len(rows),
            "truncated": truncated,
        }

    # --- CSV -------------------------------------------------------------
    def search_csv(self, source: str, column: str | None = None, value: str | None = None, limit: int = 50) -> dict[str, Any]:
        if source not in CSV_SOURCES:
            raise ValueError(f"Unknown CSV source '{source}'. Available: {sorted(CSV_SOURCES)}")
        with (self.s.data_dir / CSV_SOURCES[source]).open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
        if column:
            if column not in columns:
                raise ValueError(f"Unknown column '{column}' for {source}. Columns: {columns}")
            if value is None:
                raise ValueError("'value' is required when 'column' is given")
            rows = [r for r in rows if str(r[column]).strip().lower() == str(value).strip().lower()]
        return {
            "source": f"csv:{source}",
            "columns": columns,
            "rows": rows[:limit],
            "row_count": min(len(rows), limit),
            "truncated": len(rows) > limit,
        }

    # --- JSON customer master -------------------------------------------
    def lookup_customer(self, customer_id: str | None = None, name: str | None = None) -> dict[str, Any]:
        if not customer_id and not name:
            raise ValueError("Provide customer_id or name")
        if customer_id:
            hit = self.customers.get(customer_id)
            matches = [hit] if hit else []
        else:
            needle = (name or "").lower()
            matches = [c for c in self.customers.values() if needle in c["name"].lower()]
        out: dict[str, Any] = {"source": "json:customers", "matches": matches}
        if not matches:
            out["note"] = "No matching customer exists in the customer master."
        return out

    # --- Policies --------------------------------------------------------
    def search_policies(self, query: str, top_k: int = 4) -> dict[str, Any]:
        hits = self.policy_index.search(query, k=max(1, min(int(top_k), 8)))
        return {
            "source": "policy",
            "results": [
                {**{k: v for k, v in meta.items()}, "score": score} for score, _, meta in hits
            ],
            "note": "" if hits else "No policy text matched this query.",
        }

    # --- Source description for the system prompt ------------------------
    def describe_sources(self) -> str:
        parts = ["1. SQLite database (tool: query_database, read-only SQL). Schema:\n" + DDL.strip()]
        csv_lines = []
        for name, fname in CSV_SOURCES.items():
            with (self.s.data_dir / fname).open(newline="", encoding="utf-8") as f:
                cols = next(csv.reader(f))
            csv_lines.append(f"   - {name}: columns {cols}")
        parts.append("2. CSV files (tool: search_csv):\n" + "\n".join(csv_lines))
        sample = next(iter(self.customers.values()))
        parts.append(
            f"3. JSON customer master (tool: lookup_customer). Fields: {list(sample.keys())}"
        )
        docs = {}
        for _, meta in self.policy_chunks:
            docs[meta["doc"]] = f"   - {meta['title']} (effective {meta['effective']}; status: {meta['status']})"
        parts.append("4. Policy documents (tool: search_policies):\n" + "\n".join(docs.values()))
        parts.append("5. Long-term memory of past runs and operator decisions (tool: recall_memory).")
        return "\n".join(parts)
