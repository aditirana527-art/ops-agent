"""Runtime settings, loaded from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o"}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # optional dependency
    except ImportError:
        return
    load_dotenv(ROOT / ".env")


@dataclass
class Settings:
    provider: str = "anthropic"
    model: str = ""
    data_dir: Path = ROOT / "data"
    runtime_dir: Path = ROOT / "runtime"
    max_steps: int = 12
    max_tokens: int = 4096
    dry_run: bool = False
    today: str = ""

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.runtime_dir = Path(self.runtime_dir)
        if not self.model:
            self.model = DEFAULT_MODELS.get(self.provider, "")
        if not self.today:
            self.today = date.today().isoformat()

    # Derived paths -----------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.runtime_dir / "ops.db"

    @property
    def audit_path(self) -> Path:
        return self.runtime_dir / "audit_log.jsonl"

    @property
    def memory_path(self) -> Path:
        return self.runtime_dir / "memory.jsonl"

    @property
    def outbox_dir(self) -> Path:
        return self.runtime_dir / "outbox"

    @property
    def policy_dir(self) -> Path:
        return self.data_dir / "policies"

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
        return cls(
            provider=provider,
            model=os.getenv("LLM_MODEL", "").strip(),
            runtime_dir=Path(os.getenv("AGENT_RUNTIME_DIR", ROOT / "runtime")),
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "12")),
            max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "4096")),
            dry_run=os.getenv("AGENT_DRY_RUN", "").lower() in {"1", "true", "yes"},
            today=os.getenv("AGENT_TODAY", "").strip(),
        )
