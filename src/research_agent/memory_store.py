from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .budget import estimate_tokens

MEMORY_TTL_DAYS = 7


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, question: str, summary: str) -> None:
        entry = {
            "question": question,
            "summary": summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    def load_recent(self, max_tokens: int) -> list[str]:
        now = datetime.now(timezone.utc)
        lines = self.path.read_text(encoding="utf-8").splitlines()
        selected: list[str] = []
        consumed = 0
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip corrupt entries
            # Drop entries older than TTL; use continue not break
            # so out-of-order entries don't cut off valid newer ones
            ts = payload.get("timestamp")
            if ts:
                try:
                    age_days = (now - datetime.fromisoformat(ts)).days
                    if age_days > MEMORY_TTL_DAYS:
                        continue
                except ValueError:
                    continue  # skip entries with unparseable timestamps
            candidate = f"Past session: {payload['question']} -> {payload['summary']}"
            size = estimate_tokens(candidate)
            if consumed + size > max_tokens:
                break
            selected.append(candidate)
            consumed += size
        selected.reverse()
        return selected
