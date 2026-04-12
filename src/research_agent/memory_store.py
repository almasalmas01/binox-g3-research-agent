from __future__ import annotations

import json
from pathlib import Path

from .budget import estimate_tokens


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, question: str, summary: str) -> None:
        entry = {"question": question, "summary": summary}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    def load_recent(self, max_tokens: int) -> list[str]:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        selected: list[str] = []
        consumed = 0
        for line in reversed(lines):
            if not line.strip():
                continue
            payload = json.loads(line)
            candidate = f"Past session: {payload['question']} -> {payload['summary']}"
            size = estimate_tokens(candidate)
            if consumed + size > max_tokens:
                break
            selected.append(candidate)
            consumed += size
        selected.reverse()
        return selected

