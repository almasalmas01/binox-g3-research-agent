from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from .agent import ResearchAgent


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    agent = ResearchAgent(project_root)
    agent.memory.path.write_text("", encoding="utf-8")
    examples = json.loads((project_root / "examples" / "questions.json").read_text(encoding="utf-8"))

    for idx, example in enumerate(examples["questions"], start=1):
        result = agent.answer(example)
        print(f"=== Example {idx} ===")
        print(f"Question: {example}")
        print(f"Subquestions: {result.subquestions}")
        print(f"Retrieved chunks: {len(result.retrieved)}")
        print(f"Context tokens used: {result.context_tokens_used}")
        print(result.answer)
        print()


if __name__ == "__main__":
    main()
