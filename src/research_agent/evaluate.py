from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv

from .agent import ResearchAgent
from .models import AgentConfig


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    config = AgentConfig()
    agent = ResearchAgent(project_root, config=config)
    agent.memory.path.write_text("", encoding="utf-8")
    examples = json.loads((project_root / "examples" / "questions.json").read_text(encoding="utf-8"))

    total_tokens = 0
    total_sources = 0
    total_time = 0.0

    for idx, example in enumerate(examples["questions"], start=1):
        print(f"\n{'='*60}")
        print(f"Q{idx}: {example}")
        print("=" * 60)

        start = time.time()
        result = agent.answer(example)
        elapsed = time.time() - start

        total_tokens += result.context_tokens_used
        total_sources += len(result.retrieved)
        total_time += elapsed

        print(f"Sub-questions ({len(result.subquestions)}):")
        for q in result.subquestions:
            print(f"  - {q}")

        print(f"\nSources ({len(result.retrieved)}):")
        seen: set[str] = set()
        for item in result.retrieved:
            if item.chunk.source not in seen:
                seen.add(item.chunk.source)
                print(f"  [{item.score:.2f}] {item.chunk.title[:55]}  {item.chunk.source}")

        print(f"\nTokens used: {result.context_tokens_used} / {config.max_context_tokens}  "
              f"({result.context_tokens_used / config.max_context_tokens * 100:.0f}% of budget)")
        print(f"Time: {elapsed:.1f}s")
        print(f"\nAnswer:\n{result.answer}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  Questions:      {len(examples['questions'])}")
    print(f"  Total sources:  {total_sources}")
    print(f"  Avg tokens:     {total_tokens // len(examples['questions'])} / {config.max_context_tokens}")
    print(f"  Total time:     {total_time:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
