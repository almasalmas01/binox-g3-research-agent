from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from .agent import ResearchAgent
from .models import AgentConfig


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    config = AgentConfig()

    # Use a temporary memory file so evaluation never touches real session history
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = ResearchAgent(Path(tmpdir), config=config)
        # Point memory back to a file in the tmp dir (already done by the temp project_root)
        examples_path = project_root / "examples" / "questions.json"
        examples = json.loads(examples_path.read_text(encoding="utf-8"))
        questions = examples.get("questions", [])

        if not questions:
            print("No questions found in examples/questions.json.")
            return

        total_tokens = 0
        total_sources = 0
        total_time = 0.0
        failures = 0

        for idx, example in enumerate(questions, start=1):
            print(f"\n{'='*60}")
            print(f"Q{idx}: {example}")
            print("=" * 60)

            start = time.time()
            try:
                result = agent.answer(example)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                failures += 1
                continue
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

            budget_pct = result.context_tokens_used / config.max_context_tokens * 100
            print(f"\nTokens used: {result.context_tokens_used} / {config.max_context_tokens}  ({budget_pct:.0f}% of budget)")
            print(f"Time: {elapsed:.1f}s")
            print(f"\nAnswer:\n{result.answer}")

        answered = len(questions) - failures
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"  Questions:      {len(questions)}")
        print(f"  Answered:       {answered}")
        print(f"  Failures:       {failures}")
        if answered > 0:
            print(f"  Total sources:  {total_sources}")
            print(f"  Avg tokens:     {total_tokens // answered} / {config.max_context_tokens}")
            print(f"  Total time:     {total_time:.1f}s")
        print("=" * 60)


if __name__ == "__main__":
    main()
