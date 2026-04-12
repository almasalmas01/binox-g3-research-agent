from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from .agent import ResearchAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Binox G3 research agent.")
    parser.add_argument("question", nargs="?", help="Research question to answer.")
    parser.add_argument("--examples", action="store_true", help="Print bundled example questions.")
    parser.add_argument("--reset-memory", action="store_true", help="Clear episodic memory before answering.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]

    if args.examples:
        payload = json.loads((project_root / "examples" / "questions.json").read_text(encoding="utf-8"))
        print(json.dumps(payload, indent=2))
        return

    if not args.question:
        parser.error("Provide a research question or use --examples.")

    agent = ResearchAgent(project_root)
    if args.reset_memory:
        agent.memory.path.write_text("", encoding="utf-8")
    result = agent.answer(args.question)

    print("SUBQUESTIONS")
    for item in result.subquestions:
        print(f"  - {item}")
    print(f"\nSOURCES ({len(result.retrieved)} retrieved)")
    seen_urls: set[str] = set()
    for item in result.retrieved:
        url = item.chunk.source
        if url not in seen_urls:
            seen_urls.add(url)
            print(f"  - {item.chunk.title[:60]}  [{url}]")
    print(f"\nCONTEXT TOKENS USED: {result.context_tokens_used} / {agent.config.max_context_tokens}")
    print("\nANSWER")
    print(result.answer)


if __name__ == "__main__":
    main()
