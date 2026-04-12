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
    parser.add_argument("--interactive", "-i", action="store_true", help="Start an interactive multi-turn session.")
    return parser


def _budget_bar(used: int, total: int, width: int = 20) -> str:
    filled = round(used / total * width) if total > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _confidence_label(score: int) -> str:
    if score >= 75:
        return "HIGH"
    if score >= 45:
        return "MEDIUM"
    return "LOW"


def _confidence_bar(score: int, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]

    if args.examples:
        payload = json.loads((project_root / "examples" / "questions.json").read_text(encoding="utf-8"))
        print(json.dumps(payload, indent=2))
        return

    if args.interactive:
        agent = ResearchAgent(project_root)
        if args.reset_memory:
            agent.memory.path.write_text("", encoding="utf-8")
        print("Interactive research session. Type 'quit' or 'exit' to stop, '--reset' to clear memory.\n")
        while True:
            try:
                question = input("Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in ("quit", "exit"):
                break
            if question == "--reset":
                agent.memory.path.write_text("", encoding="utf-8")
                print("[Memory cleared]\n")
                continue
            if not question:
                continue
            _run_and_print(agent, question)
            print()
        return

    if not args.question:
        parser.error("Provide a research question, use --examples, or use --interactive.")

    agent = ResearchAgent(project_root)
    if args.reset_memory:
        agent.memory.path.write_text("", encoding="utf-8")
    _run_and_print(agent, args.question)


def _run_and_print(agent: ResearchAgent, question: str) -> None:
    print()
    result = agent.answer(question)

    total = agent.config.max_context_tokens
    unused = total - result.memory_tokens_used - result.context_tokens_used
    query_label = {"new_topic": "New topic", "default": "Default", "follow_up": "Follow-up"}.get(result.query_type, result.query_type)
    conf_label = _confidence_label(result.confidence_score)

    print("\nSUBQUESTIONS")
    for item in result.subquestions:
        print(f"  - {item}")

    print(f"\nSOURCES ({len(result.retrieved)} retrieved)")
    seen_urls: set[str] = set()
    for item in result.retrieved:
        url = item.chunk.source
        if url not in seen_urls:
            seen_urls.add(url)
            print(f"  [{item.score:.2f}] {item.chunk.title[:55]}  [{url}]")

    if result.evidence_gaps:
        print(f"\nEVIDENCE GAPS ({len(result.evidence_gaps)} sub-question{'s' if len(result.evidence_gaps) != 1 else ''} with no results)")
        for gap in result.evidence_gaps:
            print(f"  ! {gap}")

    print(f"\n── CONFIDENCE ── {conf_label} ──")
    print(f"  Score    [{_confidence_bar(result.confidence_score)}]  {result.confidence_score:>3} / 100")

    print(f"\n── TOKEN BUDGET ({total}) ── {query_label} ──")
    print(f"  Memory   [{_budget_bar(result.memory_tokens_used, total)}]  {result.memory_tokens_used:>4} / {total}  ({result.memory_tokens_used / total * 100:4.1f}%)")
    print(f"  Context  [{_budget_bar(result.context_tokens_used, total)}]  {result.context_tokens_used:>4} / {total}  ({result.context_tokens_used / total * 100:4.1f}%)")
    print(f"  Unused   [{_budget_bar(unused, total)}]  {unused:>4} / {total}  ({unused / total * 100:4.1f}%)")

    print("\nANSWER")
    print(result.answer)


if __name__ == "__main__":
    main()
