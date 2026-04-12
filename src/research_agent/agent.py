from __future__ import annotations

import json
import os
import re
from pathlib import Path

from google import genai

from .budget import estimate_tokens
from .memory_store import MemoryStore
from .models import AgentConfig, QueryResult, RetrievedChunk
from .planner import extract_terms, split_question
from .search import search_web
from .summarizer import summarize_chunk

# Memory token budget per novelty tier.
# Context gets whatever remains from max_context_tokens after memory is subtracted.
_MEMORY_BUDGET: dict[str, int] = {
    "new_topic": 200,   # fresh topic — maximise fresh evidence
    "default":   600,   # some overlap — balanced split
    "follow_up": 900,   # continuing a thread — weight prior context heavily
}


class ResearchAgent:
    def __init__(self, project_root: Path, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.memory = MemoryStore(project_root / "memory" / "session_memory.jsonl")
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def answer(self, question: str) -> QueryResult:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        subquestions = self._decompose_question(question)

        # Classify the question to determine budget split
        raw_memory = self.memory.load_recent(self.config.max_memory_tokens)
        query_type = _classify_question(question, raw_memory)
        memory_budget = _MEMORY_BUDGET[query_type]

        # Re-load memory with the dynamic budget
        memory_used = self.memory.load_recent(memory_budget)
        memory_tokens_used = sum(estimate_tokens(m) for m in memory_used)

        retrieved, evidence_gaps = self._search_unique(subquestions)
        if not retrieved:
            print("[agent] warning: no results retrieved for any sub-question.")

        compressed_context, context_tokens_used, context_sources = self._compress_with_budget(
            retrieved, memory_tokens_used
        )
        answer = self._synthesize(question, subquestions, compressed_context, memory_used, context_sources, query_type)
        confidence_score = _compute_confidence(subquestions, retrieved, evidence_gaps)

        first_line = _strip_markdown(answer.splitlines()[0]) if answer.strip() else question
        self.memory.append(question, first_line)

        return QueryResult(
            question=question,
            subquestions=subquestions,
            retrieved=retrieved,
            memory_used=memory_used,
            memory_tokens_used=memory_tokens_used,
            context_tokens_used=context_tokens_used,
            query_type=query_type,
            answer=answer,
            confidence_score=confidence_score,
            evidence_gaps=evidence_gaps,
        )

    def _decompose_question(self, question: str) -> list[str]:
        prompt = f"""Break the following research question into 2-4 focused sub-questions suitable for web search.
Each sub-question should be a complete, standalone search query.
Return ONLY a JSON array of strings, nothing else.

Example input: "What are the risks of EVs in Asia and how does Thailand compare to Vietnam?"
Example output: ["EV risks Southeast Asia", "Thailand EV market overview", "Vietnam EV market overview"]

Question: {question}"""

        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=prompt,
            )
            text = (response.text or "").strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            subquestions = json.loads(text.strip())
            if isinstance(subquestions, list) and subquestions and all(isinstance(q, str) for q in subquestions):
                return subquestions[:4]
        except Exception as exc:
            print(f"[planner] LLM decomposition failed, falling back to regex: {exc}")

        return split_question(question)

    def _search_unique(self, subquestions: list[str]) -> tuple[list[RetrievedChunk], list[str]]:
        best: dict[str, RetrievedChunk] = {}
        evidence_gaps: list[str] = []
        total = len(subquestions)
        for idx, subquestion in enumerate(subquestions, start=1):
            print(f"  [{idx}/{total}] Searching: \"{subquestion}\"...", end=" ", flush=True)
            try:
                results = search_web(subquestion, self.config.top_k_per_subquestion)
            except Exception as exc:
                print(f"error ({exc})")
                results = []
            if not results:
                print("no results")
                evidence_gaps.append(subquestion)
            else:
                print(f"{len(results)} source{'s' if len(results) != 1 else ''}")
                for result in results:
                    chunk_id = result.chunk.chunk_id
                    if chunk_id not in best or result.score > best[chunk_id].score:
                        best[chunk_id] = result
        return sorted(best.values(), key=lambda r: r.score, reverse=True), evidence_gaps

    def _compress_with_budget(
        self, retrieved: list[RetrievedChunk], memory_tokens_used: int
    ) -> tuple[list[str], int, list[RetrievedChunk]]:
        budget = max(0, self.config.max_context_tokens - memory_tokens_used)
        compressed: list[str] = []
        sources: list[RetrievedChunk] = []
        consumed = 0

        for result in retrieved:
            summary = summarize_chunk(result, max_words=self.config.max_chunk_summary_tokens // 2)
            size = estimate_tokens(summary)
            if consumed + size > budget:
                break
            compressed.append(summary)
            sources.append(result)
            consumed += size
        return compressed, consumed, sources

    def _synthesize(
        self,
        question: str,
        subquestions: list[str],
        compressed_context: list[str],
        memory_used: list[str],
        context_sources: list[RetrievedChunk],
        query_type: str = "default",
    ) -> str:
        context_block = "\n".join(f"- {item}" for item in compressed_context) or "- No supporting context was retrieved."
        memory_block = "\n".join(f"- {item}" for item in memory_used) if memory_used else None

        seen: set[str] = set()
        sources: list[tuple[int, str, str]] = []
        for result in context_sources:
            url = result.chunk.source
            if url not in seen:
                seen.add(url)
                sources.append((len(sources) + 1, result.chunk.title, url))

        source_list = "\n".join(f"[{n}] {title} — {url}" for n, title, url in sources)
        memory_section = f"\nPast session memory:\n{memory_block}\n" if memory_block else ""

        query_type_instructions = {
            "follow_up": "This is a FOLLOW-UP question — the user has asked about this topic before. "
                         "Weight information from past session memory heavily and build on prior context rather than repeating it.",
            "new_topic": "This is a NEW TOPIC the user has not asked about before. "
                         "Focus entirely on the retrieved evidence; do not speculate about prior context.",
            "default":   "This question has some overlap with past sessions. "
                         "Blend retrieved evidence with any relevant prior context.",
        }
        query_instruction = query_type_instructions.get(query_type, "")

        prompt = f"""You are a concise research assistant operating under a strict token budget.

{query_instruction}

Sub-questions identified:
{chr(10).join(f"- {q}" for q in subquestions)}

Numbered sources available for citation:
{source_list or "- None"}

Evidence retrieved (compressed to fit within {self.config.max_context_tokens} tokens):
{context_block}
{memory_section}
Answer the following question in 3-5 paragraphs. Cite sources using [N] notation. Flag any gaps in the evidence.

Question: {question}"""

        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=prompt,
            )
            answer = response.text or "No answer generated."
        except Exception as exc:
            return f"Synthesis failed: {exc}"

        if sources:
            reference_block = "\n\n---\n**Sources**\n" + "\n".join(
                f"[{n}] [{title}]({url})" for n, title, url in sources
            )
            answer = answer + reference_block
        return answer


def _classify_question(question: str, memory_entries: list[str]) -> str:
    """
    Classify the question as 'new_topic', 'default', or 'follow_up'
    by measuring term overlap between the question and recent memory entries.
    This drives dynamic budget allocation.
    """
    if not memory_entries:
        return "new_topic"

    question_terms = set(extract_terms(question))
    if not question_terms:
        return "default"

    best_overlap = 0.0
    for entry in memory_entries:
        entry_terms = set(extract_terms(entry))
        if not entry_terms:
            continue
        overlap = len(question_terms & entry_terms) / len(question_terms)
        if overlap > best_overlap:
            best_overlap = overlap

    if best_overlap >= 0.35:
        return "follow_up"
    if best_overlap <= 0.05:
        return "new_topic"
    return "default"


def _compute_confidence(
    subquestions: list[str],
    retrieved: list[RetrievedChunk],
    evidence_gaps: list[str],
) -> int:
    """
    Compute a 0–100 confidence score based on three factors:
    - Coverage: fraction of sub-questions that returned at least one result (50%)
    - Relevance: mean Tavily relevance score of retrieved chunks (35%)
    - Depth: how full the result set is relative to the maximum possible (15%)
    """
    if not subquestions:
        return 0

    coverage = 1.0 - len(evidence_gaps) / len(subquestions)

    if retrieved:
        avg_score = sum(r.score for r in retrieved) / len(retrieved)
        # Tavily scores are typically 0.0–1.0; clamp defensively
        avg_score = min(max(avg_score, 0.0), 1.0)
    else:
        avg_score = 0.0

    # How many unique sources did we get relative to what we asked for?
    max_possible = len(subquestions) * 3  # top_k_per_subquestion default
    depth = min(len(retrieved) / max_possible, 1.0) if max_possible > 0 else 0.0

    raw = coverage * 0.50 + avg_score * 0.35 + depth * 0.15
    return round(raw * 100)


def _strip_markdown(text: str) -> str:
    """Remove common markdown tokens so memory summaries are plain text."""
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    return text.strip()
