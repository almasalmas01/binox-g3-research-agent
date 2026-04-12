from __future__ import annotations

import json
import os
from pathlib import Path

from google import genai

from .budget import estimate_tokens
from .memory_store import MemoryStore
from .models import AgentConfig, QueryResult, RetrievedChunk
from .planner import split_question
from .search import search_web
from .summarizer import summarize_chunk


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
        memory_used = self.memory.load_recent(self.config.max_memory_tokens)

        retrieved = self._search_unique(subquestions)
        compressed_context, context_tokens_used, context_sources = self._compress_with_budget(retrieved, memory_used)
        answer = self._synthesize(question, subquestions, compressed_context, memory_used, context_sources)

        first_line = answer.splitlines()[0] if answer.strip() else question
        self.memory.append(question, first_line)

        return QueryResult(
            question=question,
            subquestions=subquestions,
            retrieved=retrieved,
            memory_used=memory_used,
            context_tokens_used=context_tokens_used,
            answer=answer,
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

    def _search_unique(self, subquestions: list[str]) -> list[RetrievedChunk]:
        best: dict[str, RetrievedChunk] = {}
        for subquestion in subquestions:
            try:
                results = search_web(subquestion, self.config.top_k_per_subquestion)
            except Exception as exc:
                print(f"[search] warning: search failed for '{subquestion}': {exc}")
                results = []
            for result in results:
                chunk_id = result.chunk.chunk_id
                if chunk_id not in best or result.score > best[chunk_id].score:
                    best[chunk_id] = result
        return sorted(best.values(), key=lambda r: r.score, reverse=True)

    def _compress_with_budget(
        self, retrieved: list[RetrievedChunk], memory_used: list[str]
    ) -> tuple[list[str], int, list[RetrievedChunk]]:
        memory_cost = sum(estimate_tokens(item) for item in memory_used)
        budget = max(0, self.config.max_context_tokens - memory_cost)
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
    ) -> str:
        context_block = "\n".join(f"- {item}" for item in compressed_context) or "- No supporting context was retrieved."
        memory_block = "\n".join(f"- {item}" for item in memory_used) or "- No past memory loaded."

        # Only number sources that actually appear in the compressed context
        seen: set[str] = set()
        sources: list[tuple[int, str, str]] = []
        for result in context_sources:
            url = result.chunk.source
            if url not in seen:
                seen.add(url)
                sources.append((len(sources) + 1, result.chunk.title, url))

        source_list = "\n".join(f"[{n}] {title} — {url}" for n, title, url in sources)

        prompt = f"""You are a concise research assistant operating under a strict token budget.

Sub-questions identified:
{chr(10).join(f"- {q}" for q in subquestions)}

Numbered sources available for citation:
{source_list or "- None"}

Evidence retrieved (compressed to fit within {self.config.max_context_tokens} tokens):
{context_block}

Past session memory:
{memory_block}

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
