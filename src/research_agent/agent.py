from __future__ import annotations

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
        subquestions = split_question(question)
        memory_used = self.memory.load_recent(self.config.max_memory_tokens)

        retrieved = self._search_unique(subquestions)
        compressed_context, context_tokens_used = self._compress_with_budget(retrieved, memory_used)
        answer = self._synthesize(question, subquestions, compressed_context, memory_used)

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

    def _search_unique(self, subquestions: list[str]) -> list[RetrievedChunk]:
        seen_ids: set[str] = set()
        merged: list[RetrievedChunk] = []
        for subquestion in subquestions:
            try:
                results = search_web(subquestion, self.config.top_k_per_subquestion)
            except Exception as exc:
                print(f"[search] warning: search failed for '{subquestion}': {exc}")
                results = []
            for chunk in results:
                if chunk.chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk.chunk_id)
                merged.append(RetrievedChunk(chunk=chunk, score=1.0))
        return merged

    def _compress_with_budget(self, retrieved: list[RetrievedChunk], memory_used: list[str]) -> tuple[list[str], int]:
        budget = self.config.max_context_tokens - sum(estimate_tokens(item) for item in memory_used)
        compressed: list[str] = []
        consumed = 0

        for result in retrieved:
            summary = summarize_chunk(result, max_words=self.config.max_chunk_summary_tokens // 2)
            size = estimate_tokens(summary)
            if consumed + size > budget:
                break
            compressed.append(summary)
            consumed += size
        return compressed, consumed

    def _synthesize(
        self,
        question: str,
        subquestions: list[str],
        compressed_context: list[str],
        memory_used: list[str],
    ) -> str:
        context_block = "\n".join(f"- {item}" for item in compressed_context) or "- No supporting context was retrieved."
        memory_block = "\n".join(f"- {item}" for item in memory_used) or "- No past memory loaded."

        prompt = f"""You are a concise research assistant operating under a strict token budget.

Sub-questions identified:
{chr(10).join(f"- {q}" for q in subquestions)}

Evidence retrieved (compressed to fit within {self.config.max_context_tokens} tokens):
{context_block}

Past session memory:
{memory_block}

Answer the following question in 3-5 paragraphs. Be direct, cite the sources where relevant, and flag any gaps in the evidence.

Question: {question}"""

        try:
            response = self.client.models.generate_content(
                model="models/gemini-2.5-flash",
                contents=prompt,
            )
            return response.text or "No answer generated."
        except Exception as exc:
            return f"Synthesis failed: {exc}"
