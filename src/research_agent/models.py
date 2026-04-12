from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    published_at: str
    text: str
    token_estimate: int


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float
    matched_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AgentConfig:
    max_context_tokens: int = 1800
    max_memory_tokens: int = 600
    max_chunk_summary_tokens: int = 120
    top_k_per_subquestion: int = 3
    model: str = "models/gemini-2.5-flash"


@dataclass(slots=True)
class QueryResult:
    question: str
    subquestions: list[str]
    retrieved: list[RetrievedChunk]
    memory_used: list[str]
    context_tokens_used: int
    answer: str

