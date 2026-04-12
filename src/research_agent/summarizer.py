from __future__ import annotations

import re

from .budget import clamp_words
from .models import RetrievedChunk
from .planner import extract_terms


def summarize_chunk(result: RetrievedChunk, max_words: int = 45) -> str:
    text = result.chunk.text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    query_terms = set(result.matched_terms or extract_terms(text))
    ranked = sorted(
        sentences,
        key=lambda sentence: sentence_score(sentence, query_terms),
        reverse=True,
    )
    best = ranked[0].strip() if ranked else text
    summary = f"{result.chunk.title} ({result.chunk.published_at}): {best}"
    return clamp_words(summary, max_words)


def sentence_score(sentence: str, query_terms: set[str]) -> tuple[int, int]:
    lowered = sentence.lower()
    matches = sum(1 for term in query_terms if term in lowered)
    # tiebreak: prefer shorter sentences (more likely to be direct statements)
    return matches, -len(sentence)

