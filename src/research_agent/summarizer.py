from __future__ import annotations

import re

from .budget import clamp_words
from .models import RetrievedChunk
from .planner import extract_terms


def summarize_chunk(result: RetrievedChunk, max_words: int = 45) -> str:
    text = result.chunk.text.strip()
    header = f"{result.chunk.title} ({result.chunk.published_at}): "

    if not text:
        return clamp_words(header.rstrip(": "), max_words)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return clamp_words(f"{header}{text}", max_words)

    query_terms = set(result.matched_terms or extract_terms(text))

    # Score all sentences, keep their original index for re-ordering
    scored = sorted(
        enumerate(sentences),
        key=lambda pair: sentence_score(pair[1], query_terms),
        reverse=True,
    )

    # Greedily pick sentences by score until the word budget runs out.
    # Use continue (not break) so shorter sentences later can still fit.
    words_remaining = max_words - len(header.split())
    selected: list[tuple[int, str]] = []
    for idx, sentence in scored:
        word_count = len(sentence.split())
        if word_count > words_remaining:
            continue  # skip sentences that are too long, try the next one
        selected.append((idx, sentence))
        words_remaining -= word_count
        if words_remaining <= 0:
            break

    if not selected:
        return clamp_words(f"{header}{scored[0][1]}", max_words)

    # Re-order by original position so summary reads naturally
    selected.sort(key=lambda pair: pair[0])
    body = " ".join(s for _, s in selected)
    return clamp_words(f"{header}{body}", max_words)


def sentence_score(sentence: str, query_terms: set[str]) -> tuple[int, int]:
    lowered = sentence.lower()
    matches = sum(1 for term in query_terms if term in lowered)
    # tiebreak: prefer shorter sentences (more likely to be direct statements)
    return matches, -len(sentence)
