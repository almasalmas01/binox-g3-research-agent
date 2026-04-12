from __future__ import annotations

import re

from .budget import clamp_words
from .models import RetrievedChunk
from .planner import extract_terms


def summarize_chunk(result: RetrievedChunk, max_words: int = 45) -> str:
    text = result.chunk.text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    query_terms = set(result.matched_terms or extract_terms(text))

    # Score all sentences, keep their original index for re-ordering
    scored = sorted(
        enumerate(sentences),
        key=lambda pair: sentence_score(pair[1], query_terms),
        reverse=True,
    )

    # Greedily pick sentences by score until the word budget runs out
    header = f"{result.chunk.title} ({result.chunk.published_at}): "
    words_remaining = max_words - len(header.split())
    selected: list[tuple[int, str]] = []
    for idx, sentence in scored:
        sentence = sentence.strip()
        if not sentence:
            continue
        word_count = len(sentence.split())
        if word_count > words_remaining:
            break
        selected.append((idx, sentence))
        words_remaining -= word_count
        if words_remaining <= 0:
            break

    if not selected:
        # Fallback: just take the highest-scoring sentence and clamp it
        best = scored[0][1].strip() if scored else text
        return clamp_words(f"{header}{best}", max_words)

    # Re-order by original position so summary reads naturally
    selected.sort(key=lambda pair: pair[0])
    body = " ".join(s for _, s in selected)
    return clamp_words(f"{header}{body}", max_words)


def sentence_score(sentence: str, query_terms: set[str]) -> tuple[int, int]:
    lowered = sentence.lower()
    matches = sum(1 for term in query_terms if term in lowered)
    # tiebreak: prefer shorter sentences (more likely to be direct statements)
    return matches, -len(sentence)
