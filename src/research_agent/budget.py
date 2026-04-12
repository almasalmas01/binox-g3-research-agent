from __future__ import annotations

import re


WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    # Approximation: English prose averages roughly 0.75 words per token.
    units = WORD_RE.findall(text)
    return max(1, int(len(units) / 0.75))


def clamp_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:") + "..."

