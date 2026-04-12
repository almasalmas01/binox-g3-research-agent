from __future__ import annotations

import re


CONNECTORS = re.compile(r"\b(and|also|plus|compare|versus|vs\.?|with)\b", re.IGNORECASE)
COMPARE_PATTERN = re.compile(
    r"compare\s+(?P<items>.+?)\s+for\s+(?P<topic>.+?)(?:[?]|$)",
    re.IGNORECASE,
)


def split_question(question: str) -> list[str]:
    normalized = " ".join(question.strip().split())
    compare_match = COMPARE_PATTERN.search(normalized)
    if compare_match:
        items = split_compare_items(compare_match.group("items"))
        topic = compare_match.group("topic").strip(" .?")
        subquestions = [f"{item} for {topic}" for item in items if len(item.split()) <= 4]
        if subquestions:
            return dedupe_preserve_order(subquestions[:4])

    parts = [part.strip(" ,.") for part in CONNECTORS.split(normalized) if part.strip(" ,.").lower() not in {"and", "also", "plus", "compare", "versus", "vs", "with"}]

    subquestions: list[str] = []
    if len(parts) > 1:
        for part in parts:
            if len(part.split()) >= 3:
                subquestions.append(part)

    if not subquestions:
        clauses = [piece.strip(" ,.") for piece in re.split(r"[?;]", normalized) if piece.strip(" ,.")]
        subquestions = clauses or [normalized]

    return dedupe_preserve_order(subquestions[:4])


def extract_terms(text: str) -> list[str]:
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "for", "to", "of",
        "in", "on", "by", "with", "what", "which", "how", "should", "would",
        "could", "from", "that", "this", "into", "about", "after", "before",
        "than", "then", "them", "they", "their", "your", "our", "who", "why",
        "can", "do", "does", "did", "at", "as", "it", "its",
    }
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", text.lower())
    return [token for token in tokens if token not in stop_words and len(token) > 2]


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def split_compare_items(text: str) -> list[str]:
    normalized = text.replace(", and ", ", ").replace(" and ", ", ")
    return [item.strip(" ,.") for item in normalized.split(",") if item.strip(" ,.")]
