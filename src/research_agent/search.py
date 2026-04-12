from __future__ import annotations

import hashlib
import os

from tavily import TavilyClient

from .budget import estimate_tokens
from .models import Chunk

_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _client


def search_web(query: str, top_k: int = 3) -> list[Chunk]:
    client = _get_client()
    response = client.search(query=query, max_results=top_k, search_depth="basic")

    chunks: list[Chunk] = []
    for idx, result in enumerate(response.get("results", [])):
        content = result.get("content") or result.get("snippet") or ""
        if not content:
            continue
        url = result["url"]
        stable_id = hashlib.md5(url.encode()).hexdigest()[:12]
        chunks.append(
            Chunk(
                chunk_id=f"web-{stable_id}-{idx}",
                doc_id=url,
                title=result.get("title") or url,
                source=url,
                published_at=result.get("published_date") or "unknown",
                text=content,
                token_estimate=estimate_tokens(content),
            )
        )
    return chunks
