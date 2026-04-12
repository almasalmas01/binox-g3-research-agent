# Evaluation And Trade-Offs

## Problem Framing

The G3 prompt asks for a research agent that answers complex queries under memory constraints. I treated this as a systems design problem: the interesting challenge is not which LLM to call, but how to control what goes into that call and why.

The prototype optimizes for:
- explicit, inspectable resource limits
- a single, predictable LLM call per query
- modular components that can be upgraded independently
- reproducible local execution with no hidden state

## Architecture Overview

```
User question
     │
     ▼
 split_question()          ← deterministic regex decomposition
     │
     ▼
 search_web() × N          ← Tavily live web search per sub-question
     │
     ▼
 compress_with_budget()    ← 1,800 token cap enforced before LLM
     │
     ▼
 load_recent_memory()      ← 600 token cap from past sessions
     │
     ▼
 Gemini 2.5 Flash          ← single LLM call for synthesis
     │
     ▼
 save to memory.jsonl
```

## Memory Strategy

Two bounded layers run in parallel:

**Working context** — web search results, compressed to fit within `max_context_tokens = 1800`.
Each result is reduced to its single most query-relevant sentence before packing. Results are added one at a time until the next one would exceed the budget, then the loop stops.

**Episodic memory** — past Q&A pairs stored in `memory/session_memory.jsonl`.
On each run the agent loads the most recent entries that fit within `max_memory_tokens = 600`, loaded newest-first. The episodic budget is carved out first, so past sessions can never crowd out fresh evidence from the current query.

## Why One LLM Call Per Query

Most agent frameworks make multiple LLM calls per query: one to plan, one to retrieve, one to summarize, one to synthesize. Each call adds latency, cost, and a new failure point.

This agent uses a single LLM call — only for synthesis. Everything before it (decomposition, retrieval, compression, budget enforcement) is deterministic Python. This makes the system:
- cheaper to run
- easier to debug (each stage is independently inspectable)
- more predictable in cost per query

The trade-off is that the decomposition and compression are weaker than they would be with LLM assistance. That is an acceptable trade-off at this stage.

## Why Compression Over Raw Retrieval

With a small token budget, the agent can either:
- pass a few raw documents with high fidelity
- pass more compressed evidence with lower fidelity

Compression was chosen because broader coverage is more useful than perfect preservation of a single source for most research questions. The current compressor picks the highest-scoring sentence per chunk using term-overlap ranking. This is weak for nuanced documents but fast and zero-cost.

## Constraints (Self-Defined)

| Parameter | Value | Reason |
|---|---|---|
| `max_context_tokens` | 1,800 | Forces selective evidence use |
| `max_memory_tokens` | 600 | Keeps episodic context from dominating |
| `max_chunk_summary_tokens` | 120 | Limits per-result verbosity |
| `top_k_per_subquestion` | 3 | Bounds Tavily API usage |

## Known Limitations

| Limitation | Impact | Upgrade Path |
|---|---|---|
| Regex question decomposition | Misses complex or ambiguous phrasings | Replace with an LLM planner call |
| Single-sentence compression | Loses nuance in long documents | Use an extractive summarizer or small LLM |
| No embedding retrieval | Keyword-free queries return weak results | Add offline embedding cache with cosine search |
| No citation links in answer | Gemini names sources but doesn't hyperlink them | Parse `result.chunk.source` and inject into prompt |
| Episodic memory is append-only | Grows unbounded over many sessions | Add TTL or relevance-based pruning |

## What This Demonstrates

The submission shows:
- system decomposition under a real constraint
- cost-aware architecture (one API call per query, bounded search results)
- two-layer memory design with independent budgets
- evaluation thinking (this document)
- awareness of failure modes and upgrade paths

That is closer to real agent engineering work than wrapping a single prompt in an API call.

## Self-Assessment

Strong on: architecture clarity, constraint enforcement, reproducibility, honest trade-off documentation.

Weak on: answer fluency for highly specific queries, graceful handling of decomposition failures, citation quality.

The weakest single component is the sentence-level compressor — it works but it is naive. Replacing it with a small summarization model would have the highest ROI of any single upgrade.
