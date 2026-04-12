# Evaluation And Trade-Offs

## Problem Framing

The G3 prompt asks for a research agent that answers complex queries under memory constraints. I treated this as a systems design problem: the interesting challenge is not which LLM to call, but how to control what goes into that call and why.

The prototype optimizes for:
- explicit, inspectable resource limits
- a single, predictable LLM call per query (not counting question decomposition)
- modular components that can be upgraded independently
- reproducible local execution with no hidden state

## Architecture Overview

```
User question
     │
     ▼
 _decompose_question()     ← LLM call (Gemini) → 2-4 focused sub-questions
     │  (regex fallback if LLM fails)
     ▼
 load_recent_memory()      ← preview load for classification
     │
     ▼
 _classify_question()      ← term overlap → "new_topic" / "default" / "follow_up"
     │
     ▼
 dynamic budget allocation ← budget split selected from 3-tier table
     │
     ▼
 search_web() × N          ← Tavily live web search per sub-question
     │
     ▼
 _search_unique()          ← deduplication by stable MD5 chunk ID
     │
     ▼
 _compress_with_budget()   ← multi-sentence compression within token cap
     │
     ▼
 Gemini 2.5 Flash          ← single synthesis call with numbered citations
     │
     ▼
 save to memory.jsonl      ← persists Q&A for future sessions (TTL = 7 days)
     │
     ▼
 Answer + budget bar printed
```

## Memory Strategy

Two bounded layers run with dynamically allocated budgets:

**Working context** — compressed web search results, fitted within the context token budget.
Each result is reduced to its highest-scoring sentences using query-term overlap scoring, not just the document's own terms. Sentences are selected greedily; if a sentence is too long for the remaining budget it is skipped (not cut off), so shorter sentences later in the document can still be included.

**Episodic memory** — past Q&A pairs stored in `memory/session_memory.jsonl`.
Entries carry ISO timestamps and expire after 7 days (TTL). On each run the agent loads the most recent entries that fit within the memory budget. The episodic budget is carved out first, so past sessions can never crowd out fresh evidence.

### Dynamic Budget Allocation

The agent classifies the incoming question against recent memory using term overlap, then selects one of three budget splits:

| Query type | Memory tokens | Context tokens | When triggered |
|---|---|---|---|
| `new_topic` | 200 | 1,600 | No memory, or overlap ≤ 5% |
| `default` | 600 | 1,200 | Overlap between 5% and 35% |
| `follow_up` | 900 | 900 | Overlap ≥ 35% |

For follow-up questions the synthesis prompt explicitly instructs the LLM to weight memory heavily and build on prior context rather than repeating it. For new topics the LLM is told to focus entirely on retrieved evidence.

### Budget Visualization

The CLI prints a live token budget bar after each query:

```
── TOKEN BUDGET (1800) ── Follow-up ──
  Memory   [████░░░░░░░░░░░░░░░░]   450 / 1800  (25.0%)
  Context  [████████░░░░░░░░░░░░]   720 / 1800  (40.0%)
  Unused   [████████░░░░░░░░░░░░]   630 / 1800  (35.0%)
```

This makes the constraint system visible at a glance rather than a hidden implementation detail.

## Why One LLM Call Per Query (For Synthesis)

Most agent frameworks make multiple LLM calls per query: one to plan, one to retrieve, one to summarize, one to synthesize. Each call adds latency, cost, and a new failure point.

This agent uses one LLM call for decomposition and one for synthesis. Everything else — retrieval, compression, budget enforcement, deduplication — is deterministic Python. This makes the system:
- cheaper to run
- easier to debug (each stage is independently inspectable)
- more predictable in cost per query

The decomposition call is fast (no search, small prompt) and falls back gracefully to regex if the LLM returns malformed JSON.

## Why Compression Over Raw Retrieval

With a small token budget, the agent can either:
- pass a few raw documents with high fidelity
- pass more compressed evidence with lower fidelity

Compression was chosen because broader coverage is more useful than perfect preservation of a single source for most research questions. The compressor scores sentences by overlap with the actual search query terms (not just the document's own vocabulary), ensuring relevance to what was actually searched rather than what the document is broadly about. A `continue`-based loop (not `break`) ensures short sentences later in a document are not discarded just because one long sentence couldn't fit.

## Deduplication

Tavily often returns the same URL for multiple sub-questions. Chunk IDs are derived from `hashlib.md5(url)` — stable regardless of position or order — so the same page is only included once. If the same URL appears across sub-questions, the copy with the higher relevance score wins.

## Source Citations

The synthesis prompt gives Gemini a numbered source list:

```
[1] Title — https://example.com
[2] Title — https://other.com
```

Gemini is instructed to cite inline using `[N]` notation. After synthesis a formatted reference block is appended to the answer. This keeps citations machine-readable and allows the user to verify claims against specific sources.

## Constraints (Self-Defined)

| Parameter | Value | Reason |
|---|---|---|
| `max_context_tokens` | 1,800 | Forces selective evidence use |
| `max_memory_tokens` | 600 | Default cap; overridden by dynamic allocation |
| `max_chunk_summary_tokens` | 120 | Limits per-result verbosity |
| `top_k_per_subquestion` | 3 | Bounds Tavily API usage |
| Memory TTL | 7 days | Prevents stale context from polluting future sessions |

## Known Limitations

| Limitation | Impact | Upgrade Path |
|---|---|---|
| Term-overlap classification | Misses semantic similarity (synonyms, paraphrases) | Replace with embedding cosine similarity |
| Extractive compression | Loses nuance in long or narrative documents | Use a small summarization model per chunk |
| No embedding retrieval | Keyword-free queries may return weak results | Add offline embedding cache with cosine search |
| Memory is append-only | Grows within TTL window | Add relevance-based pruning to complement TTL |
| Single Tavily search per sub-question | May miss niche sources | Add secondary search provider as fallback |

## What This Demonstrates

The submission shows:
- system decomposition under a real constraint
- cost-aware architecture (two API calls per query, bounded search results)
- adaptive memory strategy (dynamic budget allocation by novelty)
- two-layer memory design with independent budgets and TTL expiry
- transparent constraint visualization (budget bar in CLI)
- evaluation thinking (this document)
- awareness of failure modes and upgrade paths

## Self-Assessment

Strong on: architecture clarity, constraint enforcement, dynamic adaptation, reproducibility, honest trade-off documentation, citation transparency.

Weak on: answer fluency for highly specific queries, semantic classification (term overlap is a blunt instrument), compression quality for dense documents.

The highest-ROI single upgrade would be replacing term-overlap classification with sentence-embedding cosine similarity — this would make follow-up detection much more robust and unlock better budget allocation across paraphrased or synonym-heavy questions.
