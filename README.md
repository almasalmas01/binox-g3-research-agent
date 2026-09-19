# Binox G3: Deep Research Agent Under Memory Constraints

A research agent that answers any complex question using live web search, while operating under explicit token budget constraints. Built for the Binox G3 take-home assessment.

## Tech Talk

[View the Tech Talk slides (PDF)](TechTalk_15.pdf).

## How It Works

```
User question
     │
     ▼
 Gemini (decompose)        ← LLM breaks question into 2-4 sub-questions
     │  (regex fallback on failure)
     ▼
 classify_question()       ← term overlap → new_topic / default / follow_up
     │
     ▼
 dynamic budget split      ← memory vs context allocation selected per query
     │
     ▼
 search_web() × N          ← Tavily live web search per sub-question
     │
     ▼
 compress_with_budget()    ← multi-sentence compression within token cap
     │
     ▼
 Gemini (synthesize)       ← answer with numbered [N] citations + source list
     │
     ▼
 save to memory.jsonl      ← persists Q&A, expires after 7 days
     │
     ▼
 Answer + budget bar printed
```

The agent uses two LLM calls per query: one fast decomposition call and one synthesis call. Everything between them — retrieval, deduplication, compression, budget enforcement — is deterministic Python.

## Memory Architecture

Two separate token budgets run in parallel, with the split chosen dynamically based on how novel the question is:

| Query type | Memory | Context | Triggered when |
|---|---|---|---|
| New topic | 200 tokens | 1,600 tokens | No prior sessions or < 5% term overlap |
| Default | 600 tokens | 1,200 tokens | 5–35% overlap with memory |
| Follow-up | 900 tokens | 900 tokens | ≥ 35% overlap — user is continuing a prior thread |

The episodic budget is always carved out first. Past sessions expire after 7 days (TTL). Raw web results are never passed to the LLM — each result is compressed to its highest-scoring sentences using query-term overlap before being packed into context.

## Project Structure

```
binox-g3-research-agent/
├── src/research_agent/
│   ├── agent.py          # Main orchestration loop
│   ├── planner.py        # Question decomposition
│   ├── search.py         # Tavily web search wrapper
│   ├── summarizer.py     # Per-chunk compression
│   ├── budget.py         # Token estimation and clamping
│   ├── memory_store.py   # JSONL episodic memory
│   ├── models.py         # Dataclasses (Chunk, QueryResult, AgentConfig)
│   └── cli.py            # Command-line interface
├── memory/               # Session memory written at runtime
├── examples/questions.json
├── tests/test_agent.py
├── evaluation.md
└── pyproject.toml
```

## Setup

**Requirements**: Python 3.11+

```bash
git clone https://github.com/almasalmas01/binox-g3-research-agent
cd binox-g3-research-agent

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .
```

Create a `.env` file in the project root:

```
TAVILY_API_KEY=your-tavily-key-here
GEMINI_API_KEY=your-gemini-key-here
```

- Tavily free tier: [tavily.com](https://tavily.com) — 1,000 searches/month
- Gemini API key: [aistudio.google.com](https://aistudio.google.com) — free tier available

After `pip install -e .`, the following commands are available directly in your shell:

| Command | Equivalent |
|---|---|
| `research-agent "..."` | `python -m research_agent.cli "..."` |
| `research-agent-eval` | `python -m research_agent.evaluate` |

## Usage

```bash
# Ask any research question
research-agent "What are the risks of AI regulation in the EU?"

# Multi-part comparison question
research-agent "Compare the EV markets in Indonesia, Thailand, and Vietnam. Which should an EV charging startup enter first?"

# Clear episodic memory before a fresh session
research-agent --reset-memory "What is the current state of quantum computing?"

# Start an interactive multi-turn session (memory persists between questions)
research-agent --interactive

# See example questions
research-agent --examples

# Run evaluation across all example questions
research-agent-eval
```

## Example Output

```
  [1/3] Searching: "EV market Indonesia overview"... 3 sources
  [2/3] Searching: "Thailand EV charging infrastructure"... 3 sources
  [3/3] Searching: "Vietnam EV adoption statistics"... no results

SUBQUESTIONS
  - EV market Indonesia overview
  - EV charging infrastructure Thailand
  - Vietnam EV adoption statistics

SOURCES (5 retrieved)
  [0.92] Indonesia EV Market Report 2025  [https://example.com/...]
  [0.87] Thailand EV charging station data  [https://example.com/...]

EVIDENCE GAPS (1 sub-question with no results)
  ! Vietnam EV adoption statistics

── CONFIDENCE ── MEDIUM ──
  Score    [██████████████░░░░░░]   68 / 100

── TOKEN BUDGET (1800) ── New topic ──
  Memory   [░░░░░░░░░░░░░░░░░░░░]     0 / 1800   (0.0%)
  Context  [████████████░░░░░░░░]  1080 / 1800  (60.0%)
  Unused   [████████░░░░░░░░░░░░]   720 / 1800  (40.0%)

ANSWER
Indonesia presents the strongest near-term opportunity for an EV charging
startup, driven by its large population and government EV incentive programs [1].
Thailand shows strong infrastructure investment but a more competitive market [2].
Note: evidence for Vietnam was limited — treat those conclusions with caution.

---
**Sources**
[1] [Indonesia EV Market Report 2025](https://example.com/...)
[2] [Thailand EV charging station data](https://example.com/...)
```

The output shows every stage of the agent's work: live search progress per sub-question, source relevance scores, evidence gaps where retrieval failed, a confidence score, and the token budget breakdown.

## Constraints

Defined in `src/research_agent/models.py` (`AgentConfig`):

```python
max_context_tokens: int = 1800       # total token budget per query
max_memory_tokens: int = 600         # default memory cap (overridden by dynamic allocation)
max_chunk_summary_tokens: int = 120  # per-result compression target
top_k_per_subquestion: int = 3       # web results fetched per sub-question
model: str = "models/gemini-2.5-flash"
```

These can be overridden by passing a custom `AgentConfig` to `ResearchAgent`.

## Running Tests

28 unit tests, no real API calls — all LLM and search calls are mocked.

```bash
python -m unittest discover -s tests
```

## Trade-offs

See [`evaluation.md`](evaluation.md) for a full discussion of architecture choices and known limitations.
