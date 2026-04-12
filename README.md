# Binox G3: Deep Research Agent Under Memory Constraints

A research agent that answers any complex question using live web search, while operating under explicit token budget constraints. Built for the Binox G3 take-home assessment.

## How It Works

```
User question
     │
     ▼
 split_question()          ← breaks into sub-questions (regex, no LLM)
     │
     ▼
 search_web() × N          ← Tavily live web search per sub-question
     │
     ▼
 compress_with_budget()    ← 1,800 token cap enforced here
     │
     ▼
 load_recent_memory()      ← 600 token cap from past sessions
     │
     ▼
 Gemini 2.5 Flash          ← single LLM call for synthesis
     │
     ▼
 save to memory.jsonl      ← persists Q&A for future sessions
     │
     ▼
 Answer printed
```

The key design choice: **one LLM call per query**. Everything else — decomposition, retrieval, compression, budget enforcement — is deterministic Python. This keeps costs predictable and the system fully inspectable.

## Memory Architecture

Two separate token budgets run in parallel:

| Layer | What it holds | Token cap |
|---|---|---|
| Working context | Compressed web search results | 1,200 tokens (1,800 − 600) |
| Episodic memory | Past Q&A pairs from `session_memory.jsonl` | 600 tokens |

The episodic budget is carved out first, so past sessions can never crowd out fresh evidence. Raw web results are never passed to the LLM — each result is compressed to its single most relevant sentence before being packed into context.

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
git clone <repo-url>
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

## Usage

```bash
# Ask any research question
python -m research_agent.cli "What are the risks of AI regulation in the EU?"

# Multi-part question
python -m research_agent.cli "Compare the EV markets in Indonesia, Thailand, and Vietnam. Which should an EV charging startup enter first?"

# Clear episodic memory before a fresh session
python -m research_agent.cli --reset-memory "What is the current state of quantum computing?"

# See example questions
python -m research_agent.cli --examples

# Run evaluation across all example questions
python -m research_agent.evaluate
```

## Example Output

```
SUBQUESTIONS
- EV market Indonesia
- EV market Thailand
- EV market Vietnam

CONTEXT TOKENS USED
412

ANSWER
Indonesia presents the strongest near-term opportunity for an EV charging startup...
[Gemini-synthesized answer with source citations and gap analysis]
```

The CLI always prints the sub-questions and token count, making the constraint system visible.

## Constraints

Defined in `src/research_agent/models.py` (`AgentConfig`):

```python
max_context_tokens: int = 1800       # total context budget per query
max_memory_tokens: int = 600         # reserved for episodic memory
max_chunk_summary_tokens: int = 120  # per-result compression target
top_k_per_subquestion: int = 3       # web results fetched per sub-question
```

These can be overridden by passing a custom `AgentConfig` to `ResearchAgent`.

## Running Tests

```bash
python -m unittest discover -s tests
```

## Trade-offs

See [`evaluation.md`](evaluation.md) for a full discussion of architecture choices and known limitations.
