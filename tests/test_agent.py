import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from research_agent import AgentConfig, ResearchAgent
from research_agent.budget import clamp_words, estimate_tokens
from research_agent.memory_store import MemoryStore
from research_agent.models import Chunk, RetrievedChunk
from research_agent.planner import split_question
from research_agent.summarizer import summarize_chunk


class PlannerTest(unittest.TestCase):
    def test_splits_compare_question(self) -> None:
        parts = split_question("Compare Indonesia, Thailand, and Vietnam for EV market entry.")
        self.assertGreater(len(parts), 1)
        self.assertLessEqual(len(parts), 4)

    def test_single_question_passthrough(self) -> None:
        parts = split_question("What is the capital of France?")
        self.assertEqual(len(parts), 1)

    def test_deduplicates(self) -> None:
        parts = split_question("What are risks and what are risks?")
        self.assertEqual(len(set(p.lower() for p in parts)), len(parts))


class BudgetTest(unittest.TestCase):
    def test_estimate_tokens_nonzero(self) -> None:
        self.assertGreater(estimate_tokens("hello world this is a test"), 0)

    def test_clamp_words_truncates(self) -> None:
        result = clamp_words("one two three four five", max_words=3)
        self.assertTrue(result.endswith("..."))
        self.assertIn("one two three", result)

    def test_clamp_words_passthrough(self) -> None:
        result = clamp_words("short text", max_words=10)
        self.assertEqual(result, "short text")


class MemoryStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        tmp_file = Path(self.tmp_dir.name) / "memory.jsonl"
        self.store = MemoryStore(tmp_file)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_append_and_load(self) -> None:
        self.store.append("What is X?", "X is a thing.")
        entries = self.store.load_recent(max_tokens=500)
        self.assertEqual(len(entries), 1)
        self.assertIn("What is X?", entries[0])

    def test_load_respects_token_budget(self) -> None:
        for i in range(20):
            self.store.append(f"Question {i} with some extra words to use tokens", f"Answer {i}")
        entries = self.store.load_recent(max_tokens=50)
        total = sum(estimate_tokens(e) for e in entries)
        self.assertLessEqual(total, 50)


class SummarizerTest(unittest.TestCase):
    def _make_result(self, text: str) -> RetrievedChunk:
        chunk = Chunk(
            chunk_id="c1", doc_id="d1", title="Test Title",
            source="http://example.com", published_at="2026-01-01",
            text=text, token_estimate=estimate_tokens(text),
        )
        return RetrievedChunk(chunk=chunk, score=1.0, matched_terms=["risk", "market"])

    def test_summary_within_word_limit(self) -> None:
        result = self._make_result(
            "The market is growing fast. Risks include regulation. Competition is fierce. "
            "Infrastructure is lacking. Demand is uncertain."
        )
        summary = summarize_chunk(result, max_words=20)
        self.assertLessEqual(len(summary.split()), 21)

    def test_summary_includes_title(self) -> None:
        result = self._make_result("Some text about market risks and competition.")
        summary = summarize_chunk(result, max_words=30)
        self.assertIn("Test Title", summary)


class AgentIntegrationTest(unittest.TestCase):
    def _make_fake_chunk(self, text: str, idx: int = 0) -> Chunk:
        return Chunk(
            chunk_id=f"web-fake-{idx}", doc_id=f"http://example.com/{idx}",
            title=f"Source {idx}", source=f"http://example.com/{idx}",
            published_at="2026-01-01", text=text,
            token_estimate=estimate_tokens(text),
        )

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"})
    @patch("research_agent.agent.genai")
    @patch("research_agent.agent.search_web")
    def test_agent_respects_context_budget(self, mock_search, mock_genai) -> None:
        mock_search.return_value = [
            self._make_fake_chunk("The EV market in this region is growing rapidly with strong demand.", i)
            for i in range(3)
        ]
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text="Mocked answer.")
        mock_genai.Client.return_value = mock_client

        agent = ResearchAgent(
            Path(__file__).resolve().parents[1],
            config=AgentConfig(max_context_tokens=300, max_memory_tokens=100, top_k_per_subquestion=2),
        )
        result = agent.answer("Compare Indonesia and Thailand for EV charging market entry.")

        self.assertLessEqual(result.context_tokens_used, 300)
        self.assertTrue(result.subquestions)
        self.assertTrue(result.answer)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"})
    @patch("research_agent.agent.genai")
    @patch("research_agent.agent.search_web")
    def test_agent_saves_to_memory(self, mock_search, mock_genai) -> None:
        mock_search.return_value = [self._make_fake_chunk("Some relevant text.", 0)]
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text="Answer line one.\nMore detail.")
        mock_genai.Client.return_value = mock_client

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "memory").mkdir()
            agent = ResearchAgent(tmp_path, config=AgentConfig())
            agent.answer("What is the state of EV adoption in Asia?")
            entries = agent.memory.load_recent(max_tokens=1000)
            self.assertEqual(len(entries), 1)


if __name__ == "__main__":
    unittest.main()
