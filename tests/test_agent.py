import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from research_agent import AgentConfig, ResearchAgent
from research_agent.agent import _classify_question
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

    def test_skips_corrupt_lines(self) -> None:
        self.store.append("Good question", "Good answer")
        with self.store.path.open("a") as f:
            f.write("THIS IS NOT JSON\n")
        self.store.append("Another good question", "Another good answer")
        entries = self.store.load_recent(max_tokens=1000)
        self.assertEqual(len(entries), 2)

    def test_ttl_expires_old_entries(self) -> None:
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        old_entry = json.dumps({"question": "Old Q", "summary": "Old A", "timestamp": old_ts})
        with self.store.path.open("a") as f:
            f.write(old_entry + "\n")
        self.store.append("Recent question", "Recent answer")
        entries = self.store.load_recent(max_tokens=1000)
        self.assertEqual(len(entries), 1)
        self.assertIn("Recent question", entries[0])

    def test_skips_entries_missing_fields(self) -> None:
        bad_entry = json.dumps({"timestamp": datetime.now(timezone.utc).isoformat()})
        with self.store.path.open("a") as f:
            f.write(bad_entry + "\n")
        self.store.append("Good question", "Good answer")
        entries = self.store.load_recent(max_tokens=1000)
        self.assertEqual(len(entries), 1)


class BudgetClassifierTest(unittest.TestCase):
    def test_new_topic_when_no_memory(self) -> None:
        self.assertEqual(_classify_question("What is quantum computing?", []), "new_topic")

    def test_follow_up_on_high_overlap(self) -> None:
        memory = ["Past session: What are EV charging risks? -> EV charging risks include regulation and safety."]
        result = _classify_question("What are the main EV charging risks in Asia?", memory)
        self.assertEqual(result, "follow_up")

    def test_new_topic_on_low_overlap(self) -> None:
        memory = ["Past session: What is quantum computing? -> Quantum computing uses qubits."]
        result = _classify_question("What are the EV charging trends in Southeast Asia?", memory)
        self.assertEqual(result, "new_topic")


class SummarizerTest(unittest.TestCase):
    def _make_result(self, text: str, query: str = "risk market") -> RetrievedChunk:
        chunk = Chunk(
            chunk_id="c1", doc_id="d1", title="Test Title",
            source="http://example.com", published_at="2026-01-01",
            text=text, token_estimate=estimate_tokens(text),
        )
        return RetrievedChunk(chunk=chunk, score=1.0, query=query)

    def test_summary_within_word_limit(self) -> None:
        result = self._make_result(
            "The market is growing fast. Risks include regulation. Competition is fierce. "
            "Infrastructure is lacking. Demand is uncertain.",
        )
        summary = summarize_chunk(result, max_words=20)
        self.assertLessEqual(len(summary.split()), 21)

    def test_summary_includes_title(self) -> None:
        result = self._make_result("Some text about market risks and competition.")
        summary = summarize_chunk(result, max_words=30)
        self.assertIn("Test Title", summary)

    def test_handles_empty_text(self) -> None:
        result = self._make_result("")
        summary = summarize_chunk(result, max_words=20)
        self.assertIsInstance(summary, str)

    def test_uses_query_terms_for_scoring(self) -> None:
        # Sentence matching the query should be preferred
        result = self._make_result(
            "Unrelated sentence about weather. The market risk is high in this region.",
            query="market risk",
        )
        summary = summarize_chunk(result, max_words=30)
        self.assertIn("market risk", summary.lower())


class AgentIntegrationTest(unittest.TestCase):
    def _make_fake_result(self, text: str, url: str = "http://example.com", score: float = 0.9, query: str = "test") -> RetrievedChunk:
        import hashlib
        stable_id = hashlib.md5(url.encode()).hexdigest()[:12]
        chunk = Chunk(
            chunk_id=f"web-{stable_id}",
            doc_id=url,
            title=f"Source at {url}",
            source=url,
            published_at="2026-01-01",
            text=text,
            token_estimate=estimate_tokens(text),
        )
        return RetrievedChunk(chunk=chunk, score=score, query=query)

    def _make_agent(self, mock_genai: MagicMock, tmp_path: Path, config: AgentConfig | None = None) -> ResearchAgent:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = MagicMock(text="Mocked answer.")
        mock_genai.Client.return_value = mock_client
        return ResearchAgent(tmp_path, config=config or AgentConfig())

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"})
    @patch("research_agent.agent.genai")
    @patch("research_agent.agent.search_web")
    def test_agent_respects_context_budget(self, mock_search, mock_genai) -> None:
        mock_search.return_value = [
            self._make_fake_result("The EV market in this region is growing rapidly with strong demand.", f"http://example.com/{i}")
            for i in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(
                mock_genai, Path(tmpdir),
                AgentConfig(max_context_tokens=300, max_memory_tokens=100, top_k_per_subquestion=2),
            )
            result = agent.answer("Compare Indonesia and Thailand for EV charging market entry.")
            self.assertLessEqual(result.context_tokens_used, 300)
            self.assertTrue(result.subquestions)
            self.assertTrue(result.answer)
            self.assertIn(result.query_type, ("new_topic", "default", "follow_up"))

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"})
    @patch("research_agent.agent.genai")
    @patch("research_agent.agent.search_web")
    def test_agent_saves_to_memory(self, mock_search, mock_genai) -> None:
        mock_search.return_value = [self._make_fake_result("Some relevant text.")]
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(mock_genai, Path(tmpdir))
            agent.answer("What is the state of EV adoption in Asia?")
            entries = agent.memory.load_recent(max_tokens=1000)
            self.assertEqual(len(entries), 1)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"})
    @patch("research_agent.agent.genai")
    @patch("research_agent.agent.search_web")
    def test_empty_question_raises(self, mock_search, mock_genai) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(mock_genai, Path(tmpdir))
            with self.assertRaises(ValueError):
                agent.answer("   ")

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"})
    @patch("research_agent.agent.genai")
    @patch("research_agent.agent.search_web")
    def test_decompose_falls_back_on_bad_json(self, mock_search, mock_genai) -> None:
        mock_search.return_value = [self._make_fake_result("Some text.")]
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            MagicMock(text="not valid json at all"),
            MagicMock(text="Fallback answer."),
        ]
        mock_genai.Client.return_value = mock_client
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = ResearchAgent(Path(tmpdir), config=AgentConfig())
            result = agent.answer("What is quantum computing?")
            self.assertTrue(result.subquestions)
            self.assertTrue(result.answer)

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"})
    @patch("research_agent.agent.genai")
    @patch("research_agent.agent.search_web")
    def test_deduplication_across_subquestions(self, mock_search, mock_genai) -> None:
        # Same URL returned by two different sub-question searches
        same_result = self._make_fake_result("Shared content.", url="http://shared.com")
        mock_search.side_effect = [
            [same_result],
            [same_result, self._make_fake_result("Unique content.", url="http://unique.com")],
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = self._make_agent(mock_genai, Path(tmpdir))
            result = agent.answer("Compare X and Y")
            urls = [r.chunk.source for r in result.retrieved]
            self.assertEqual(len(urls), len(set(urls)), "Duplicate URLs found in retrieved results")


if __name__ == "__main__":
    unittest.main()
