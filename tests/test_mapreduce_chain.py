"""Tests for multi-paper synthesis (map-reduce and refine)."""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel

from src.synthesis.synthesizer import (
    synthesize,
    synthesize_map_reduce,
    synthesize_refine,
    build_map_prompt,
    build_reduce_prompt,
    build_refine_prompt,
    build_initial_refine_prompt,
    _group_docs_by_title,
    _combine_doc_content,
)


def _make_docs():
    """Create synthetic documents from 3 papers."""
    return [
        Document(
            page_content="Paper A discusses attention mechanisms in detail.",
            metadata={"title": "Paper A", "authors": "Smith", "page": 1, "source": "a.pdf"},
        ),
        Document(
            page_content="Paper A also covers multi-head transformers.",
            metadata={"title": "Paper A", "authors": "Smith", "page": 2, "source": "a.pdf"},
        ),
        Document(
            page_content="Paper B proposes BERT for pre-training.",
            metadata={"title": "Paper B", "authors": "Jones", "page": 5, "source": "b.pdf"},
        ),
        Document(
            page_content="Paper C compares different architectures.",
            metadata={"title": "Paper C", "authors": "Lee", "page": 3, "source": "c.pdf"},
        ),
    ]


class TestGroupDocsByTitle:
    """Tests for _group_docs_by_title helper."""

    def test_groups_correctly(self):
        docs = _make_docs()
        groups = _group_docs_by_title(docs)
        assert len(groups) == 3
        assert len(groups["Paper A"]) == 2
        assert len(groups["Paper B"]) == 1
        assert len(groups["Paper C"]) == 1

    def test_empty_docs(self):
        assert _group_docs_by_title([]) == {}


class TestCombineDocContent:
    """Tests for _combine_doc_content helper."""

    def test_combines_content(self):
        docs = [
            Document(page_content="Part 1.", metadata={}),
            Document(page_content="Part 2.", metadata={}),
        ]
        result = _combine_doc_content(docs)
        assert "Part 1." in result
        assert "Part 2." in result

    def test_truncation(self):
        docs = [Document(page_content="x" * 5000, metadata={})]
        result = _combine_doc_content(docs, max_chars=100)
        assert len(result) <= 100


class TestBuildPrompts:
    """Tests for prompt template builders."""

    def test_map_prompt_has_required_vars(self):
        prompt = build_map_prompt()
        text = prompt.format(title="T", content="C", question="Q")
        assert "T" in text
        assert "C" in text
        assert "Q" in text

    def test_reduce_prompt_has_required_vars(self):
        prompt = build_reduce_prompt()
        text = prompt.format(question="Q", summaries="S")
        assert "Q" in text
        assert "S" in text

    def test_refine_prompt_has_required_vars(self):
        prompt = build_refine_prompt()
        text = prompt.format(
            current_synthesis="CS", title="T", new_summary="NS", question="Q"
        )
        assert "CS" in text
        assert "T" in text

    def test_initial_refine_prompt_has_required_vars(self):
        prompt = build_initial_refine_prompt()
        text = prompt.format(title="T", summary="S", question="Q")
        assert "T" in text
        assert "S" in text


class TestSynthesizeMapReduce:
    """Tests for map-reduce synthesis."""

    def test_map_called_once_per_paper(self):
        docs = _make_docs()  # 3 unique papers
        # 3 map responses + 1 reduce response = 4 LLM calls
        fake_llm = FakeListChatModel(
            responses=[
                "Summary of Paper A about attention and transformers.",
                "Summary of Paper B about BERT.",
                "Summary of Paper C about architecture comparison.",
                "**Common Themes**: All papers discuss neural architectures.",
            ]
        )
        result = synthesize_map_reduce(docs, question="Compare architectures", llm=fake_llm)
        assert result["papers_analyzed"] == 3
        assert len(result["paper_summaries"]) == 3
        assert result["strategy"] == "map_reduce"

    def test_reduce_produces_synthesis(self):
        docs = _make_docs()
        fake_llm = FakeListChatModel(
            responses=["Summary A", "Summary B", "Summary C", "Final synthesis."]
        )
        result = synthesize_map_reduce(docs, question="Compare", llm=fake_llm)
        assert "synthesis" in result
        assert len(result["synthesis"]) > 0

    def test_sources_contain_all_papers(self):
        docs = _make_docs()
        fake_llm = FakeListChatModel(
            responses=["SA", "SB", "SC", "Final."]
        )
        result = synthesize_map_reduce(docs, question="Compare", llm=fake_llm)
        source_titles = [s["title"] for s in result["sources"]]
        assert "Paper A" in source_titles
        assert "Paper B" in source_titles
        assert "Paper C" in source_titles

    def test_sources_include_page_numbers(self):
        docs = _make_docs()
        fake_llm = FakeListChatModel(responses=["S1", "S2", "S3", "Final"])
        result = synthesize_map_reduce(docs, question="Q", llm=fake_llm)
        paper_a_source = next(s for s in result["sources"] if s["title"] == "Paper A")
        assert 1 in paper_a_source["pages"]
        assert 2 in paper_a_source["pages"]


class TestSynthesizeRefine:
    """Tests for refine synthesis."""

    def test_refine_processes_all_papers(self):
        docs = _make_docs()  # 3 papers
        # 3 map calls + 1 initial refine + 2 refine iterations = 6 LLM calls
        fake_llm = FakeListChatModel(
            responses=[
                "Summary A",
                "Initial synthesis based on Paper A.",
                "Summary B",
                "Refined synthesis incorporating Paper B.",
                "Summary C",
                "Final refined synthesis incorporating Paper C.",
            ]
        )
        result = synthesize_refine(docs, question="Compare", llm=fake_llm)
        assert result["papers_analyzed"] == 3
        assert result["strategy"] == "refine"
        assert len(result["paper_summaries"]) == 3

    def test_refine_returns_final_synthesis(self):
        docs = _make_docs()
        fake_llm = FakeListChatModel(
            responses=["S1", "Init", "S2", "Refined1", "S3", "Final refined."]
        )
        result = synthesize_refine(docs, question="Q", llm=fake_llm)
        assert result["synthesis"] == "Final refined."

    def test_refine_sources_contain_all_papers(self):
        docs = _make_docs()
        fake_llm = FakeListChatModel(
            responses=["S1", "Init", "S2", "R1", "S3", "Final"]
        )
        result = synthesize_refine(docs, question="Q", llm=fake_llm)
        source_titles = [s["title"] for s in result["sources"]]
        assert "Paper A" in source_titles
        assert "Paper B" in source_titles
        assert "Paper C" in source_titles


class TestSynthesize:
    """Tests for the synthesize entry point."""

    def test_map_reduce_strategy(self):
        docs = _make_docs()
        fake_llm = FakeListChatModel(responses=["S1", "S2", "S3", "Final"])
        result = synthesize(docs, question="Q", llm=fake_llm, strategy="map_reduce")
        assert result["strategy"] == "map_reduce"

    def test_refine_strategy(self):
        docs = _make_docs()
        fake_llm = FakeListChatModel(
            responses=["S1", "Init", "S2", "R1", "S3", "Final"]
        )
        result = synthesize(docs, question="Q", llm=fake_llm, strategy="refine")
        assert result["strategy"] == "refine"

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            synthesize(
                [Document(page_content="x", metadata={"title": "T"})],
                question="Q",
                llm=MagicMock(),
                strategy="unknown",
            )

    def test_empty_docs_raises(self):
        with pytest.raises(ValueError, match="No documents"):
            synthesize([], question="Q", llm=MagicMock(), strategy="map_reduce")

    def test_single_paper_works(self):
        docs = [
            Document(
                page_content="Only paper content.",
                metadata={"title": "Solo Paper", "page": 1, "source": "solo.pdf"},
            )
        ]
        fake_llm = FakeListChatModel(
            responses=["Summary of solo paper.", "Synthesis of single paper."]
        )
        result = synthesize(docs, question="Q", llm=fake_llm, strategy="map_reduce")
        assert result["papers_analyzed"] == 1
        assert len(result["sources"]) == 1
