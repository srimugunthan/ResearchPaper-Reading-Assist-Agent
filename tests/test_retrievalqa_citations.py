"""Tests for RetrievalQA chain with citation formatting."""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.qna.qa import ask, create_qa_chain
from src.qna.prompts import build_qa_prompt, format_citations


class TestFormatCitations:
    """Tests for citation formatting."""

    def test_formats_single_citation(self):
        """Should format a single document into a citation string."""
        doc = Document(
            page_content="Self-attention is...",
            metadata={"title": "Attention Is All You Need", "authors": "Vaswani et al.", "page": 5},
        )
        citation = format_citations([doc])
        assert "Attention Is All You Need" in citation
        assert "Vaswani et al." in citation
        assert "p.5" in citation

    def test_formats_multiple_citations(self):
        """Should format multiple documents into separate citation lines."""
        docs = [
            Document(page_content="A", metadata={"title": "Paper A", "authors": "Smith", "page": 1}),
            Document(page_content="B", metadata={"title": "Paper B", "authors": "Jones", "page": 10}),
        ]
        citation = format_citations(docs)
        assert "Paper A" in citation
        assert "Paper B" in citation

    def test_handles_missing_authors(self):
        """Should handle documents without authors metadata gracefully."""
        doc = Document(
            page_content="Some text",
            metadata={"title": "A Paper", "page": 3},
        )
        citation = format_citations([doc])
        assert "A Paper" in citation
        assert "p.3" in citation

    def test_empty_docs_returns_empty_string(self):
        """Empty doc list should return empty string."""
        assert format_citations([]) == ""


class TestBuildQaPrompt:
    """Tests for QA prompt template."""

    def test_prompt_includes_citation_instruction(self):
        """The prompt template should instruct the LLM to cite sources."""
        prompt = build_qa_prompt()
        prompt_text = prompt.format(context="some context", question="some question")
        assert "cite" in prompt_text.lower() or "source" in prompt_text.lower()

    def test_prompt_includes_context_and_question_vars(self):
        """The prompt template should have context and question variables."""
        prompt = build_qa_prompt()
        prompt_text = prompt.format(context="CONTEXT_PLACEHOLDER", question="QUESTION_PLACEHOLDER")
        assert "CONTEXT_PLACEHOLDER" in prompt_text
        assert "QUESTION_PLACEHOLDER" in prompt_text


class TestAsk:
    """Tests for the ask function."""

    def test_returns_answer_with_citations(self):
        """ask() should return an answer dict with 'answer' and 'sources' keys."""
        mock_docs = [
            Document(
                page_content="Self-attention allows...",
                metadata={"title": "Attention Is All You Need", "authors": "Vaswani et al.", "page": 5},
            ),
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs

        from langchain_core.language_models import FakeListChatModel
        fake_llm = FakeListChatModel(responses=[
            "Self-attention allows the model to attend to all positions. [Attention Is All You Need, Vaswani et al., p.5]"
        ])

        result = ask(
            question="What is self-attention?",
            retriever=mock_retriever,
            llm=fake_llm,
        )

        assert "answer" in result
        assert "sources" in result
        assert len(result["sources"]) == 1
        assert "Attention Is All You Need" in result["answer"]

    def test_no_docs_returns_idk_response(self):
        """When no docs are retrieved, should return an 'I don't know' response."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        from langchain_core.language_models import FakeListChatModel
        fake_llm = FakeListChatModel(responses=["I don't have enough information to answer."])

        result = ask(
            question="What is quantum gravity?",
            retriever=mock_retriever,
            llm=fake_llm,
        )

        assert "answer" in result
        assert result["sources"] == []

    def test_sources_contain_metadata(self):
        """Each source in the result should contain title, authors, and page."""
        mock_docs = [
            Document(
                page_content="BERT uses...",
                metadata={"title": "BERT", "authors": "Devlin et al.", "page": 12},
            ),
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs

        from langchain_core.language_models import FakeListChatModel
        fake_llm = FakeListChatModel(responses=["BERT uses masked language modeling."])

        result = ask(
            question="How does BERT work?",
            retriever=mock_retriever,
            llm=fake_llm,
        )

        assert len(result["sources"]) == 1
        src = result["sources"][0]
        assert src["title"] == "BERT"
        assert src["authors"] == "Devlin et al."
        assert src["page"] == 12
