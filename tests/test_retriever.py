"""Tests for retriever wrapper."""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document

from src.core.retriever import create_retriever, retrieve


class TestCreateRetriever:
    """Tests for retriever creation."""

    def test_creates_retriever_from_vectorstore(self):
        """create_retriever should return a retriever from a vector store."""
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = MagicMock()
        retriever = create_retriever(mock_store, k=3)
        assert retriever is not None

    def test_k_parameter_is_passed(self):
        """The k parameter should be forwarded to the retriever."""
        mock_store = MagicMock()
        mock_store.as_retriever.return_value = MagicMock()
        create_retriever(mock_store, k=7)
        mock_store.as_retriever.assert_called_once()
        call_kwargs = mock_store.as_retriever.call_args
        search_kwargs = call_kwargs.kwargs.get("search_kwargs", {})
        assert search_kwargs.get("k") == 7


class TestRetrieve:
    """Tests for the retrieve function."""

    def test_returns_documents_with_metadata(self):
        """retrieve should return documents with source, title, page metadata."""
        mock_docs = [
            Document(
                page_content=f"Content {i}",
                metadata={"source": f"paper{i}.pdf", "title": f"Paper {i}", "page": i},
            )
            for i in range(3)
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs

        results = retrieve(mock_retriever, "attention mechanism")
        assert len(results) == 3
        for doc in results:
            assert "source" in doc.metadata
            assert "title" in doc.metadata
            assert "page" in doc.metadata

    def test_returns_correct_number_of_docs(self):
        """retrieve should return the number of documents the retriever provides."""
        mock_docs = [
            Document(page_content="doc", metadata={"source": "a.pdf", "title": "A", "page": 0})
        ]
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = mock_docs

        results = retrieve(mock_retriever, "query")
        assert len(results) == 1

    def test_empty_result_returns_empty_list(self):
        """retrieve should return empty list when no docs match."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        results = retrieve(mock_retriever, "nonexistent topic")
        assert results == []

    def test_query_is_passed_to_retriever(self):
        """The query string should be passed to the retriever's invoke."""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []

        retrieve(mock_retriever, "transformer architecture")
        mock_retriever.invoke.assert_called_once_with("transformer architecture")
