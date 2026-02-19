"""Tests for metadata filter utilities."""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

from src.synthesis.filters import (
    build_title_filter,
    build_source_filter,
    build_combined_filter,
    retrieve_by_filter,
    get_unique_titles,
)


class TestBuildTitleFilter:
    """Tests for build_title_filter."""

    def test_single_title(self):
        result = build_title_filter(["Paper A"])
        assert result == {"title": "Paper A"}

    def test_multiple_titles(self):
        result = build_title_filter(["Paper A", "Paper B"])
        assert result == {"title": {"$in": ["Paper A", "Paper B"]}}

    def test_empty_list_returns_none(self):
        assert build_title_filter([]) is None


class TestBuildSourceFilter:
    """Tests for build_source_filter."""

    def test_single_source(self):
        result = build_source_filter(["/path/a.pdf"])
        assert result == {"source": "/path/a.pdf"}

    def test_multiple_sources(self):
        result = build_source_filter(["/a.pdf", "/b.pdf"])
        assert result == {"source": {"$in": ["/a.pdf", "/b.pdf"]}}

    def test_empty_returns_none(self):
        assert build_source_filter([]) is None


class TestBuildCombinedFilter:
    """Tests for build_combined_filter."""

    def test_title_only(self):
        result = build_combined_filter(titles=["Paper A"])
        assert result == {"title": "Paper A"}

    def test_source_only(self):
        result = build_combined_filter(sources=["/a.pdf"])
        assert result == {"source": "/a.pdf"}

    def test_both_combined_with_and(self):
        result = build_combined_filter(titles=["Paper A"], sources=["/a.pdf"])
        assert "$and" in result
        assert len(result["$and"]) == 2

    def test_no_filters_returns_none(self):
        assert build_combined_filter() is None


class TestRetrieveByFilter:
    """Tests for retrieve_by_filter."""

    def test_calls_similarity_search_with_filter(self):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            Document(page_content="content", metadata={"title": "Paper A", "page": 1})
        ]
        where = {"title": "Paper A"}
        results = retrieve_by_filter(mock_store, query="attention", where=where, k=10)
        mock_store.similarity_search.assert_called_once_with("attention", k=10, filter=where)
        assert len(results) == 1

    def test_no_filter_passes_none(self):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []
        retrieve_by_filter(mock_store, query="test", where=None, k=5)
        mock_store.similarity_search.assert_called_once_with("test", k=5, filter=None)

    def test_returns_empty_list_when_no_matches(self):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []
        results = retrieve_by_filter(mock_store, query="nothing", k=5)
        assert results == []


class TestGetUniqueTitles:
    """Tests for get_unique_titles."""

    def test_returns_unique_sorted_titles(self):
        mock_store = MagicMock()
        mock_store.get.return_value = {
            "metadatas": [
                {"title": "Paper B"},
                {"title": "Paper A"},
                {"title": "Paper B"},
                {"title": "Paper C"},
            ]
        }
        titles = get_unique_titles(mock_store)
        assert titles == ["Paper A", "Paper B", "Paper C"]

    def test_empty_store_returns_empty_list(self):
        mock_store = MagicMock()
        mock_store.get.return_value = {"metadatas": []}
        titles = get_unique_titles(mock_store)
        assert titles == []

    def test_handles_missing_title_metadata(self):
        mock_store = MagicMock()
        mock_store.get.return_value = {
            "metadatas": [
                {"title": "Paper A"},
                {"source": "no_title.pdf"},
                {"title": "Paper B"},
            ]
        }
        titles = get_unique_titles(mock_store)
        assert titles == ["Paper A", "Paper B"]
