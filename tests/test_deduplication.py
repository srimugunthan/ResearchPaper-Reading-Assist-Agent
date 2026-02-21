"""Tests for post-retrieval deduplication."""
import pytest
from langchain_core.documents import Document
from src.core.vectorstore import deduplicate_results


class TestDeduplicateResults:
    def test_removes_identical_content(self):
        docs = [
            Document(page_content="The transformer model uses attention.", metadata={"page": 1}),
            Document(page_content="The transformer model uses attention.", metadata={"page": 2}),
        ]
        result = deduplicate_results(docs)
        assert len(result) == 1
        assert result[0].metadata["page"] == 1  # Keeps first (highest-ranked)

    def test_keeps_distinct_content(self):
        docs = [
            Document(page_content="Transformers use self-attention mechanisms.", metadata={}),
            Document(page_content="CNNs use convolutional filters for feature extraction.", metadata={}),
        ]
        result = deduplicate_results(docs)
        assert len(result) == 2

    def test_empty_input(self):
        assert deduplicate_results([]) == []

    def test_single_document(self):
        docs = [Document(page_content="Only one document here.", metadata={})]
        result = deduplicate_results(docs)
        assert len(result) == 1

    def test_preserves_order(self):
        docs = [
            Document(page_content="First unique content about methods and approaches.", metadata={"rank": 1}),
            Document(page_content="Second unique content about results and findings.", metadata={"rank": 2}),
            Document(page_content="First unique content about methods and approaches.", metadata={"rank": 3}),
        ]
        result = deduplicate_results(docs)
        assert len(result) == 2
        assert result[0].metadata["rank"] == 1
        assert result[1].metadata["rank"] == 2

    def test_near_duplicate_with_high_similarity(self):
        docs = [
            Document(page_content="The transformer model uses self-attention for sequence processing.", metadata={}),
            Document(page_content="The transformer model uses self-attention for sequence processing tasks.", metadata={}),
        ]
        result = deduplicate_results(docs, similarity_threshold=0.9)
        assert len(result) == 1

    def test_custom_threshold_strict(self):
        docs = [
            Document(page_content="The cat sat on the mat.", metadata={}),
            Document(page_content="The cat sat on a mat.", metadata={}),
        ]
        # With very high threshold, both kept (they differ slightly)
        result = deduplicate_results(docs, similarity_threshold=0.99)
        assert len(result) == 2

    def test_custom_threshold_loose(self):
        docs = [
            Document(page_content="The cat sat on the mat.", metadata={}),
            Document(page_content="The cat sat on a mat.", metadata={}),
        ]
        # With loose threshold, duplicate removed
        result = deduplicate_results(docs, similarity_threshold=0.7)
        assert len(result) == 1

    def test_multiple_duplicates_groups(self):
        docs = [
            Document(page_content="Content A about transformers.", metadata={"group": "A"}),
            Document(page_content="Content B about CNNs.", metadata={"group": "B"}),
            Document(page_content="Content A about transformers.", metadata={"group": "A_dup"}),
            Document(page_content="Content B about CNNs.", metadata={"group": "B_dup"}),
        ]
        result = deduplicate_results(docs)
        assert len(result) == 2
        assert result[0].metadata["group"] == "A"
        assert result[1].metadata["group"] == "B"

    def test_preserves_metadata(self):
        docs = [
            Document(
                page_content="Some unique content.",
                metadata={"title": "Paper A", "page": 3, "chunk_type": "abstract", "section": "Abstract"},
            ),
        ]
        result = deduplicate_results(docs)
        assert result[0].metadata["title"] == "Paper A"
        assert result[0].metadata["chunk_type"] == "abstract"
        assert result[0].metadata["section"] == "Abstract"
