"""Tests for PDF loader with metadata extraction."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.ingestion.loader import load_pdf


class TestLoadPdf:
    """Tests for load_pdf function."""

    @patch("src.ingestion.loader.PyPDFLoader")
    def test_returns_documents_with_text(self, mock_loader_cls):
        """load_pdf should return documents with page content."""
        mock_docs = [
            Document(
                page_content="Introduction to transformers.",
                metadata={"source": "/tmp/paper.pdf", "page": 0},
            ),
            Document(
                page_content="Methods section content.",
                metadata={"source": "/tmp/paper.pdf", "page": 1},
            ),
        ]
        mock_loader_cls.return_value.load.return_value = mock_docs

        result = load_pdf("/tmp/paper.pdf")
        assert len(result) == 2
        assert result[0].page_content == "Introduction to transformers."
        assert result[1].page_content == "Methods section content."

    @patch("src.ingestion.loader.PyPDFLoader")
    def test_metadata_includes_required_keys(self, mock_loader_cls):
        """Each document should have source, title, page, total_pages in metadata."""
        mock_docs = [
            Document(
                page_content="Some content",
                metadata={"source": "/tmp/paper.pdf", "page": 0},
            ),
        ]
        mock_loader_cls.return_value.load.return_value = mock_docs

        result = load_pdf("/tmp/paper.pdf")
        meta = result[0].metadata
        assert "source" in meta
        assert "title" in meta
        assert "page" in meta
        assert "total_pages" in meta

    @patch("src.ingestion.loader.PyPDFLoader")
    def test_total_pages_matches_doc_count(self, mock_loader_cls):
        """total_pages should reflect the number of pages loaded."""
        mock_docs = [
            Document(page_content=f"Page {i}", metadata={"source": "/tmp/p.pdf", "page": i})
            for i in range(5)
        ]
        mock_loader_cls.return_value.load.return_value = mock_docs

        result = load_pdf("/tmp/p.pdf")
        assert all(doc.metadata["total_pages"] == 5 for doc in result)

    @patch("src.ingestion.loader.PyPDFLoader")
    def test_title_extracted_from_filename(self, mock_loader_cls):
        """Title should default to the filename without extension."""
        mock_docs = [
            Document(
                page_content="Content",
                metadata={"source": "/tmp/attention_is_all_you_need.pdf", "page": 0},
            ),
        ]
        mock_loader_cls.return_value.load.return_value = mock_docs

        result = load_pdf("/tmp/attention_is_all_you_need.pdf")
        assert result[0].metadata["title"] == "attention_is_all_you_need"

    @patch("src.ingestion.loader.PyPDFLoader")
    def test_corrupted_pdf_raises_handled_exception(self, mock_loader_cls):
        """A corrupted PDF should raise a ValueError, not crash."""
        mock_loader_cls.return_value.load.side_effect = Exception("Cannot read PDF")

        with pytest.raises(ValueError, match="Failed to load PDF"):
            load_pdf("/tmp/corrupted.pdf")

    @patch("src.ingestion.loader.PyPDFLoader")
    def test_empty_pdf_returns_empty_list(self, mock_loader_cls):
        """A PDF with no extractable text should return empty list."""
        mock_loader_cls.return_value.load.return_value = []

        result = load_pdf("/tmp/empty.pdf")
        assert result == []
