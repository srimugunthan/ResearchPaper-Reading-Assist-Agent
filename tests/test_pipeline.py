"""Tests for the ingestion pipeline (end-to-end with mocks)."""
import os
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.ingestion.pipeline import ingest_folder, ingest_pdf, compute_file_hash


class TestComputeFileHash:
    """Tests for file hashing."""

    def test_same_content_same_hash(self, tmp_path):
        """Two files with the same content should have the same hash."""
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"same content")
        f2.write_bytes(b"same content")
        assert compute_file_hash(str(f1)) == compute_file_hash(str(f2))

    def test_different_content_different_hash(self, tmp_path):
        """Two files with different content should have different hashes."""
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert compute_file_hash(str(f1)) != compute_file_hash(str(f2))


class TestIngestPdf:
    """Tests for single PDF ingestion."""

    @patch("src.ingestion.pipeline.load_pdf")
    def test_ingest_single_pdf(self, mock_load, tmp_path):
        """ingest_pdf should load, split, and store a single PDF."""
        mock_load.return_value = [
            Document(
                page_content="A " * 600,  # ~1200 chars, will be split
                metadata={"source": "/tmp/paper.pdf", "title": "Paper", "page": 0, "total_pages": 1},
            ),
        ]

        from src.core.embeddings import FakeEmbeddings
        from src.core.vectorstore import get_vectorstore

        store = get_vectorstore(
            persist_directory=str(tmp_path / "chroma_db"),
            embedding_function=FakeEmbeddings(),
        )

        count = ingest_pdf("/tmp/paper.pdf", store)
        assert count > 0
        mock_load.assert_called_once_with("/tmp/paper.pdf")


class TestIngestFolder:
    """Tests for folder ingestion pipeline."""

    @patch("src.ingestion.pipeline.load_pdf")
    def test_ingests_all_pdfs_in_folder(self, mock_load, tmp_path):
        """ingest_folder should process all PDFs found in folder."""
        # Create fake PDFs
        pdf_dir = tmp_path / "papers"
        pdf_dir.mkdir()
        (pdf_dir / "paper1.pdf").write_bytes(b"PDF content 1")
        (pdf_dir / "paper2.pdf").write_bytes(b"PDF content 2")

        mock_load.return_value = [
            Document(
                page_content="Content of paper.",
                metadata={"source": "test.pdf", "title": "Test", "page": 0, "total_pages": 1},
            ),
        ]

        from src.core.embeddings import FakeEmbeddings

        result = ingest_folder(
            folder_path=str(pdf_dir),
            persist_directory=str(tmp_path / "chroma_db"),
            embedding_function=FakeEmbeddings(),
        )

        assert result["total_files"] == 2
        assert result["files_processed"] == 2
        assert mock_load.call_count == 2

    @patch("src.ingestion.pipeline.load_pdf")
    def test_deduplication_skips_already_ingested(self, mock_load, tmp_path):
        """Running ingest_folder twice should skip already-ingested files."""
        pdf_dir = tmp_path / "papers"
        pdf_dir.mkdir()
        (pdf_dir / "paper1.pdf").write_bytes(b"PDF content 1")

        mock_load.return_value = [
            Document(
                page_content="Content of paper.",
                metadata={"source": "paper1.pdf", "title": "Paper1", "page": 0, "total_pages": 1},
            ),
        ]

        from src.core.embeddings import FakeEmbeddings

        persist_dir = str(tmp_path / "chroma_db")
        emb = FakeEmbeddings()

        # First run
        result1 = ingest_folder(str(pdf_dir), persist_dir, emb)
        assert result1["files_processed"] == 1

        # Second run — should skip
        mock_load.reset_mock()
        result2 = ingest_folder(str(pdf_dir), persist_dir, emb)
        assert result2["files_skipped"] == 1
        assert result2["files_processed"] == 0
        mock_load.assert_not_called()

    @patch("src.ingestion.pipeline.load_pdf")
    def test_handles_load_failure_gracefully(self, mock_load, tmp_path):
        """If a PDF fails to load, the pipeline should log it and continue."""
        pdf_dir = tmp_path / "papers"
        pdf_dir.mkdir()
        (pdf_dir / "good.pdf").write_bytes(b"Good PDF")
        (pdf_dir / "bad.pdf").write_bytes(b"Bad PDF")

        def side_effect(path):
            if "bad" in path:
                raise ValueError("Failed to load PDF")
            return [
                Document(
                    page_content="Good content.",
                    metadata={"source": path, "title": "Good", "page": 0, "total_pages": 1},
                ),
            ]

        mock_load.side_effect = side_effect

        from src.core.embeddings import FakeEmbeddings

        result = ingest_folder(
            str(pdf_dir),
            str(tmp_path / "chroma_db"),
            FakeEmbeddings(),
        )

        assert result["files_processed"] == 1
        assert result["files_failed"] == 1
        assert len(result["errors"]) == 1
