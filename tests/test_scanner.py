"""Tests for PDF file scanner."""
import os
import tempfile
import pytest

from src.ingestion.scanner import scan_folder


class TestScanFolder:
    """Tests for scan_folder function."""

    def test_discovers_pdf_files(self, tmp_path):
        """scan_folder should return only .pdf files."""
        (tmp_path / "paper1.pdf").touch()
        (tmp_path / "paper2.pdf").touch()
        (tmp_path / "notes.txt").touch()
        (tmp_path / "data.docx").touch()

        result = scan_folder(str(tmp_path))
        assert len(result) == 2
        assert all(f.endswith(".pdf") for f in result)

    def test_discovers_pdfs_in_nested_directories(self, tmp_path):
        """scan_folder should find PDFs recursively in subdirectories."""
        subdir = tmp_path / "subdir" / "deep"
        subdir.mkdir(parents=True)
        (tmp_path / "top.pdf").touch()
        (subdir / "nested.pdf").touch()

        result = scan_folder(str(tmp_path))
        assert len(result) == 2
        filenames = [os.path.basename(f) for f in result]
        assert "top.pdf" in filenames
        assert "nested.pdf" in filenames

    def test_empty_folder_returns_empty_list(self, tmp_path):
        """scan_folder should return empty list for folder with no PDFs."""
        (tmp_path / "readme.txt").touch()
        result = scan_folder(str(tmp_path))
        assert result == []

    def test_completely_empty_folder(self, tmp_path):
        """scan_folder should return empty list for completely empty folder."""
        result = scan_folder(str(tmp_path))
        assert result == []

    def test_nonexistent_folder_raises(self):
        """scan_folder should raise FileNotFoundError for missing folder."""
        with pytest.raises(FileNotFoundError):
            scan_folder("/nonexistent/path/to/folder")

    def test_returns_absolute_paths(self, tmp_path):
        """scan_folder should return absolute file paths."""
        (tmp_path / "paper.pdf").touch()
        result = scan_folder(str(tmp_path))
        assert len(result) == 1
        assert os.path.isabs(result[0])

    def test_ignores_hidden_pdf_files(self, tmp_path):
        """scan_folder should skip hidden files (starting with dot)."""
        (tmp_path / ".hidden.pdf").touch()
        (tmp_path / "visible.pdf").touch()
        result = scan_folder(str(tmp_path))
        assert len(result) == 1
        assert "visible.pdf" in result[0]
