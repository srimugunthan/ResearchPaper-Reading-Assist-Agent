"""Tests for text splitter."""
import pytest

from src.ingestion.splitter import split_text, get_text_splitter


class TestSplitText:
    """Tests for split_text function."""

    def test_splits_long_text_into_chunks(self):
        """Long text should be split into multiple chunks."""
        text = "A" * 3000
        chunks = split_text(text, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) > 1

    def test_chunks_do_not_exceed_chunk_size(self):
        """Each chunk should be at most chunk_size characters."""
        text = "word " * 600  # ~3000 chars
        chunks = split_text(text, chunk_size=1000, chunk_overlap=200)
        for chunk in chunks:
            assert len(chunk) <= 1000

    def test_consecutive_chunks_overlap(self):
        """Consecutive chunks should share overlapping content."""
        # Use text with clear word boundaries
        words = [f"word{i}" for i in range(500)]
        text = " ".join(words)
        chunks = split_text(text, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) >= 2
        # Check that the end of chunk N overlaps with the start of chunk N+1
        for i in range(len(chunks) - 1):
            tail = chunks[i][-100:]  # last 100 chars of current chunk
            head = chunks[i + 1][:300]  # first 300 chars of next chunk
            # At least some content from the tail should appear in the head
            # (overlap means shared words)
            tail_words = set(tail.split())
            head_words = set(head.split())
            overlap = tail_words & head_words
            assert len(overlap) > 0, f"No overlap between chunks {i} and {i+1}"

    def test_short_text_returns_single_chunk(self):
        """Text shorter than chunk_size should return a single chunk."""
        text = "Short text here."
        chunks = split_text(text, chunk_size=1000, chunk_overlap=200)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_returns_empty_list(self):
        """Empty text should return empty list."""
        chunks = split_text("", chunk_size=1000, chunk_overlap=200)
        assert chunks == []

    def test_default_parameters(self):
        """split_text should work with default chunk_size=1000 and overlap=200."""
        text = "Hello world. " * 200
        chunks = split_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 1000


class TestGetTextSplitter:
    """Tests for get_text_splitter factory."""

    def test_returns_splitter_with_configured_params(self):
        """Should return a configured splitter instance."""
        splitter = get_text_splitter(chunk_size=500, chunk_overlap=100)
        assert splitter is not None
        assert splitter._chunk_size == 500
        assert splitter._chunk_overlap == 100
