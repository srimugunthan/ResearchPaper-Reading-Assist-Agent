"""Tests for streaming callback handler."""
import pytest

from src.qna.streaming import StreamingCollector


class TestStreamingCollector:
    """Tests for StreamingCollector callback handler."""

    def test_accumulates_tokens(self):
        """Tokens passed to on_llm_new_token should be accumulated in order."""
        collector = StreamingCollector()
        collector.on_llm_new_token("Hello")
        collector.on_llm_new_token(" world")
        collector.on_llm_new_token("!")
        assert collector.get_text() == "Hello world!"

    def test_tokens_list_accessible(self):
        """Individual tokens should be accessible as a list."""
        collector = StreamingCollector()
        collector.on_llm_new_token("A")
        collector.on_llm_new_token("B")
        assert collector.tokens == ["A", "B"]

    def test_on_llm_end_marks_complete(self):
        """on_llm_end should mark the collector as complete."""
        collector = StreamingCollector()
        collector.on_llm_new_token("Hello")
        assert not collector.is_complete
        collector.on_llm_end(response=None)
        assert collector.is_complete

    def test_reset_clears_state(self):
        """reset() should clear tokens and completion flag."""
        collector = StreamingCollector()
        collector.on_llm_new_token("Hello")
        collector.on_llm_end(response=None)
        collector.reset()
        assert collector.get_text() == ""
        assert collector.tokens == []
        assert not collector.is_complete

    def test_empty_collector(self):
        """A fresh collector should have empty text and not be complete."""
        collector = StreamingCollector()
        assert collector.get_text() == ""
        assert collector.tokens == []
        assert not collector.is_complete
