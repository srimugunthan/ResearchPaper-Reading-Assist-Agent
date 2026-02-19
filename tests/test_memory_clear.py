"""Tests for memory clear functionality."""
import os
import pytest

from src.memory.memory import ConversationMemory


@pytest.fixture
def memory(tmp_path):
    filepath = str(tmp_path / "memory.json")
    return ConversationMemory(filepath=filepath)


class TestMemoryClear:
    """Tests for clearing conversation memory."""

    def test_clear_removes_all_exchanges(self, memory):
        """clear() should remove all stored exchanges."""
        memory.save_exchange("Q1", "A1")
        memory.save_exchange("Q2", "A2")
        memory.save_exchange("Q3", "A3")
        memory.clear()
        assert len(memory.get_history()) == 0

    def test_clear_persists_to_file(self, memory, tmp_path):
        """After clear(), the file should reflect empty state."""
        memory.save_exchange("Q1", "A1")
        memory.clear()
        # Create new instance from same file
        filepath = str(tmp_path / "memory.json")
        mem2 = ConversationMemory(filepath=filepath)
        assert len(mem2.get_history()) == 0

    def test_query_after_clear_returns_empty(self, memory):
        """Querying after clear should return no results."""
        memory.save_exchange("Tell me about transformers", "Transformers are...")
        memory.clear()
        history = memory.get_history()
        assert len(history) == 0

    def test_new_exchanges_after_clear(self, memory):
        """New exchanges after clear should work normally."""
        memory.save_exchange("Old Q", "Old A")
        memory.clear()
        memory.save_exchange("New Q", "New A")
        history = memory.get_history()
        assert len(history) == 1
        assert history[0]["input"] == "New Q"
        assert history[0]["output"] == "New A"
