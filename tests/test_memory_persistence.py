"""Tests for conversation memory persistence."""
import pytest

from src.memory.memory import ConversationMemory


@pytest.fixture
def memory(tmp_path):
    """Create a ConversationMemory backed by a temp file."""
    filepath = str(tmp_path / "conversation_memory.json")
    return ConversationMemory(filepath=filepath)


class TestConversationMemory:
    """Tests for ConversationMemory."""

    def test_save_and_load_exchange(self, memory):
        """Saving an exchange should make it retrievable."""
        memory.save_exchange(
            user_input="What is self-attention?",
            assistant_output="Self-attention is a mechanism..."
        )
        history = memory.get_history()
        assert len(history) == 1
        assert history[0]["input"] == "What is self-attention?"
        assert history[0]["output"] == "Self-attention is a mechanism..."

    def test_multiple_exchanges(self, memory):
        """Multiple exchanges should be stored in order."""
        memory.save_exchange("Q1", "A1")
        memory.save_exchange("Q2", "A2")
        memory.save_exchange("Q3", "A3")
        history = memory.get_history()
        assert len(history) == 3
        assert history[0]["input"] == "Q1"
        assert history[2]["input"] == "Q3"

    def test_get_recent_context(self, memory):
        """get_recent_context should return only the last N exchanges."""
        for i in range(10):
            memory.save_exchange(f"Q{i}", f"A{i}")
        recent = memory.get_recent_context(n=3)
        assert len(recent) == 3
        assert recent[0]["input"] == "Q7"
        assert recent[2]["input"] == "Q9"

    def test_clear_memory(self, memory):
        """clear() should wipe all stored exchanges."""
        memory.save_exchange("Q1", "A1")
        memory.save_exchange("Q2", "A2")
        memory.clear()
        history = memory.get_history()
        assert len(history) == 0

    def test_memory_functional_after_clear(self, memory):
        """Memory should still work after clearing."""
        memory.save_exchange("Q1", "A1")
        memory.clear()
        memory.save_exchange("Q2", "A2")
        history = memory.get_history()
        assert len(history) == 1
        assert history[0]["input"] == "Q2"

    def test_persistence_across_instances(self, tmp_path):
        """Memory should persist across instances using the same file."""
        filepath = str(tmp_path / "memory.json")
        mem1 = ConversationMemory(filepath=filepath)
        mem1.save_exchange("Q1", "A1")

        mem2 = ConversationMemory(filepath=filepath)
        history = mem2.get_history()
        assert len(history) == 1
        assert history[0]["input"] == "Q1"

    def test_exchange_has_timestamp(self, memory):
        """Each exchange should have a timestamp."""
        memory.save_exchange("Q1", "A1")
        history = memory.get_history()
        assert "timestamp" in history[0]

    def test_format_context_string(self, memory):
        """format_context should return a readable string of recent exchanges."""
        memory.save_exchange("What is BERT?", "BERT is a model...")
        memory.save_exchange("How does it work?", "It uses masked language modeling...")
        context = memory.format_context(n=2)
        assert "What is BERT?" in context
        assert "BERT is a model" in context
        assert "How does it work?" in context

    def test_empty_context_string(self, memory):
        """format_context with no history should return empty string."""
        context = memory.format_context(n=5)
        assert context == ""
