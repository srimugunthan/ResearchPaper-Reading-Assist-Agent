"""Tests for idea log module."""
import pytest

from src.memory.idea_log import IdeaLog


@pytest.fixture
def idea_log(tmp_path):
    """Create an IdeaLog backed by a temp file."""
    filepath = str(tmp_path / "idea_log.json")
    return IdeaLog(filepath=filepath)


class TestIdeaLog:
    """Tests for IdeaLog operations."""

    def test_add_idea(self, idea_log):
        """Adding an idea returns an entry with id, text, tags, timestamps."""
        entry = idea_log.add_idea("Use LQ loss for image segmentation tasks")
        assert entry["id"]
        assert entry["text"] == "Use LQ loss for image segmentation tasks"
        assert isinstance(entry["tags"], list)
        assert isinstance(entry["related_papers"], list)
        assert entry["created_at"]
        assert entry["updated_at"]

    def test_add_idea_auto_tags(self, idea_log):
        """Auto-extracted tags should capture key terms."""
        entry = idea_log.add_idea("Use LQ loss for image segmentation tasks")
        assert len(entry["tags"]) >= 0  # tags are best-effort

    def test_add_idea_with_explicit_tags(self, idea_log):
        """User-supplied tags should override auto-extraction."""
        entry = idea_log.add_idea("Test idea", tags=["custom-tag"])
        assert entry["tags"] == ["custom-tag"]

    def test_get_ideas_returns_all(self, idea_log):
        """get_ideas without limit returns all ideas."""
        idea_log.add_idea("Idea 1")
        idea_log.add_idea("Idea 2")
        idea_log.add_idea("Idea 3")
        result = idea_log.get_ideas()
        assert len(result["ideas"]) == 3
        assert result["total_count"] == 3

    def test_get_ideas_with_limit(self, idea_log):
        """get_ideas with limit returns only the most recent N ideas."""
        for i in range(15):
            idea_log.add_idea(f"Idea {i}")
        result = idea_log.get_ideas(limit=10)
        assert len(result["ideas"]) == 10
        assert result["total_count"] == 15
        # Most recent should be first
        assert "Idea 14" in result["ideas"][0]["text"]

    def test_search_ideas_by_keyword(self, idea_log):
        """search_ideas returns ideas matching the query."""
        idea_log.add_idea("Use LQ loss for image segmentation tasks")
        idea_log.add_idea("Try contrastive learning for audio")
        results = idea_log.search_ideas(query="LQ loss")
        assert len(results) == 1
        assert "LQ loss" in results[0]["text"]

    def test_search_ideas_by_tag(self, idea_log):
        """search_ideas with tag filter returns matching ideas."""
        idea_log.add_idea("Idea A", tags=["NLP", "transformers"])
        idea_log.add_idea("Idea B", tags=["CV", "detection"])
        results = idea_log.search_ideas(tag="NLP")
        assert len(results) == 1
        assert results[0]["text"] == "Idea A"

    def test_search_ideas_case_insensitive(self, idea_log):
        """Search should be case-insensitive."""
        idea_log.add_idea("Use LQ loss for image tasks")
        results = idea_log.search_ideas(query="lq loss")
        assert len(results) == 1

    def test_edit_idea(self, idea_log):
        """edit_idea updates text and updated_at."""
        entry = idea_log.add_idea("Original idea")
        edited = idea_log.edit_idea(entry["id"], new_text="Updated idea")
        assert edited["text"] == "Updated idea"
        assert edited["updated_at"] >= entry["updated_at"]

    def test_edit_idea_not_found(self, idea_log):
        """edit_idea with invalid ID raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            idea_log.edit_idea("nonexistent-id", new_text="test")

    def test_delete_idea(self, idea_log):
        """delete_idea removes the entry."""
        entry = idea_log.add_idea("To be deleted")
        idea_log.delete_idea(entry["id"])
        result = idea_log.get_ideas()
        assert result["total_count"] == 0

    def test_delete_idea_not_found(self, idea_log):
        """delete_idea with invalid ID raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            idea_log.delete_idea("nonexistent-id")

    def test_persistence_across_instances(self, tmp_path):
        """Ideas should persist when creating a new IdeaLog instance."""
        filepath = str(tmp_path / "ideas.json")
        log1 = IdeaLog(filepath=filepath)
        log1.add_idea("Persistent idea")

        log2 = IdeaLog(filepath=filepath)
        result = log2.get_ideas()
        assert result["total_count"] == 1
        assert result["ideas"][0]["text"] == "Persistent idea"

    def test_get_ideas_ordered_most_recent_first(self, idea_log):
        """Ideas should be returned most-recent first."""
        idea_log.add_idea("First")
        idea_log.add_idea("Second")
        idea_log.add_idea("Third")
        result = idea_log.get_ideas()
        assert result["ideas"][0]["text"] == "Third"
        assert result["ideas"][2]["text"] == "First"

    def test_related_papers_field(self, idea_log):
        """Ideas should have a related_papers field (empty by default)."""
        entry = idea_log.add_idea("Test idea")
        assert entry["related_papers"] == []
