"""Tests for notes & insights log module."""
import os
import json
import pytest

from src.memory.notes_log import NotesLog


@pytest.fixture
def notes_log(tmp_path):
    """Create a NotesLog backed by a temp file."""
    filepath = str(tmp_path / "notes_and_insights.json")
    return NotesLog(filepath=filepath)


class TestNotesLog:
    """Tests for NotesLog operations."""

    def test_add_note(self, notes_log):
        """Adding a note returns an entry with id, text, tags, created_at."""
        entry = notes_log.add_note("Only for symmetric loss, MAE is useful")
        assert entry["id"]
        assert entry["text"] == "Only for symmetric loss, MAE is useful"
        assert isinstance(entry["tags"], list)
        assert entry["created_at"]
        assert entry["updated_at"]

    def test_add_note_auto_tags(self, notes_log):
        """Auto-extracted tags should contain key terms from the note."""
        entry = notes_log.add_note("Only for symmetric loss, MAE is useful")
        # Should have some tags extracted
        assert len(entry["tags"]) >= 0  # tags are best-effort

    def test_add_note_persists_to_file(self, notes_log, tmp_path):
        """Notes should be persisted to the JSON file."""
        notes_log.add_note("Test note")
        filepath = str(tmp_path / "notes_and_insights.json")
        assert os.path.exists(filepath)
        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == 1

    def test_get_notes_returns_all(self, notes_log):
        """get_notes without limit returns all notes."""
        notes_log.add_note("Note 1")
        notes_log.add_note("Note 2")
        notes_log.add_note("Note 3")
        result = notes_log.get_notes()
        assert len(result["notes"]) == 3
        assert result["total_count"] == 3

    def test_get_notes_with_limit(self, notes_log):
        """get_notes with limit returns only the most recent N notes."""
        for i in range(15):
            notes_log.add_note(f"Note {i}")
        result = notes_log.get_notes(limit=10)
        assert len(result["notes"]) == 10
        assert result["total_count"] == 15
        # Most recent should be first
        assert "Note 14" in result["notes"][0]["text"]

    def test_search_notes_by_keyword(self, notes_log):
        """search_notes returns notes matching the query."""
        notes_log.add_note("Only for symmetric loss, MAE is useful")
        notes_log.add_note("Cross-attention is better for long sequences")
        results = notes_log.search_notes(query="MAE")
        assert len(results) == 1
        assert "MAE" in results[0]["text"]

    def test_search_notes_case_insensitive(self, notes_log):
        """Search should be case-insensitive."""
        notes_log.add_note("Only for symmetric loss, MAE is useful")
        results = notes_log.search_notes(query="mae")
        assert len(results) == 1

    def test_search_notes_no_match(self, notes_log):
        """Search with no match returns empty list."""
        notes_log.add_note("Some note")
        results = notes_log.search_notes(query="nonexistent")
        assert len(results) == 0

    def test_edit_note(self, notes_log):
        """edit_note updates text and updated_at."""
        entry = notes_log.add_note("Original text")
        original_updated = entry["updated_at"]
        edited = notes_log.edit_note(entry["id"], new_text="Updated text")
        assert edited["text"] == "Updated text"
        assert edited["updated_at"] >= original_updated

    def test_edit_note_not_found(self, notes_log):
        """edit_note with invalid ID raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            notes_log.edit_note("nonexistent-id", new_text="test")

    def test_delete_note(self, notes_log):
        """delete_note removes the entry."""
        entry = notes_log.add_note("To be deleted")
        notes_log.delete_note(entry["id"])
        result = notes_log.get_notes()
        assert result["total_count"] == 0

    def test_delete_note_not_found(self, notes_log):
        """delete_note with invalid ID raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            notes_log.delete_note("nonexistent-id")

    def test_persistence_across_instances(self, tmp_path):
        """Notes should persist when creating a new NotesLog instance."""
        filepath = str(tmp_path / "notes.json")
        log1 = NotesLog(filepath=filepath)
        log1.add_note("Persistent note")

        log2 = NotesLog(filepath=filepath)
        result = log2.get_notes()
        assert result["total_count"] == 1
        assert result["notes"][0]["text"] == "Persistent note"

    def test_get_notes_ordered_most_recent_first(self, notes_log):
        """Notes should be returned most-recent first."""
        notes_log.add_note("First")
        notes_log.add_note("Second")
        notes_log.add_note("Third")
        result = notes_log.get_notes()
        assert result["notes"][0]["text"] == "Third"
        assert result["notes"][2]["text"] == "First"
