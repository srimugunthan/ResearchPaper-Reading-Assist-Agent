"""Tests for intent detector module."""
import pytest

from src.memory.intent_detector import detect_intent


class TestDetectIntent:
    """Tests for detect_intent function."""

    def test_add_idea_intent(self):
        """'Add to idea log: ...' should return add_idea intent."""
        result = detect_intent("Add to idea log: use contrastive loss for audio")
        assert result["intent"] == "add_idea"
        assert result["extracted_text"] == "use contrastive loss for audio"

    def test_add_idea_intent_variation(self):
        """'I have this idea ... Add it in idea log' should return add_idea."""
        result = detect_intent(
            "I have this idea about using contrastive loss for audio. Add it in idea log"
        )
        assert result["intent"] == "add_idea"
        assert "contrastive loss" in result["extracted_text"]

    def test_add_note_intent(self):
        """'Add this to notes: ...' should return add_note intent."""
        result = detect_intent(
            "Add this to notes: 'Only for symmetric loss, MAE is useful'"
        )
        assert result["intent"] == "add_note"
        assert "Only for symmetric loss, MAE is useful" in result["extracted_text"]

    def test_add_note_intent_variation(self):
        """'Note this down: ...' should return add_note intent."""
        result = detect_intent("Note this down: cross-attention works better for long sequences")
        assert result["intent"] == "add_note"
        assert "cross-attention works better" in result["extracted_text"]

    def test_show_notes_intent(self):
        """'Show my notes' should return show_notes intent."""
        result = detect_intent("Show my notes")
        assert result["intent"] == "show_notes"

    def test_show_notes_intent_variation(self):
        """'Display my notes' should return show_notes intent."""
        result = detect_intent("Display my notes and insights")
        assert result["intent"] == "show_notes"

    def test_show_ideas_intent(self):
        """'Show my idea log' should return show_ideas intent."""
        result = detect_intent("Show my idea log")
        assert result["intent"] == "show_ideas"

    def test_show_ideas_intent_variation(self):
        """'Show my ideas' should return show_ideas intent."""
        result = detect_intent("Show my ideas")
        assert result["intent"] == "show_ideas"

    def test_search_notes_intent(self):
        """'Search my notes for MAE' should return search_notes intent."""
        result = detect_intent("Search my notes for MAE")
        assert result["intent"] == "search_notes"
        assert result["params"]["query"] == "MAE"

    def test_search_ideas_intent(self):
        """'Search my ideas for LQ loss' should return search_ideas intent."""
        result = detect_intent("Search my ideas for LQ loss")
        assert result["intent"] == "search_ideas"
        assert result["params"]["query"] == "LQ loss"

    def test_regular_question_intent(self):
        """A regular question should return question intent."""
        result = detect_intent("What is self-attention?")
        assert result["intent"] == "question"

    def test_regular_question_no_false_positive(self):
        """A question mentioning 'idea' should not trigger add_idea."""
        result = detect_intent("What is the main idea behind transformers?")
        assert result["intent"] == "question"

    def test_case_insensitive(self):
        """Intent detection should be case-insensitive."""
        result = detect_intent("ADD TO IDEA LOG: test idea")
        assert result["intent"] == "add_idea"
        assert result["extracted_text"] == "test idea"

    def test_result_structure(self):
        """All results should have intent, extracted_text, and params keys."""
        result = detect_intent("Show my notes")
        assert "intent" in result
        assert "extracted_text" in result
        assert "params" in result
