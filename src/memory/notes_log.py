"""Notes & Insights log — JSON-backed persistent storage."""
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class NotesLog:
    """Persistent notes & insights log backed by a JSON file."""

    def __init__(self, filepath: str = "notes_and_insights.json"):
        self.filepath = filepath
        self._notes: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load notes from the JSON file if it exists."""
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                self._notes = json.load(f)

    def _save(self) -> None:
        """Persist notes to the JSON file."""
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(self._notes, f, indent=2)

    def _extract_tags(self, text: str) -> List[str]:
        """Best-effort tag extraction from note text."""
        # Split on common delimiters and extract multi-word phrases
        # that look like technical terms (2+ chars, not stopwords)
        stopwords = {
            "the", "is", "are", "was", "were", "a", "an", "and", "or", "but",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "only",
            "this", "that", "it", "not", "can", "has", "have", "had", "be",
        }
        # Split on commas and periods to find phrases
        parts = re.split(r"[,.]", text)
        tags = []
        for part in parts:
            words = part.strip().split()
            # Filter out very short or stopword-only segments
            meaningful = [w for w in words if w.lower() not in stopwords and len(w) > 1]
            if meaningful:
                tag = " ".join(meaningful)
                if len(tag) > 2 and len(tag) < 50:
                    tags.append(tag)
        return tags[:5]  # Cap at 5 tags

    def add_note(self, text: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Add a new note entry.

        Args:
            text: The note content.
            tags: Optional explicit tags. If None, auto-extracted from text.

        Returns:
            The created note entry dict.
        """
        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "id": str(uuid.uuid4()),
            "text": text,
            "tags": tags if tags is not None else self._extract_tags(text),
            "created_at": now,
            "updated_at": now,
        }
        self._notes.append(entry)
        self._save()
        return entry

    def get_notes(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get notes, most recent first.

        Args:
            limit: Max number of notes to return. None = all.

        Returns:
            Dict with 'notes' (list) and 'total_count' (int).
        """
        total = len(self._notes)
        # Return most recent first
        sorted_notes = list(reversed(self._notes))
        if limit is not None:
            sorted_notes = sorted_notes[:limit]
        return {"notes": sorted_notes, "total_count": total}

    def search_notes(self, query: str) -> List[Dict[str, Any]]:
        """Search notes by keyword (case-insensitive).

        Args:
            query: Search term.

        Returns:
            List of matching note entries.
        """
        q = query.lower()
        return [n for n in self._notes if q in n["text"].lower()]

    def edit_note(self, note_id: str, new_text: str) -> Dict[str, Any]:
        """Edit a note's text.

        Args:
            note_id: The note ID.
            new_text: New text content.

        Returns:
            The updated note entry.

        Raises:
            ValueError: If note_id not found.
        """
        for note in self._notes:
            if note["id"] == note_id:
                note["text"] = new_text
                note["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save()
                return note
        raise ValueError(f"Note with id '{note_id}' not found")

    def delete_note(self, note_id: str) -> None:
        """Delete a note by ID.

        Args:
            note_id: The note ID.

        Raises:
            ValueError: If note_id not found.
        """
        for i, note in enumerate(self._notes):
            if note["id"] == note_id:
                self._notes.pop(i)
                self._save()
                return
        raise ValueError(f"Note with id '{note_id}' not found")
