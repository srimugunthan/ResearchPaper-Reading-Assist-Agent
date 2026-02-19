"""Idea Log — JSON-backed persistent storage for research ideas."""
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class IdeaLog:
    """Persistent idea log backed by a JSON file."""

    def __init__(self, filepath: str = "idea_log.json"):
        self.filepath = filepath
        self._ideas: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load ideas from the JSON file if it exists."""
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                self._ideas = json.load(f)

    def _save(self) -> None:
        """Persist ideas to the JSON file."""
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(self._ideas, f, indent=2)

    def _extract_tags(self, text: str) -> List[str]:
        """Best-effort tag extraction from idea text."""
        stopwords = {
            "the", "is", "are", "was", "were", "a", "an", "and", "or", "but",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "only",
            "this", "that", "it", "not", "can", "has", "have", "had", "be",
            "use", "try", "using", "about", "how", "what", "why", "when",
        }
        parts = re.split(r"[,.]", text)
        tags = []
        for part in parts:
            words = part.strip().split()
            meaningful = [w for w in words if w.lower() not in stopwords and len(w) > 1]
            if meaningful:
                tag = " ".join(meaningful)
                if len(tag) > 2 and len(tag) < 50:
                    tags.append(tag)
        return tags[:5]

    def add_idea(
        self,
        text: str,
        tags: Optional[List[str]] = None,
        related_papers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Add a new idea entry.

        Args:
            text: The idea content.
            tags: Optional explicit tags. If None, auto-extracted.
            related_papers: Optional list of related paper IDs.

        Returns:
            The created idea entry dict.
        """
        now = datetime.now(timezone.utc).isoformat()
        entry = {
            "id": str(uuid.uuid4()),
            "text": text,
            "tags": tags if tags is not None else self._extract_tags(text),
            "related_papers": related_papers or [],
            "created_at": now,
            "updated_at": now,
        }
        self._ideas.append(entry)
        self._save()
        return entry

    def get_ideas(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get ideas, most recent first.

        Args:
            limit: Max number of ideas to return. None = all.

        Returns:
            Dict with 'ideas' (list) and 'total_count' (int).
        """
        total = len(self._ideas)
        sorted_ideas = list(reversed(self._ideas))
        if limit is not None:
            sorted_ideas = sorted_ideas[:limit]
        return {"ideas": sorted_ideas, "total_count": total}

    def search_ideas(
        self,
        query: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search ideas by keyword and/or tag.

        Args:
            query: Keyword to search in idea text (case-insensitive).
            tag: Tag to filter by (case-insensitive).

        Returns:
            List of matching idea entries.
        """
        results = self._ideas
        if query:
            q = query.lower()
            results = [i for i in results if q in i["text"].lower()]
        if tag:
            t = tag.lower()
            results = [
                i for i in results
                if any(t in tg.lower() for tg in i.get("tags", []))
            ]
        return results

    def edit_idea(self, idea_id: str, new_text: str) -> Dict[str, Any]:
        """Edit an idea's text.

        Args:
            idea_id: The idea ID.
            new_text: New text content.

        Returns:
            The updated idea entry.

        Raises:
            ValueError: If idea_id not found.
        """
        for idea in self._ideas:
            if idea["id"] == idea_id:
                idea["text"] = new_text
                idea["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save()
                return idea
        raise ValueError(f"Idea with id '{idea_id}' not found")

    def delete_idea(self, idea_id: str) -> None:
        """Delete an idea by ID.

        Raises:
            ValueError: If idea_id not found.
        """
        for i, idea in enumerate(self._ideas):
            if idea["id"] == idea_id:
                self._ideas.pop(i)
                self._save()
                return
        raise ValueError(f"Idea with id '{idea_id}' not found")
