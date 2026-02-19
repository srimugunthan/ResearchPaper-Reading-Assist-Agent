"""Conversation memory — JSON-backed persistent storage."""
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class ConversationMemory:
    """Persistent conversation memory backed by a JSON file."""

    def __init__(self, filepath: str = "conversation_memory.json"):
        self.filepath = filepath
        self._history: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        """Load history from the JSON file if it exists."""
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                self._history = json.load(f)

    def _save(self) -> None:
        """Persist history to the JSON file."""
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(self._history, f, indent=2)

    def save_exchange(self, user_input: str, assistant_output: str) -> None:
        """Save a conversation exchange.

        Args:
            user_input: The user's message.
            assistant_output: The assistant's response.
        """
        self._history.append({
            "input": user_input,
            "output": assistant_output,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        self._save()

    def get_history(self) -> List[Dict[str, Any]]:
        """Get all conversation history."""
        return list(self._history)

    def get_recent_context(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent N exchanges.

        Args:
            n: Number of recent exchanges to return.

        Returns:
            List of the most recent exchanges.
        """
        return self._history[-n:] if self._history else []

    def format_context(self, n: int = 5) -> str:
        """Format recent exchanges as a readable context string.

        Args:
            n: Number of recent exchanges to include.

        Returns:
            Formatted string of recent conversation.
        """
        recent = self.get_recent_context(n)
        if not recent:
            return ""
        parts = []
        for ex in recent:
            parts.append(f"User: {ex['input']}")
            parts.append(f"Assistant: {ex['output']}")
        return "\n".join(parts)

    def clear(self) -> None:
        """Clear all conversation memory."""
        self._history = []
        self._save()
