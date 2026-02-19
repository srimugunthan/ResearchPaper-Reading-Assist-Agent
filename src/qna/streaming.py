"""Streaming callback handler for token-by-token LLM output."""
from typing import Any, List

from langchain_core.callbacks import BaseCallbackHandler


class StreamingCollector(BaseCallbackHandler):
    """Collects streaming tokens from an LLM for later use (e.g., in Streamlit)."""

    def __init__(self):
        super().__init__()
        self.tokens: List[str] = []
        self.is_complete: bool = False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when the LLM emits a new token."""
        self.tokens.append(token)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Called when the LLM finishes generating."""
        self.is_complete = True

    def get_text(self) -> str:
        """Return all accumulated tokens as a single string."""
        return "".join(self.tokens)

    def reset(self) -> None:
        """Clear all state for reuse."""
        self.tokens = []
        self.is_complete = False
