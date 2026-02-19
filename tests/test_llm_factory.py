"""Tests for LLM factory."""
import pytest
from unittest.mock import patch, MagicMock

from src.core.llm import get_llm


class TestGetLlm:
    """Tests for get_llm factory function."""

    def test_fake_provider_returns_callable(self):
        """Fake provider should return a working LLM without API keys."""
        llm = get_llm(provider="fake")
        assert llm is not None
        result = llm.invoke("Hello")
        assert result is not None

    def test_fake_provider_returns_configured_responses(self):
        """Fake provider should return responses from the provided list."""
        llm = get_llm(provider="fake", fake_responses=["Answer A", "Answer B"])
        result = llm.invoke("Question 1")
        assert "Answer A" in result.content

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_gemini_provider_creates_correct_model(self, mock_cls):
        """Gemini provider should instantiate ChatGoogleGenerativeAI."""
        mock_cls.return_value = MagicMock()
        llm = get_llm(provider="gemini", model_name="gemini-2.0-flash")
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs.get("model") == "gemini-2.0-flash"

    @patch("langchain_ollama.ChatOllama")
    def test_ollama_provider_creates_correct_model(self, mock_cls):
        """Ollama provider should instantiate ChatOllama."""
        mock_cls.return_value = MagicMock()
        llm = get_llm(provider="ollama", model_name="llama3")
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs.get("model") == "llama3"

    def test_unknown_provider_raises(self):
        """Unknown provider should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm(provider="nonexistent")

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_temperature_passed_to_provider(self, mock_cls):
        """Temperature should be configurable."""
        mock_cls.return_value = MagicMock()
        get_llm(provider="gemini", temperature=0.7)
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.7

    @patch("langchain_google_genai.ChatGoogleGenerativeAI")
    def test_default_provider_is_gemini(self, mock_cls):
        """Default provider should be gemini."""
        mock_cls.return_value = MagicMock()
        get_llm()
        mock_cls.assert_called_once()
