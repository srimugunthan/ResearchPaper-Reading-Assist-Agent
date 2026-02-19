"""LLM factory — supports Gemini (cloud), Ollama (local), and fake (testing)."""
from typing import List, Optional


def get_llm(
    provider: str = "gemini",
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    fake_responses: Optional[List[str]] = None,
):
    """Create a chat model instance.

    Args:
        provider: One of "gemini", "ollama", "fake".
        model_name: Model name (provider-specific defaults apply).
        temperature: Sampling temperature.
        fake_responses: List of canned responses for fake provider.

    Returns:
        A LangChain chat model instance.
    """
    if provider == "fake":
        from langchain_core.language_models import FakeListChatModel
        responses = fake_responses or ["This is a mock response."]
        return FakeListChatModel(responses=responses)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-2.0-flash",
            temperature=temperature,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name or "llama3",
            temperature=temperature,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
