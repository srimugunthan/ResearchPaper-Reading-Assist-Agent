"""Embeddings factory — supports sentence-transformers, fake (for testing)."""
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Deterministic fake embeddings for testing."""

    def embed_documents(self, texts):
        return [[float(i % 10) * 0.1] * 384 for i, _ in enumerate(texts)]

    def embed_query(self, text):
        return [0.5] * 384


def get_embeddings(provider: str = "sentence-transformers", model_name: str = "all-MiniLM-L6-v2") -> Embeddings:
    """Create an embeddings instance.

    Args:
        provider: One of "sentence-transformers", "fake".
        model_name: Model name for sentence-transformers.

    Returns:
        An Embeddings instance.
    """
    if provider == "fake":
        return FakeEmbeddings()

    if provider == "sentence-transformers":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model_name)

    raise ValueError(f"Unknown embeddings provider: {provider}")
