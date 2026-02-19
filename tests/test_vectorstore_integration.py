"""Tests for vector store integration with Chroma."""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from src.core.vectorstore import get_vectorstore, store_documents, query_store


class FakeEmbeddings:
    """Fake embeddings that return deterministic vectors."""

    def embed_documents(self, texts):
        return [[float(i)] * 384 for i, _ in enumerate(texts)]

    def embed_query(self, text):
        return [0.5] * 384


class TestGetVectorStore:
    """Tests for vectorstore creation."""

    def test_creates_chroma_instance(self, tmp_path):
        """get_vectorstore should return a Chroma instance."""
        store = get_vectorstore(
            persist_directory=str(tmp_path / "chroma_db"),
            embedding_function=FakeEmbeddings(),
        )
        assert store is not None


class TestStoreDocuments:
    """Tests for storing documents in the vector store."""

    def test_stores_chunks_and_retrieves_them(self, tmp_path):
        """Stored documents should be retrievable."""
        embeddings = FakeEmbeddings()
        store = get_vectorstore(
            persist_directory=str(tmp_path / "chroma_db"),
            embedding_function=embeddings,
        )

        docs = [
            Document(
                page_content=f"Chunk {i} about transformers",
                metadata={"source": "paper.pdf", "title": "Transformers", "page": i},
            )
            for i in range(5)
        ]

        store_documents(store, docs)

        # Query and verify
        results = query_store(store, "transformers", k=5)
        assert len(results) >= 1

    def test_stored_documents_have_metadata(self, tmp_path):
        """Stored documents should preserve metadata fields."""
        embeddings = FakeEmbeddings()
        store = get_vectorstore(
            persist_directory=str(tmp_path / "chroma_db"),
            embedding_function=embeddings,
        )

        docs = [
            Document(
                page_content="Self-attention mechanism explained.",
                metadata={"source": "attention.pdf", "title": "Attention", "page": 3},
            )
        ]
        store_documents(store, docs)

        results = query_store(store, "attention", k=1)
        assert len(results) == 1
        meta = results[0].metadata
        assert meta["source"] == "attention.pdf"
        assert meta["title"] == "Attention"
        assert meta["page"] == 3

    def test_duplicate_documents_not_stored_twice(self, tmp_path):
        """Storing the same document twice should not create duplicates."""
        embeddings = FakeEmbeddings()
        store = get_vectorstore(
            persist_directory=str(tmp_path / "chroma_db"),
            embedding_function=embeddings,
        )

        doc = Document(
            page_content="Unique content about BERT.",
            metadata={"source": "bert.pdf", "title": "BERT", "page": 0},
        )
        store_documents(store, [doc])
        store_documents(store, [doc])

        results = query_store(store, "BERT", k=10)
        # Should only have 1 document, not 2
        contents = [r.page_content for r in results]
        assert contents.count("Unique content about BERT.") == 1

    def test_empty_docs_list(self, tmp_path):
        """Storing an empty list should not fail."""
        embeddings = FakeEmbeddings()
        store = get_vectorstore(
            persist_directory=str(tmp_path / "chroma_db"),
            embedding_function=embeddings,
        )
        store_documents(store, [])  # Should not raise


class TestGetEmbeddings:
    """Tests for embeddings factory."""

    def test_get_embeddings_returns_callable(self):
        """Embeddings factory should return an embeddings object."""
        from src.core.embeddings import get_embeddings
        emb = get_embeddings(provider="fake")
        assert hasattr(emb, "embed_documents")
        assert hasattr(emb, "embed_query")
