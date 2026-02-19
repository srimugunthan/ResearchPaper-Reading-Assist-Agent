"""Chroma vector store integration and helper functions."""
import hashlib
from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma


def get_vectorstore(
    persist_directory: str,
    embedding_function,
    collection_name: str = "research_papers",
) -> Chroma:
    """Create or load a Chroma vector store.

    Args:
        persist_directory: Path to persist the Chroma DB.
        embedding_function: Embeddings instance.
        collection_name: Name of the Chroma collection.

    Returns:
        A Chroma vector store instance.
    """
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )


def _doc_id(doc: Document) -> str:
    """Generate a deterministic ID for a document based on content and metadata."""
    content = doc.page_content
    source = doc.metadata.get("source", "")
    page = str(doc.metadata.get("page", ""))
    raw = f"{source}:{page}:{content}"
    return hashlib.sha256(raw.encode()).hexdigest()


def store_documents(store: Chroma, documents: List[Document]) -> int:
    """Store documents in the vector store, skipping duplicates.

    Args:
        store: Chroma vector store instance.
        documents: List of Document objects to store.

    Returns:
        Number of new documents added.
    """
    if not documents:
        return 0

    # Generate IDs for deduplication
    ids = [_doc_id(doc) for doc in documents]

    # Check which IDs already exist
    existing = store.get(ids=ids)
    existing_ids = set(existing["ids"]) if existing["ids"] else set()

    # Filter to only new documents
    new_docs = []
    new_ids = []
    for doc, doc_id in zip(documents, ids):
        if doc_id not in existing_ids:
            new_docs.append(doc)
            new_ids.append(doc_id)

    if new_docs:
        store.add_documents(new_docs, ids=new_ids)

    return len(new_docs)


def query_store(
    store: Chroma, query: str, k: int = 5
) -> List[Document]:
    """Query the vector store for similar documents.

    Args:
        store: Chroma vector store instance.
        query: Search query text.
        k: Number of results to return.

    Returns:
        List of matching Document objects.
    """
    return store.similarity_search(query, k=k)
