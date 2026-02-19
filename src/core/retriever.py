"""Retriever wrappers for vector store search."""
from typing import List

from langchain_core.documents import Document


def create_retriever(vectorstore, k: int = 5):
    """Create a retriever from a vector store.

    Args:
        vectorstore: A LangChain vector store instance.
        k: Number of documents to retrieve per query.

    Returns:
        A retriever instance.
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve(retriever, query: str) -> List[Document]:
    """Retrieve documents matching a query.

    Args:
        retriever: A LangChain retriever instance.
        query: The search query.

    Returns:
        List of matching Document objects.
    """
    return retriever.invoke(query)
