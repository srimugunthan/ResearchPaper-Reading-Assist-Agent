"""Metadata filter utilities for multi-paper synthesis."""
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


def build_title_filter(titles: List[str]) -> Optional[Dict[str, Any]]:
    """Build a Chroma where filter to match documents by title."""
    if not titles:
        return None
    if len(titles) == 1:
        return {"title": titles[0]}
    return {"title": {"$in": titles}}


def build_source_filter(sources: List[str]) -> Optional[Dict[str, Any]]:
    """Build a Chroma where filter to match documents by source path."""
    if not sources:
        return None
    if len(sources) == 1:
        return {"source": sources[0]}
    return {"source": {"$in": sources}}


def build_combined_filter(
    titles: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Build a combined Chroma where filter from multiple criteria."""
    filters = []
    title_f = build_title_filter(titles or [])
    if title_f:
        filters.append(title_f)
    source_f = build_source_filter(sources or [])
    if source_f:
        filters.append(source_f)

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def retrieve_by_filter(
    vectorstore,
    query: str,
    where: Optional[Dict[str, Any]] = None,
    k: int = 20,
) -> List[Document]:
    """Retrieve documents from the vector store with optional metadata filter."""
    return vectorstore.similarity_search(query, k=k, filter=where)


def get_unique_titles(vectorstore) -> List[str]:
    """Get all unique paper titles in the vector store."""
    result = vectorstore.get()
    titles = set()
    for meta in result.get("metadatas", []):
        if meta and "title" in meta:
            titles.add(meta["title"])
    return sorted(titles)
