"""Retriever wrappers for vector store search with hybrid BM25 support and metadata filtering."""
from typing import Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers.bm25 import BM25Retriever


# Maps query intent keywords to chunk_types that should be prioritized.
SECTION_PRIORITY_KEYWORDS: Dict[str, List[str]] = {
    "abstract": [
        "main idea", "summary", "overview", "contribution", "key finding",
        "hypothesis", "what is this paper about", "tldr", "gist",
    ],
    "methods": [
        "method", "approach", "algorithm", "technique", "implementation",
        "how did they", "experimental setup", "procedure",
    ],
    "results": [
        "result", "finding", "performance", "accuracy", "evaluation",
        "outcome", "experiment",
    ],
    "conclusion": [
        "conclusion", "future work", "limitation", "implication",
    ],
}


def detect_section_filter(query: str) -> Optional[List[str]]:
    """Detect if a query should prioritize specific paper sections.

    Args:
        query: The user's search query.

    Returns:
        List of chunk_types to filter for, or None if no filtering needed.
    """
    query_lower = query.lower()
    for chunk_type, keywords in SECTION_PRIORITY_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                if chunk_type == "abstract":
                    return ["abstract", "introduction"]
                return [chunk_type]
    return None


class SimpleEnsembleRetriever(BaseRetriever):
    """Simple ensemble retriever combining multiple retrievers with weighted ranking."""

    retrievers: List[BaseRetriever]
    weights: List[float]

    def __init__(self, retrievers: List[BaseRetriever], weights: List[float]):
        """Initialize with retrievers and weights.

        Args:
            retrievers: List of BaseRetriever instances
            weights: List of floats summing to 1.0 for weighting each retriever
        """
        if len(retrievers) != len(weights):
            raise ValueError("Number of retrievers must match number of weights")
        if not (0.99 <= sum(weights) <= 1.01):  # Allow small floating point error
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

        super().__init__(
            retrievers=retrievers,
            weights=weights,
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents from all retrievers and combine with weighted ranking.

        Args:
            query: The search query

        Returns:
            List of combined and deduplicated documents
        """
        # Retrieve from all retrievers
        all_docs = {}  # {doc_id: (doc, total_score)}

        for retriever, weight in zip(self.retrievers, self.weights):
            docs = retriever.invoke(query)
            for idx, doc in enumerate(docs):
                # Score based on rank position and weight
                score = weight * (1.0 / (1.0 + idx))  # Higher score for earlier positions

                doc_id = doc.metadata.get("source", "") + doc.page_content[:50]
                if doc_id in all_docs:
                    all_docs[doc_id] = (doc, all_docs[doc_id][1] + score)
                else:
                    all_docs[doc_id] = (doc, score)

        # Sort by combined score and return documents
        sorted_docs = sorted(all_docs.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs]


def create_retriever(
    vectorstore,
    documents: Optional[List[Document]] = None,
    k: int = 5,
    use_hybrid: bool = True,
    metadata_filter: Optional[dict] = None,
):
    """Create a retriever from a vector store with optional BM25 hybrid search.

    Args:
        vectorstore: A LangChain vector store instance.
        documents: List of Document objects for BM25 indexing (optional if use_hybrid=True).
        k: Number of documents to retrieve per query.
        use_hybrid: If True, attempt to use SimpleEnsembleRetriever combining vector + BM25.
                   If documents are not provided, gracefully falls back to vector-only search.
        metadata_filter: Optional Chroma metadata filter dict, e.g. {"chunk_type": {"$in": ["abstract"]}}.

    Returns:
        A retriever instance (SimpleEnsembleRetriever if use_hybrid=True and documents available,
        else vector retriever).
    """
    # Vector retriever (dense search)
    search_kwargs = {"k": k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    vector_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    if not use_hybrid:
        return vector_retriever

    # BM25 retriever (sparse keyword search)
    # Gracefully fall back to vector-only if documents aren't available
    if documents is None:
        import warnings
        warnings.warn(
            "Hybrid retrieval requested but documents not provided. "
            "Falling back to vector-only retrieval. "
            "To enable BM25 hybrid search, pass documents list.",
            UserWarning
        )
        return vector_retriever

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    # Ensemble with 60% vector, 40% BM25 weighting
    ensemble = SimpleEnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    return ensemble


def create_filtered_retriever(
    vectorstore,
    query: str,
    documents: Optional[List[Document]] = None,
    k: int = 5,
    use_hybrid: bool = True,
):
    """Create a retriever with automatic section filtering based on query intent.

    Analyzes the query to detect if specific paper sections should be prioritized
    (e.g., "main idea" queries prioritize abstract/introduction chunks).

    Args:
        vectorstore: A LangChain vector store instance.
        query: The user's search query (used for section filter detection).
        documents: List of Document objects for BM25 indexing.
        k: Number of documents to retrieve per query.
        use_hybrid: If True, attempt hybrid BM25+vector retrieval.

    Returns:
        A retriever instance with appropriate metadata filtering.
    """
    section_types = detect_section_filter(query)
    metadata_filter = None
    if section_types:
        metadata_filter = {"chunk_type": {"$in": section_types}}

    return create_retriever(
        vectorstore=vectorstore,
        documents=documents,
        k=k,
        use_hybrid=use_hybrid,
        metadata_filter=metadata_filter,
    )


def retrieve(retriever, query: str) -> List[Document]:
    """Retrieve documents matching a query.

    Args:
        retriever: A LangChain retriever instance.
        query: The search query.

    Returns:
        List of matching Document objects.
    """
    return retriever.invoke(query)
