"""PDF loader with metadata extraction."""
import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(pdf_path: str) -> List[Document]:
    """Load a PDF file and return documents with enriched metadata.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        List of Document objects with page content and metadata
        (source, title, page, total_pages).

    Raises:
        ValueError: If the PDF cannot be read.
    """
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
    except Exception as e:
        raise ValueError(f"Failed to load PDF: {pdf_path} — {e}") from e

    if not docs:
        return []

    total_pages = len(docs)
    title = os.path.splitext(os.path.basename(pdf_path))[0]

    for doc in docs:
        doc.metadata["title"] = title
        doc.metadata["total_pages"] = total_pages
        doc.metadata.setdefault("source", pdf_path)
        doc.metadata.setdefault("page", 0)

    return docs
