"""PDF loader with metadata extraction."""
import os
import re
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def _looks_like_author_line(text: str) -> bool:
    """Check if a line looks like an author list rather than a title."""
    # Author lines typically contain: superscript markers (*, †, 1, 2),
    # multiple commas (listing authors), "and" between names, etc.
    if re.search(r'[*†‡§∗]', text):
        return True
    # Names followed by superscript numbers like "John Smith1"
    if re.search(r'[A-Z][a-z]+\s*\d', text):
        return True
    # "Published as" or "Accepted at" conference headers
    if re.match(r'(published|accepted|presented|submitted)\s', text.lower()):
        return True
    return False


def _extract_title_from_content(first_page_text: str) -> str:
    """Extract the paper title from the first page content.

    Takes only the first non-empty line (or two lines if the first is short),
    stopping before author names, affiliations, or section headers.
    """
    lines = first_page_text.strip().split("\n")
    title_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if title_lines:
                break
            continue
        lower = stripped.lower()
        # Stop at common post-title markers
        if any(lower.startswith(kw) for kw in [
            "abstract", "introduction", "keywords",
            "1 ", "1.", "author", "department", "university",
            "school of", "institute", "faculty",
        ]):
            break
        # Stop at emails, affiliations, or author-like lines
        if "@" in stripped or stripped.startswith("{"):
            break
        if _looks_like_author_line(stripped):
            break
        title_lines.append(stripped)
        # Titles are typically 1-2 lines, cap at 150 chars total
        combined = " ".join(title_lines)
        if len(combined) > 150 or len(title_lines) >= 2:
            break
    return " ".join(title_lines) if title_lines else ""


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
    filename_title = os.path.splitext(os.path.basename(pdf_path))[0]

    # Try to extract the real paper title from first page content
    title = _extract_title_from_content(docs[0].page_content) or filename_title

    for doc in docs:
        doc.metadata["title"] = title
        doc.metadata["total_pages"] = total_pages
        doc.metadata.setdefault("source", pdf_path)
        doc.metadata.setdefault("page", 0)

    return docs
