"""Text splitter for chunking PDF content with optional section-aware parsing."""
import re
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Regex patterns mapping section headers to chunk types.
# Each pattern matches the full line (short lines only, <=80 chars).
SECTION_PATTERNS = [
    (r'(?i)^abstract\s*$', 'abstract'),
    (r'(?i)^(?:\d+\.?\s*)?introduction\s*$', 'introduction'),
    (r'(?i)^(?:\d+\.?\s*)?(?:related\s+work|background|literature\s+review)\s*$', 'introduction'),
    (r'(?i)^(?:\d+\.?\s*)?(?:method(?:s|ology)?|approach|experimental\s+setup|proposed\s+method)\s*$', 'methods'),
    (r'(?i)^(?:\d+\.?\s*)?(?:results?\s+and\s+discussion|discussion)\s*$', 'results'),
    (r'(?i)^(?:\d+\.?\s*)?(?:results?|findings?|experiments?|evaluation)\s*$', 'results'),
    (r'(?i)^(?:\d+\.?\s*)?(?:conclusion|conclusions|summary|concluding\s+remarks)\s*$', 'conclusion'),
    (r'(?i)^(?:\d+\.?\s*)?(?:references|bibliography)\s*$', 'references'),
]


def get_text_splitter(
    chunk_size: int = 1000, chunk_overlap: int = 200
) -> RecursiveCharacterTextSplitter:
    """Create a configured RecursiveCharacterTextSplitter.

    Args:
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        Configured text splitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def split_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[str]:
    """Split text into overlapping chunks.

    Args:
        text: Input text to split.
        chunk_size: Maximum size of each chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def detect_sections(text: str) -> List[Tuple[str, str, int, int]]:
    """Detect academic paper sections in text by matching line-level regex patterns.

    Args:
        text: Full text of the paper.

    Returns:
        List of (section_name, chunk_type, start_char_idx, end_char_idx).
        Returns empty list if no sections detected.
    """
    lines = text.split('\n')
    sections: List[Tuple[str, str, int, int]] = []
    current_pos = 0

    for line in lines:
        stripped = line.strip()
        line_len = len(line) + 1  # +1 for the \n

        # Section headers are typically short
        if stripped and len(stripped) <= 80:
            for pattern, chunk_type in SECTION_PATTERNS:
                if re.match(pattern, stripped):
                    sections.append((stripped, chunk_type, current_pos, -1))
                    break

        current_pos += line_len

    # Fill in end positions: each section ends where the next begins
    for i in range(len(sections) - 1):
        sections[i] = (sections[i][0], sections[i][1], sections[i][2], sections[i + 1][2])
    if sections:
        sections[-1] = (sections[-1][0], sections[-1][1], sections[-1][2], len(text))

    return sections


def split_text_with_sections(
    text: str,
    metadata: Dict,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """Split text with section awareness. Falls back to flat chunking if no sections detected.

    Args:
        text: Full text content (may span multiple pages).
        metadata: Base metadata dict to attach to each chunk (source, title, etc.).
        chunk_size: Maximum chunk size.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of Document objects with chunk_type and section in metadata.
    """
    if not text:
        return []

    sections = detect_sections(text)
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not sections:
        # Graceful fallback: no sections detected
        chunks = splitter.split_text(text)
        return [
            Document(
                page_content=chunk,
                metadata={**metadata, "chunk_type": "general", "section": ""},
            )
            for chunk in chunks
        ]

    documents = []
    for section_name, chunk_type, start, end in sections:
        section_text = text[start:end].strip()
        if not section_text:
            continue
        chunks = splitter.split_text(section_text)
        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **metadata,
                        "chunk_type": chunk_type,
                        "section": section_name,
                    },
                )
            )

    return documents
