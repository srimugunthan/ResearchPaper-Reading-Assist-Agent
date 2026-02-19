"""Ingestion pipeline: scan -> load -> split -> embed -> store."""
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.ingestion.scanner import scan_folder
from src.ingestion.loader import load_pdf
from src.ingestion.splitter import split_text
from src.core.vectorstore import get_vectorstore, store_documents

logger = logging.getLogger(__name__)

# File to track ingested file hashes for deduplication
INGESTED_REGISTRY_FILE = "ingested_files.json"


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file's contents."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _load_registry(persist_directory: str) -> Dict[str, str]:
    """Load the ingested files registry from disk."""
    registry_path = os.path.join(persist_directory, INGESTED_REGISTRY_FILE)
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            return json.load(f)
    return {}


def _save_registry(persist_directory: str, registry: Dict[str, str]):
    """Save the ingested files registry to disk."""
    os.makedirs(persist_directory, exist_ok=True)
    registry_path = os.path.join(persist_directory, INGESTED_REGISTRY_FILE)
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)


def ingest_pdf(pdf_path: str, store, chunk_size: int = 1000, chunk_overlap: int = 200) -> int:
    """Ingest a single PDF: load -> split -> store.

    Args:
        pdf_path: Path to the PDF file.
        store: Chroma vector store instance.
        chunk_size: Chunk size for text splitting.
        chunk_overlap: Overlap between chunks.

    Returns:
        Number of chunks stored.
    """
    docs = load_pdf(pdf_path)
    if not docs:
        return 0

    all_chunks = []
    for doc in docs:
        chunks = split_text(doc.page_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for chunk_text in chunks:
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={**doc.metadata},
            )
            all_chunks.append(chunk_doc)

    count = store_documents(store, all_chunks)
    return count


def ingest_folder(
    folder_path: str,
    persist_directory: str,
    embedding_function: Embeddings,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    collection_name: str = "research_papers",
) -> Dict[str, Any]:
    """Ingest all PDFs from a folder into the vector store.

    Args:
        folder_path: Path to folder containing PDFs.
        persist_directory: Path to persist Chroma DB.
        embedding_function: Embeddings instance.
        chunk_size: Chunk size for text splitting.
        chunk_overlap: Overlap between chunks.
        collection_name: Chroma collection name.

    Returns:
        Dict with ingestion stats: total_files, files_processed,
        files_skipped, files_failed, errors.
    """
    pdf_paths = scan_folder(folder_path)
    registry = _load_registry(persist_directory)

    store = get_vectorstore(
        persist_directory=persist_directory,
        embedding_function=embedding_function,
        collection_name=collection_name,
    )

    stats = {
        "total_files": len(pdf_paths),
        "files_processed": 0,
        "files_skipped": 0,
        "files_failed": 0,
        "errors": [],
        "total_chunks": 0,
    }

    for pdf_path in pdf_paths:
        file_hash = compute_file_hash(pdf_path)
        registry_key = f"{pdf_path}:{file_hash}"

        if registry_key in registry:
            logger.info(f"Skipping already-ingested: {pdf_path}")
            stats["files_skipped"] += 1
            continue

        start = time.time()
        try:
            chunks_added = ingest_pdf(pdf_path, store, chunk_size, chunk_overlap)
            elapsed = time.time() - start
            logger.info(
                f"Ingested {os.path.basename(pdf_path)}: "
                f"{chunks_added} chunks in {elapsed:.1f}s"
            )
            registry[registry_key] = file_hash
            stats["files_processed"] += 1
            stats["total_chunks"] += chunks_added
        except Exception as e:
            logger.error(f"Failed to ingest {pdf_path}: {e}")
            stats["files_failed"] += 1
            stats["errors"].append({"file": pdf_path, "error": str(e)})

    _save_registry(persist_directory, registry)
    return stats
