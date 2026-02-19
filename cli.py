"""CLI entry point for the ResearchPaper-reading-Assist Agent ingestion pipeline."""
import argparse
import logging
import sys

from dotenv import load_dotenv
load_dotenv()

from src.core.embeddings import get_embeddings
from src.ingestion.pipeline import ingest_folder

DEFAULT_PERSIST_DIR = "./chroma_db"
DEFAULT_EMBEDDING_PROVIDER = "sentence-transformers"


def main():
    parser = argparse.ArgumentParser(
        description="Ingest research paper PDFs into the vector store."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to folder containing PDF research papers.",
    )
    parser.add_argument(
        "--persist-dir",
        default=DEFAULT_PERSIST_DIR,
        help=f"Path to persist Chroma DB (default: {DEFAULT_PERSIST_DIR}).",
    )
    parser.add_argument(
        "--embedding-provider",
        default=DEFAULT_EMBEDDING_PROVIDER,
        choices=["sentence-transformers", "fake"],
        help=f"Embedding provider (default: {DEFAULT_EMBEDDING_PROVIDER}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for text splitting (default: 1000).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for text splitting (default: 200).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info(f"Starting ingestion from: {args.folder}")
    logging.info(f"Persist directory: {args.persist_dir}")
    logging.info(f"Embedding provider: {args.embedding_provider}")

    try:
        embedding_function = get_embeddings(provider=args.embedding_provider)
    except Exception as e:
        logging.error(f"Failed to initialize embeddings: {e}")
        sys.exit(1)

    try:
        result = ingest_folder(
            folder_path=args.folder,
            persist_directory=args.persist_dir,
            embedding_function=embedding_function,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    logging.info("=== Ingestion Complete ===")
    logging.info(f"Total files found: {result['total_files']}")
    logging.info(f"Files processed:   {result['files_processed']}")
    logging.info(f"Files skipped:     {result['files_skipped']}")
    logging.info(f"Files failed:      {result['files_failed']}")
    logging.info(f"Total chunks:      {result['total_chunks']}")

    if result["errors"]:
        logging.warning("Errors encountered:")
        for err in result["errors"]:
            logging.warning(f"  {err['file']}: {err['error']}")

    sys.exit(0 if result["files_failed"] == 0 else 1)


if __name__ == "__main__":
    main()
