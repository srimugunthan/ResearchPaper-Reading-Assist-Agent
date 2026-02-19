"""Recursively discover PDF files in a given folder."""
import os
from typing import List


def scan_folder(folder_path: str) -> List[str]:
    """Scan a folder recursively and return absolute paths of all PDF files.

    Args:
        folder_path: Path to the root folder to scan.

    Returns:
        List of absolute paths to .pdf files found.

    Raises:
        FileNotFoundError: If folder_path does not exist.
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    pdf_paths = []
    for root, _dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.startswith("."):
                continue
            if filename.lower().endswith(".pdf"):
                pdf_paths.append(os.path.abspath(os.path.join(root, filename)))

    return sorted(pdf_paths)
