"""Ingestion: load documents from disk and split them into overlapping chunks."""
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md"}


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into chunks of `chunk_size` characters, where each chunk repeats
    the last `overlap` characters of the previous one, so an idea sitting on a
    boundary is not cut in half."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size, or the window never moves forward")
    chunks = []
    start = 0                              # left edge of the sliding window
    while start < len(text):
        end = start + chunk_size           # right edge = left edge + window size
        chunks.append(text[start:end])
        if end >= len(text):               # reached the end of the document
            break
        start = end - overlap              # step forward, keeping `overlap` characters of context
    return chunks


def load_chunks(docs_dir: Path, chunk_size: int = 500, overlap: int = 100) -> tuple[list[str], list[str]]:
    """Read every .txt / .md file in docs_dir, chunk it, and return two parallel
    lists: the chunks, and the filename each chunk came from.
    Files whose name starts with '_' are skipped (drafts, scratch)."""
    chunks, sources = [], []
    for path in sorted(Path(docs_dir).iterdir()):
        if path.suffix not in SUPPORTED_SUFFIXES or path.name.startswith("_"):
            continue
        for chunk in chunk_text(path.read_text(encoding="utf-8"), chunk_size, overlap):
            chunks.append(chunk)
            sources.append(path.name)
    return chunks, sources
