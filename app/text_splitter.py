"""
Splits long documents into overlapping chunks suitable for embedding.

Uses a simple recursive-ish strategy: split on paragraphs first, then
fall back to hard character windows for oversized paragraphs. Overlap
between consecutive chunks preserves context across chunk boundaries.
"""
from typing import List


def _split_paragraphs(text: str) -> List[str]:
    raw = [p.strip() for p in text.split("\n\n")]
    return [p for p in raw if p]


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    """
    Returns a list of text chunks, each roughly `chunk_size` characters,
    with `chunk_overlap` characters of overlap between consecutive chunks.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    paragraphs = _split_paragraphs(text)
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        # If a single paragraph is itself too big, hard-split it.
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                start = end - chunk_overlap
            continue

        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            chunks.append(current)
            # start new chunk, carrying overlap from the tail of the previous one
            overlap_text = current[-chunk_overlap:] if current else ""
            current = f"{overlap_text}\n\n{para}".strip()

    if current:
        chunks.append(current)

    return [c.strip() for c in chunks if c.strip()]
