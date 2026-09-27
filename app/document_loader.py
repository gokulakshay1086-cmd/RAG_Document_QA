"""
Loads raw text out of uploaded documents (PDF, DOCX, TXT, MD).
"""
from pathlib import Path

from pypdf import PdfReader
import docx


class UnsupportedFileTypeError(Exception):
    pass


def load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
    return "\n\n".join(pages)


def load_docx(path: Path) -> str:
    document = docx.Document(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
    ".md": load_txt,
}


def load_document(path: Path) -> str:
    """Dispatch to the correct loader based on file extension."""
    suffix = path.suffix.lower()
    if suffix not in LOADERS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{suffix}'. Supported: {list(LOADERS.keys())}"
        )
    text = LOADERS[suffix](path)
    if not text or not text.strip():
        raise ValueError("No extractable text found in document (it may be a scanned image PDF).")
    return text
