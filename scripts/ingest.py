"""
CLI utility to batch-ingest all documents in a folder directly into the
vector store, without going through the API.

Usage:
    python scripts/ingest.py ./my_documents_folder
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.rag_pipeline import get_pipeline  # noqa: E402
from app.document_loader import LOADERS  # noqa: E402


def main(folder: str):
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"Error: {folder} is not a directory.")
        sys.exit(1)

    pipeline = get_pipeline()
    files = [f for f in folder_path.iterdir() if f.suffix.lower() in LOADERS]

    if not files:
        print(f"No supported files ({list(LOADERS.keys())}) found in {folder}.")
        return

    print(f"Found {len(files)} file(s) to ingest.\n")
    for f in files:
        try:
            result = pipeline.ingest_file(f, original_filename=f.name)
            print(f"[OK] {f.name} -> {result.chunks_indexed} chunks (id={result.document_id})")
        except Exception as e:
            print(f"[FAIL] {f.name}: {e}")

    print(f"\nDone. Vector store now has {pipeline.store.total_chunks} total chunks.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/ingest.py <folder_path>")
        sys.exit(1)
    main(sys.argv[1])
