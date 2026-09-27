"""
A small, persistent vector store built on FAISS.

Stores:
  - index.faiss   -> the FAISS similarity index (inner product on
                      normalized vectors == cosine similarity)
  - metadata.json -> parallel list of {document_id, filename, chunk_index, text}
                      so we can map a FAISS row back to its source chunk
"""
import json
import threading
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int, persist_dir: Path):
        self.dimension = dimension
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_dir / "index.faiss"
        self.meta_path = self.persist_dir / "metadata.json"

        self._lock = threading.Lock()
        self.metadata: List[Dict[str, Any]] = []
        self.index = faiss.IndexFlatIP(dimension)  # cosine similarity via normalized vectors

        self._load()

    # ---------- persistence ----------

    def _load(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    # ---------- writes ----------

    def add_document(
        self,
        filename: str,
        chunks: List[str],
        embeddings: np.ndarray,
        document_id: Optional[str] = None,
    ) -> str:
        """Adds all chunks of one document to the index. Returns the document_id."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")

        document_id = document_id or str(uuid.uuid4())[:8]

        with self._lock:
            self.index.add(embeddings)
            for i, chunk in enumerate(chunks):
                self.metadata.append(
                    {
                        "document_id": document_id,
                        "filename": filename,
                        "chunk_index": i,
                        "text": chunk,
                    }
                )
            self._save()

        return document_id

    def delete_document(self, document_id: str) -> int:
        """
        Removes a document by rebuilding the index without its vectors.
        FAISS's flat index doesn't support in-place deletion, so we rebuild.
        Returns the number of chunks removed.
        """
        with self._lock:
            keep_idx = [i for i, m in enumerate(self.metadata) if m["document_id"] != document_id]
            removed = len(self.metadata) - len(keep_idx)
            if removed == 0:
                return 0

            if keep_idx:
                all_vectors = self.index.reconstruct_n(0, self.index.ntotal)
                kept_vectors = all_vectors[keep_idx]
            else:
                kept_vectors = np.zeros((0, self.dimension), dtype="float32")

            new_index = faiss.IndexFlatIP(self.dimension)
            if kept_vectors.shape[0] > 0:
                new_index.add(kept_vectors)

            self.index = new_index
            self.metadata = [self.metadata[i] for i in keep_idx]
            self._save()

        return removed

    # ---------- reads ----------

    def search(
        self, query_embedding: np.ndarray, top_k: int = 4, document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        # Over-fetch when filtering by document_id, since FAISS doesn't
        # support filtered search natively in a flat index.
        fetch_k = top_k * 5 if document_id else top_k
        fetch_k = min(fetch_k, self.index.ntotal)

        query = query_embedding.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if document_id and meta["document_id"] != document_id:
                continue
            # row_index lets callers (e.g. the hybrid retriever) line this hit
            # back up with the same row in a parallel keyword-search index.
            results.append({**meta, "score": float(score), "row_index": int(idx)})
            if len(results) >= top_k:
                break

        return results

    def list_documents(self) -> List[Dict[str, Any]]:
        counts: Dict[str, Dict[str, Any]] = {}
        for m in self.metadata:
            doc_id = m["document_id"]
            if doc_id not in counts:
                counts[doc_id] = {"document_id": doc_id, "filename": m["filename"], "chunks": 0}
            counts[doc_id]["chunks"] += 1
        return list(counts.values())

    @property
    def total_chunks(self) -> int:
        return len(self.metadata)
