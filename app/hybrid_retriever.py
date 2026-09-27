"""
Hybrid retrieval: combines dense semantic search (FAISS/embeddings) with
sparse keyword search (BM25).

Why hybrid: pure embedding search is great at "meaning" but can miss exact
terms it's never seen emphasized before -- names, dates, codes, acronyms
("Section 6A", "OROP", "Nov 24, 2025"). BM25 is the opposite: great at exact
term overlap, poor at paraphrase. Combining both, then re-ranking the
merged candidate pool, consistently retrieves better passages than either
alone -- this is the standard approach used in production RAG systems.

The BM25 index is rebuilt lazily, in-memory, whenever the underlying
vector store's contents change size (cheap at the document counts this
project targets; for very large corpora you'd persist/incrementally
update it instead).
"""
import re
from typing import List, Optional, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi

from app.vector_store import VectorStore

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class HybridRetriever:
    def __init__(self, store: VectorStore, alpha: float = 0.5):
        """
        alpha: weight given to semantic (embedding) score vs keyword (BM25)
        score when combining, in [0, 1]. 0.5 weighs them equally.
        """
        self.store = store
        self.alpha = alpha
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_size = -1  # number of chunks the cached BM25 index was built from

    def _ensure_bm25(self) -> None:
        """Rebuilds the BM25 index if the corpus has grown/shrunk since last build."""
        current_size = len(self.store.metadata)
        if self._bm25 is not None and self._bm25_size == current_size:
            return
        corpus = [_tokenize(m["text"]) for m in self.store.metadata]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._bm25_size = current_size

    def search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.store.total_chunks == 0:
            return []

        self._ensure_bm25()

        # --- semantic candidates (already filtered by document_id) ---
        semantic_pool = min(top_k * 4, self.store.total_chunks)
        semantic_hits = self.store.search(query_embedding, top_k=semantic_pool, document_id=document_id)

        # --- keyword candidates ---
        allowed_indices = (
            [i for i, m in enumerate(self.store.metadata) if m["document_id"] == document_id]
            if document_id
            else list(range(len(self.store.metadata)))
        )
        keyword_hits: List[Dict[str, Any]] = []
        if self._bm25 is not None and allowed_indices:
            all_scores = self._bm25.get_scores(_tokenize(query_text))
            ranked = sorted(allowed_indices, key=lambda i: all_scores[i], reverse=True)
            keyword_pool = min(top_k * 4, len(ranked))
            for idx in ranked[:keyword_pool]:
                meta = self.store.metadata[idx]
                keyword_hits.append({**meta, "score": float(all_scores[idx]), "row_index": idx})

        # --- merge on row_index, normalize each score type, blend ---
        merged: Dict[int, Dict[str, Any]] = {}
        for h in semantic_hits:
            merged[h["row_index"]] = {"meta": h, "semantic": h["score"], "keyword": 0.0}
        for h in keyword_hits:
            entry = merged.setdefault(h["row_index"], {"meta": h, "semantic": 0.0, "keyword": 0.0})
            entry["keyword"] = h["score"]

        if not merged:
            return []

        row_indices = list(merged.keys())
        sem_scores = _min_max_normalize([merged[i]["semantic"] for i in row_indices])
        kw_scores = _min_max_normalize([merged[i]["keyword"] for i in row_indices])

        combined: List[Dict[str, Any]] = []
        for row_idx, sem_norm, kw_norm in zip(row_indices, sem_scores, kw_scores):
            meta = merged[row_idx]["meta"]
            blended = self.alpha * sem_norm + (1 - self.alpha) * kw_norm
            combined.append({**meta, "score": blended})

        combined.sort(key=lambda h: h["score"], reverse=True)
        return combined[:top_k]
