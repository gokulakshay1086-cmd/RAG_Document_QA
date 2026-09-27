"""
Re-ranks a candidate pool of retrieved chunks using a cross-encoder.

Why this helps: bi-encoder embedding search (what FAISS does) scores the
query and each chunk independently, then compares vectors -- fast, but
approximate. A cross-encoder feeds the (query, chunk) pair into the model
*together*, so it can directly attend to how they relate -- slower (can't
be pre-computed/indexed) but noticeably more accurate. The standard
production pattern is: cheap retrieval gets a wide candidate pool (e.g. 20),
then a cross-encoder re-ranks just that pool down to the final top-k.
"""
from functools import lru_cache
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

from app.config import settings


class Reranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs)

        reranked = [{**c, "score": float(s)} for c, s in zip(candidates, scores)]
        reranked.sort(key=lambda c: c["score"], reverse=True)
        return reranked[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(settings.reranker_model)
