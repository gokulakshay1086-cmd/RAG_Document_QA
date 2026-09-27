"""
Tests for the hybrid retriever's BM25 + semantic score blending.

These don't require downloading any models (BM25 is pure Python), so they
run fast and offline. The re-ranker (cross-encoder) is not covered here
since it requires downloading a model on first use -- see the note in
test_api.py's docstring if you want to add a network-gated test for it.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.vector_store import VectorStore
from app.hybrid_retriever import HybridRetriever, _tokenize, _min_max_normalize


def test_tokenize_lowercases_and_splits():
    tokens = _tokenize("Justice Surya Kant, 53rd CJI!")
    assert tokens == ["justice", "surya", "kant", "53rd", "cji"]


def test_min_max_normalize_handles_constant_values():
    assert _min_max_normalize([5.0, 5.0, 5.0]) == [1.0, 1.0, 1.0]


def test_min_max_normalize_scales_to_unit_range():
    result = _min_max_normalize([0.0, 5.0, 10.0])
    assert result == [0.0, 0.5, 1.0]


def _build_store(tmp_path):
    """Four chunks where embeddings alone would NOT surface the exact-term match."""
    store = VectorStore(dimension=4, persist_dir=tmp_path)
    # All embeddings deliberately similar/orthogonal-ish so keyword overlap
    # is what should decide the winner for a query containing "OROP".
    embeddings = np.array(
        [
            [1, 0, 0, 0],
            [0.9, 0.1, 0, 0],
            [0.8, 0.2, 0, 0],
            [0.85, 0.15, 0, 0],
        ],
        dtype="float32",
    )
    chunks = [
        "The court discussed general pension policy for government employees.",
        "He upheld the validity of the One Rank One Pension OROP scheme for the armed forces.",
        "The bench reviewed unrelated matters concerning citizenship law.",
        "Several judgments referenced constitutional interpretation broadly.",
    ]
    store.add_document("test.txt", chunks, embeddings)
    return store


def test_hybrid_search_surfaces_exact_keyword_match(tmp_path):
    store = _build_store(tmp_path)
    retriever = HybridRetriever(store, alpha=0.5)

    # Query embedding intentionally closest to chunk 0 (pure semantic would
    # rank chunk 0 first), but the query text contains the exact term "OROP"
    # which only appears in chunk 1.
    query_embedding = np.array([1, 0, 0, 0], dtype="float32")
    results = retriever.search("What did the court say about OROP?", query_embedding, top_k=2)

    assert len(results) > 0
    top_texts = [r["text"] for r in results]
    assert any("OROP" in t for t in top_texts)


def test_hybrid_search_respects_document_id_filter(tmp_path):
    store = _build_store(tmp_path)
    other_embeddings = np.array([[0, 0, 1, 0]], dtype="float32")
    other_doc_id = store.add_document("other.txt", ["A completely different document about cooking."], other_embeddings)

    retriever = HybridRetriever(store, alpha=0.5)
    query_embedding = np.array([0, 0, 1, 0], dtype="float32")
    results = retriever.search("cooking", query_embedding, top_k=5, document_id=other_doc_id)

    assert all(r["document_id"] == other_doc_id for r in results)


def test_hybrid_search_empty_store_returns_empty(tmp_path):
    store = VectorStore(dimension=4, persist_dir=tmp_path)
    retriever = HybridRetriever(store, alpha=0.5)
    results = retriever.search("anything", np.zeros(4, dtype="float32"), top_k=3)
    assert results == []
