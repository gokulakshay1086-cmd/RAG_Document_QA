"""
Basic tests covering text splitting, the vector store, and the API layer.

Run with:
    pytest -v
"""
import io
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.text_splitter import chunk_text
from app.vector_store import VectorStore
from app.main import app


# ---------------- text_splitter ----------------

def test_chunk_text_basic():
    text = "para one.\n\npara two.\n\npara three."
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) == 1  # small enough to fit in one chunk


def test_chunk_text_splits_long_text():
    paragraph = "sentence. " * 200  # long paragraph
    chunks = chunk_text(paragraph, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 200 + 20  # allow small slack from overlap logic


def test_chunk_overlap_must_be_smaller():
    with pytest.raises(ValueError):
        chunk_text("hello world", chunk_size=10, chunk_overlap=10)


# ---------------- vector_store ----------------

def test_vector_store_add_and_search(tmp_path):
    store = VectorStore(dimension=4, persist_dir=tmp_path)
    embeddings = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype="float32"
    )
    doc_id = store.add_document("test.txt", ["chunk a", "chunk b", "chunk c"], embeddings)

    query = np.array([1, 0, 0, 0], dtype="float32")
    results = store.search(query, top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "chunk a"
    assert results[0]["document_id"] == doc_id


def test_vector_store_delete(tmp_path):
    store = VectorStore(dimension=4, persist_dir=tmp_path)
    embeddings = np.eye(4, dtype="float32")
    doc_id = store.add_document("doc.txt", ["a", "b", "c", "d"], embeddings)
    assert store.total_chunks == 4

    removed = store.delete_document(doc_id)
    assert removed == 4
    assert store.total_chunks == 0


def test_vector_store_persistence(tmp_path):
    store = VectorStore(dimension=4, persist_dir=tmp_path)
    embeddings = np.eye(4, dtype="float32")
    store.add_document("doc.txt", ["a", "b", "c", "d"], embeddings)

    # Simulate reload from disk
    reloaded = VectorStore(dimension=4, persist_dir=tmp_path)
    assert reloaded.total_chunks == 4


# ---------------- API ----------------

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "llm_mode" in body


def test_upload_and_query_txt_document():
    content = b"The Eiffel Tower is located in Paris, France. It was completed in 1889."
    files = {"file": ("facts.txt", io.BytesIO(content), "text/plain")}
    upload_resp = client.post("/documents/upload", files=files)
    assert upload_resp.status_code == 200
    assert upload_resp.json()["chunks_indexed"] >= 1

    query_resp = client.post("/query", json={"question": "Where is the Eiffel Tower?"})
    assert query_resp.status_code == 200
    body = query_resp.json()
    assert len(body["sources"]) > 0
    assert "Paris" in body["sources"][0]["text"]


def test_upload_rejects_unsupported_extension():
    files = {"file": ("data.exe", io.BytesIO(b"binary"), "application/octet-stream")}
    resp = client.post("/documents/upload", files=files)
    assert resp.status_code == 400
