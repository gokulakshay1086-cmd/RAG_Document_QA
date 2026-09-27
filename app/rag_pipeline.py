"""
Ties together document loading, chunking, embedding, retrieval, and
answer generation into a single pipeline.

Answer generation has two modes:
  - "openai": uses the OpenAI chat completion API (when OPENAI_API_KEY is set)
  - "extractive": a free, fully local fallback that returns the most
    relevant retrieved passages directly, no external API needed.

This means the whole project runs and demos end-to-end with zero paid
API keys -- useful for a portfolio/resume project -- while still
supporting a real LLM when a key is provided.
"""
from pathlib import Path
from typing import List, Optional

from app.config import settings
from app.document_loader import load_document
from app.text_splitter import chunk_text
from app.embeddings import get_embedding_model
from app.vector_store import VectorStore
from app.hybrid_retriever import HybridRetriever
from app.reranker import get_reranker
from app.models import SourceChunk, QueryResponse, UploadResponse

_openai_client = None
if settings.uses_openai:
    from openai import OpenAI

    _openai_client = OpenAI(api_key=settings.openai_api_key)


class RAGPipeline:
    def __init__(self):
        self.embedder = get_embedding_model()
        self.store = VectorStore(
            dimension=self.embedder.dimension,
            persist_dir=settings.vector_store_path,
        )
        self.retriever = HybridRetriever(self.store, alpha=settings.hybrid_alpha)

    # ---------------- Ingestion ----------------

    def ingest_file(self, file_path: Path, original_filename: str) -> UploadResponse:
        text = load_document(file_path)
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise ValueError("Document produced no usable text chunks.")

        embeddings = self.embedder.encode(chunks)
        document_id = self.store.add_document(
            filename=original_filename, chunks=chunks, embeddings=embeddings
        )

        return UploadResponse(
            document_id=document_id,
            filename=original_filename,
            chunks_indexed=len(chunks),
        )

    # ---------------- Retrieval + Answering ----------------

    def answer(
        self, question: str, top_k: Optional[int] = None, document_id: Optional[str] = None
    ) -> QueryResponse:
        k = top_k or settings.top_k
        query_vec = self.embedder.encode_one(question)

        # Pull a wider candidate pool when re-ranking will narrow it back down.
        candidate_k = max(settings.retrieval_candidate_k, k) if settings.use_reranker else k

        if settings.use_hybrid_search:
            hits = self.retriever.search(question, query_vec, top_k=candidate_k, document_id=document_id)
        else:
            hits = self.store.search(query_vec, top_k=candidate_k, document_id=document_id)

        if settings.use_reranker and hits:
            hits = get_reranker().rerank(question, hits, top_k=k)
        else:
            hits = hits[:k]

        sources = [
            SourceChunk(
                document_id=h["document_id"],
                filename=h["filename"],
                chunk_index=h["chunk_index"],
                text=h["text"],
                score=h["score"],
            )
            for h in hits
        ]

        if not hits:
            return QueryResponse(
                answer="I couldn't find any relevant information in the indexed documents "
                "to answer that question. Try uploading a relevant document first.",
                sources=[],
                mode="none",
            )

        if _openai_client is not None:
            answer_text = self._generate_with_openai(question, hits)
            mode = "openai"
        else:
            answer_text = self._generate_extractive(question, hits)
            mode = "extractive"

        return QueryResponse(answer=answer_text, sources=sources, mode=mode)

    # ---------------- Generation strategies ----------------

    def _build_context(self, hits: List[dict]) -> str:
        blocks = []
        for i, h in enumerate(hits, start=1):
            blocks.append(f"[Source {i} | {h['filename']} | chunk {h['chunk_index']}]\n{h['text']}")
        return "\n\n---\n\n".join(blocks)

    def _generate_with_openai(self, question: str, hits: List[dict]) -> str:
        context = self._build_context(hits)
        system_prompt = (
            "You are a precise document Q&A assistant. Answer the user's question "
            "using ONLY the provided context sources. Cite sources inline like [Source 1]. "
            "If the answer is not contained in the context, say so plainly instead of "
            "guessing or using outside knowledge."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        response = _openai_client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()

    def _generate_extractive(self, question: str, hits: List[dict]) -> str:
        """
        Free, local fallback used when no OPENAI_API_KEY is configured.
        Returns the top matching passage(s) directly with light framing,
        so the app is fully demoable without any paid API.
        """
        top = hits[0]
        lines = [
            "No LLM API key is configured, so here is the most relevant passage "
            "found in your documents (set OPENAI_API_KEY in .env for a fully "
            "synthesized natural-language answer):\n",
            f'"{top["text"].strip()}"',
            f"\n(from {top['filename']}, chunk {top['chunk_index']}, "
            f"relevance score {top['score']:.2f})",
        ]
        if len(hits) > 1:
            lines.append(
                f"\n{len(hits) - 1} additional related passage(s) were also found "
                "-- see the sources list below."
            )
        return "\n".join(lines)


_pipeline_singleton: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline_singleton
    if _pipeline_singleton is None:
        _pipeline_singleton = RAGPipeline()
    return _pipeline_singleton
