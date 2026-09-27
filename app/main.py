"""
FastAPI entrypoint for the RAG Document Intelligence & Q&A System.

Run with:
    uvicorn app.main:app --reload --port 8000

Interactive API docs: http://localhost:8000/docs
"""
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    DocumentInfo,
)
from app.rag_pipeline import get_pipeline
from app.document_loader import UnsupportedFileTypeError

app = FastAPI(
    title="RAG Document Intelligence & Q&A System",
    description=(
        "Upload documents (PDF / DOCX / TXT / MD) and ask natural-language "
        "questions answered using Retrieval-Augmented Generation."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    pipeline = get_pipeline()
    return HealthResponse(
        status="ok",
        documents_indexed=len(pipeline.store.list_documents()),
        total_chunks=pipeline.store.total_chunks,
        llm_mode="openai" if settings.uses_openai else "extractive (no API key set)",
    )


@app.post("/documents/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    temp_name = f"{uuid.uuid4().hex}{suffix}"
    dest_path = settings.upload_path / temp_name

    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pipeline = get_pipeline()
        result = pipeline.ingest_file(dest_path, original_filename=file.filename)
        return result

    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        file.file.close()


@app.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
def list_documents():
    pipeline = get_pipeline()
    return pipeline.store.list_documents()


@app.delete("/documents/{document_id}", tags=["Documents"])
def delete_document(document_id: str):
    pipeline = get_pipeline()
    removed = pipeline.store.delete_document(document_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"message": f"Removed {removed} chunks for document {document_id}."}


@app.post("/query", response_model=QueryResponse, tags=["Q&A"])
def query(request: QueryRequest):
    pipeline = get_pipeline()
    try:
        return pipeline.answer(
            question=request.question,
            top_k=request.top_k,
            document_id=request.document_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@app.get("/", tags=["System"])
def root():
    return {
        "message": "RAG Document Intelligence & Q&A System is running.",
        "docs": "/docs",
        "health": "/health",
    }
