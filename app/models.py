"""
Pydantic schemas shared across the API.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_indexed: int
    message: str = "Document ingested successfully."


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    top_k: Optional[int] = Field(None, description="Override number of chunks retrieved")
    document_id: Optional[str] = Field(
        None, description="Restrict search to a single previously uploaded document"
    )


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    mode: str  # "openai" or "extractive"


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    total_chunks: int
    llm_mode: str
