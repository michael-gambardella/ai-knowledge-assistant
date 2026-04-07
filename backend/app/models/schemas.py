from pydantic import BaseModel, Field
from typing import Optional


# --- Document schemas ---

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    message: str


class DocumentListResponse(BaseModel):
    documents: list[dict]
    total: int


# --- Query schemas ---

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    document_id: str
    filename: str
    content: str
    relevance_score: float
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    retrieval_ms: float
    generation_ms: float
    total_ms: float


# --- Health schema ---

class HealthResponse(BaseModel):
    status: str
    environment: str
    version: str
