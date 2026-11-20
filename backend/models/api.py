"""Request and response models for the public API."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.models.search import RetrievalMode


class AskRequest(BaseModel):
    """Question and retrieval options supplied to the answer endpoint."""

    query: str = Field(min_length=1, max_length=4000)
    mode: RetrievalMode = "hybrid_rerank"
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class Citation(BaseModel):
    """Source location backing an answer."""

    document: str
    page: int
    section: str


class RetrievedChunk(BaseModel):
    """Retrieved context and relevance score returned for transparency."""

    chunk_id: str
    content: str
    document: str
    page: int
    section: str
    score: float
    rank: int


class AskResponse(BaseModel):
    """Answer, citations, retrieval evidence, and request diagnostics."""

    answer: str
    citations: List[Citation]
    retrieved_chunks: List[RetrievedChunk]
    mode: RetrievalMode
    latency_ms: float
    refused: bool


class DocumentInfo(BaseModel):
    """A PDF or Markdown document available to the retrieval system."""

    id: str
    filename: str
    size_bytes: int


class DocumentUploadResponse(BaseModel):
    """Summary of a successfully ingested document."""

    id: str
    filename: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    """API health status."""

    status: str
    app: str
    environment: str
