"""Request and response models for the public API."""

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from backend.models.search import RetrievalMode


class AskRequest(BaseModel):
    """Question and retrieval options supplied to the answer endpoint."""

    query: str = Field(min_length=1, max_length=4000)
    mode: RetrievalMode = "hybrid_rerank"
    filters: Optional[Dict[str, Any]] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    as_of_date: Optional[date] = None


class Citation(BaseModel):
    """Source location backing an answer."""

    document: str
    page: int
    section: str
    version: Optional[str] = None
    effective_date: Optional[date] = None


class RetrievedChunk(BaseModel):
    """Retrieved context and relevance score returned for transparency."""

    chunk_id: str
    content: str
    document: str
    page: int
    section: str
    score: float
    rank: int
    doc_id: Optional[str] = None
    version: Optional[str] = None
    effective_date: Optional[date] = None


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
    doc_id: str
    version: Optional[str] = None
    effective_date: Optional[date] = None
    status: str
    file_hash: str


class DocumentUploadResponse(BaseModel):
    """Summary of a successfully ingested document."""

    id: str
    filename: str
    doc_id: str
    version: Optional[str] = None
    effective_date: Optional[date] = None
    status: str
    file_hash: str
    chunks_indexed: int
    skipped: bool = False


class PolicySectionChange(BaseModel):
    """Section-level difference between two policy versions."""

    section: str
    change: str


class PolicyChangesResponse(BaseModel):
    """Summary of section changes between the latest policy versions."""

    doc_id: str
    from_version: Optional[str] = None
    to_version: Optional[str] = None
    changes: List[PolicySectionChange]


class HealthResponse(BaseModel):
    """API health status."""

    status: str
    app: str
    environment: str
