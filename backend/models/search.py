"""Search data models and retrieval result schemas."""

from datetime import date
from typing import Dict, Any, Literal, Optional
from pydantic import BaseModel, Field


RetrievalMode = Literal["hybrid", "dense", "bm25", "hybrid_rerank"]


class SearchResult(BaseModel):
    """Normalized search candidate returned across any retrieval strategy."""

    chunk_id: str = Field(description="Unique identifier for the chunk")
    content: str = Field(description="Chunk text or table content")
    source: str = Field(description="Source document filename")
    page: int = Field(description="1-based page number")
    category: str = Field(default="hr_policy", description="Document domain category")
    section: str = Field(default="General", description="Section heading")
    chunk_type: str = Field(default="text", description="Chunk type: text or table")
    extraction_method: str = Field(default="text", description="Extraction method: text, table, or ocr")
    score: float = Field(description="Relevance or fusion score")
    rank: int = Field(description="1-based retrieval ranking position")
    doc_id: Optional[str] = None
    version: Optional[str] = None
    effective_date: Optional[date] = None
    status: Optional[Literal["active", "superseded"]] = None
    file_hash: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Full metadata attributes")
