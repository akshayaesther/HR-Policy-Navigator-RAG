"""Search data models and retrieval result schemas."""

from typing import Dict, Any, Literal
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
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Full metadata attributes")
