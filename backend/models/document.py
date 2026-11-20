"""Document and chunk data models for ingestion and retrieval."""

import hashlib
import re
from typing import Dict, Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator


ExtractionMethod = Literal["text", "table", "ocr"]
ChunkType = Literal["text", "table"]


class DocumentChunk(BaseModel):
    """Represents an atomic text or table chunk extracted from a document."""

    id: str = Field(description="Unique identifier for the chunk")
    content: str = Field(description="Cleaned text or table content")
    source: str = Field(description="Source document filename or identifier")
    page: int = Field(description="1-based page number where chunk originated")
    category: str = Field(default="hr_policy", description="Document domain category")
    section: str = Field(default="General", description="Section heading context")
    chunk_type: ChunkType = Field(default="text", description="Type of chunk: text or table")
    extraction_method: ExtractionMethod = Field(default="text", description="Method used to extract this chunk")
    quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Quality score between 0.0 and 1.0")
    doc_id: Optional[str] = Field(default=None, description="Stable identifier for the policy family")
    version: Optional[str] = Field(default=None, description="Policy version")
    effective_date: Optional[str] = Field(default=None, description="Policy effective date in ISO format")
    status: Literal["active", "superseded"] = Field(
        default="active",
        description="Whether this policy version is current or superseded",
    )
    file_hash: Optional[str] = Field(default=None, description="SHA-256 hash of the source file")

    @model_validator(mode="after")
    def set_default_doc_id(self) -> "DocumentChunk":
        """Derive a policy-family ID when callers provide only the source filename."""
        if self.doc_id is None:
            stem = self.source.rsplit(".", 1)[0]
            stem = re.sub(r"(?:[_\s.-]+v\d+(?:[._-]\d+)*)$", "", stem, flags=re.IGNORECASE)
            self.doc_id = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
        return self

    def to_metadata(self) -> Dict[str, Any]:
        """Convert chunk attributes to a flat dictionary suitable for ChromaDB and search indexing."""
        metadata = {
            "source": self.source,
            "page": self.page,
            "category": self.category,
            "section": self.section,
            "chunk_type": self.chunk_type,
            "extraction_method": self.extraction_method,
            "quality": self.quality,
            "doc_id": self.doc_id,
            "version": self.version,
            "effective_date": self.effective_date,
            "status": self.status,
            "file_hash": self.file_hash,
            "content_hash": hashlib.sha256(self.content.encode("utf-8")).hexdigest(),
        }
        return {key: value for key, value in metadata.items() if value is not None}


class IngestionReport(BaseModel):
    """Summary report for document ingestion per file."""

    source: str
    total_pages: int
    total_chunks: int
    text_chunks: int
    table_chunks: int
    ocr_pages: int
    average_quality: float
    methods_breakdown: Dict[str, int]
