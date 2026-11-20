"""Document and chunk data models for ingestion and retrieval."""

from typing import Dict, Any, Literal
from pydantic import BaseModel, Field


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

    def to_metadata(self) -> Dict[str, Any]:
        """Convert chunk attributes to a flat dictionary suitable for ChromaDB and search indexing."""
        return {
            "source": self.source,
            "page": self.page,
            "category": self.category,
            "section": self.section,
            "chunk_type": self.chunk_type,
            "extraction_method": self.extraction_method,
            "quality": self.quality,
        }


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
