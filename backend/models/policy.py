"""Metadata model for versioned policy documents."""

from datetime import date
from typing import Dict, Literal, Optional

from pydantic import BaseModel


PolicyStatus = Literal["active", "superseded"]


class PolicyDocument(BaseModel):
    """Persisted identity, lifecycle, and section data for one policy file."""

    doc_id: str
    filename: str
    version: Optional[str] = None
    effective_date: Optional[date] = None
    status: PolicyStatus = "active"
    file_hash: str
    size_bytes: int
    sections: Dict[str, str]
