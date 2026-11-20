"""Helpers for extracting version metadata from policy documents."""

import hashlib
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from backend.models.document import DocumentChunk

VERSION_PATTERN = re.compile(
    r"\bversion\s*[:#]?\s*([vV]?\d+(?:\.\d+)*(?:[-+][\w.-]+)?)",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(
    r"\beffective\s+date\s*[:#]?\s*"
    r"(\d{4}-\d{2}-\d{2}|[A-Za-z]+\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
VERSION_SUFFIX_PATTERN = re.compile(r"(?:[_\s.-]+v\d+(?:[._-]\d+)*)$", re.IGNORECASE)
DOC_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$")


def derive_doc_id(filename: str, supplied_doc_id: Optional[str] = None) -> str:
    """Derive a stable policy-family ID from a filename or validate an override."""
    if supplied_doc_id:
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", supplied_doc_id.strip()).strip("_.-")
        if not DOC_ID_PATTERN.fullmatch(normalized):
            raise ValueError("doc_id must contain only letters, numbers, dots, underscores, or hyphens.")
        return normalized.lower()

    stem = Path(filename).stem
    stem = VERSION_SUFFIX_PATTERN.sub("", stem)
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    if not normalized:
        raise ValueError("Could not derive a policy family ID from the filename.")
    return normalized


def extract_policy_metadata(text: str) -> Dict[str, Optional[date]]:
    """Extract version and effective date when they appear in document text."""
    version_match = VERSION_PATTERN.search(text)
    date_match = DATE_PATTERN.search(text)
    effective_date: Optional[date] = None

    if date_match:
        raw_date = date_match.group(1).replace(",", "")
        for date_format in ("%Y-%m-%d", "%B %d %Y", "%b %d %Y"):
            try:
                effective_date = datetime.strptime(raw_date, date_format).date()
                break
            except ValueError:
                continue

    return {
        "version": version_match.group(1).lstrip("vV") if version_match else None,
        "effective_date": effective_date,
    }


def file_sha256(file_path: Path) -> str:
    """Return the SHA-256 digest of a document's bytes."""
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as document_file:
        for block in iter(lambda: document_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_policy_metadata(
    chunks: List[DocumentChunk],
    *,
    doc_id: str,
    version: Optional[str],
    effective_date: Optional[date],
    file_hash: str,
    status: str = "active",
) -> List[DocumentChunk]:
    """Attach document-level metadata to every chunk and return the chunks."""
    for chunk in chunks:
        chunk.doc_id = doc_id
        chunk.version = version
        chunk.effective_date = effective_date.isoformat() if effective_date else None
        chunk.status = status
        chunk.file_hash = file_hash
    return chunks
