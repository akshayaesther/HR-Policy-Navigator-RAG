"""Document ingestion and cleaning package."""

from backend.ingestion.cleaner import (
    fix_hyphenated_line_breaks,
    normalize_whitespace,
    detect_and_remove_headers_footers,
    compute_quality_score,
)
from backend.ingestion.document_loader import DocumentLoader, print_ingestion_report

__all__ = [
    "DocumentLoader",
    "print_ingestion_report",
    "fix_hyphenated_line_breaks",
    "normalize_whitespace",
    "detect_and_remove_headers_footers",
    "compute_quality_score",
]
