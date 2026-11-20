"""Tests for policy family, version, and effective-date extraction."""

from datetime import date

from backend.ingestion.policy_metadata import (
    derive_doc_id,
    extract_policy_metadata,
)


def test_doc_id_uses_policy_family_without_version_suffix() -> None:
    assert derive_doc_id("Leave_Policy_v2.pdf") == "leave_policy"


def test_policy_version_and_effective_date_are_extracted_from_document_text() -> None:
    metadata = extract_policy_metadata(
        "Leave Policy\nVersion: 2.0 | Effective Date: 2025-04-01"
    )

    assert metadata == {"version": "2.0", "effective_date": date(2025, 4, 1)}


def test_doc_id_can_be_supplied_explicitly() -> None:
    assert derive_doc_id("Leave_Policy_v2.pdf", "leave") == "leave"
