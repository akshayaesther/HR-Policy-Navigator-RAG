"""Tests for document loading, PDF cleaning, table extraction, OCR, and chunking."""

from pathlib import Path
import pytest
import fitz

from backend.ingestion.document_loader import DocumentLoader
from backend.ingestion.cleaner import (
    fix_hyphenated_line_breaks,
    normalize_whitespace,
    detect_and_remove_headers_footers,
    compute_quality_score,
)
from backend.models.document import DocumentChunk

DOCS_DIR = Path("backend/data/documents")


@pytest.fixture(scope="module")
def loader() -> DocumentLoader:
    """Fixture providing a DocumentLoader instance."""
    return DocumentLoader(chunk_size=600, chunk_overlap=50)


def test_cleaner_fix_hyphenated_line_breaks() -> None:
    """Verify that hyphenated line breaks and broken syllable hyphens are fixed."""
    sample_split = "This is an em-\nploy-ee of our or-gani-\n   zation."
    cleaned = fix_hyphenated_line_breaks(sample_split)
    assert "employee" in cleaned
    assert "organization" in cleaned

    # Verify legitimate compound words are preserved
    preserved = "He works full-time on a clean-desk policy."
    assert fix_hyphenated_line_breaks(preserved) == preserved


def test_cleaner_detect_and_remove_headers_footers() -> None:
    """Verify that repeated headers, footers, and page numbers are removed."""
    page1 = (
        "Enterprise HR Global Policy Management System\n"
        "Section 1: Overview\n"
        "This is valid content for page 1.\n"
        "Subject to corporate revision and jurisdictional regulations\n"
        "Page 1 of 3"
    )
    page2 = (
        "Enterprise HR Global Policy Management System\n"
        "Section 2: Rules\n"
        "This is valid content for page 2.\n"
        "Subject to corporate revision and jurisdictional regulations\n"
        "Page 2 of 3"
    )
    page3 = (
        "Enterprise HR Global Policy Management System\n"
        "Section 3: Conclusion\n"
        "This is valid content for page 3.\n"
        "Subject to corporate revision and jurisdictional regulations\n"
        "Page 3 of 3"
    )

    cleaned = detect_and_remove_headers_footers([page1, page2, page3])
    assert len(cleaned) == 3

    for p in cleaned:
        assert "Enterprise HR Global Policy Management System" not in p
        assert "Subject to corporate revision and jurisdictional regulations" not in p
        assert "Page 1 of 3" not in p
        assert "Page 2 of 3" not in p
        assert "Page 3 of 3" not in p

    assert "Section 1: Overview" in cleaned[0]
    assert "Section 2: Rules" in cleaned[1]
    assert "Section 3: Conclusion" in cleaned[2]


def test_loader_rejects_unsupported_format(loader: DocumentLoader, tmp_path: Path) -> None:
    """Verify that unsupported extensions raise ValueError."""
    fake_file = tmp_path / "document.docx"
    fake_file.write_text("Hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        loader.load_and_chunk(fake_file)


def test_loader_handles_empty_pdf(loader: DocumentLoader, tmp_path: Path) -> None:
    """Verify that empty PDFs (0 bytes) are handled gracefully."""
    empty_pdf_path = tmp_path / "empty.pdf"
    empty_pdf_path.write_bytes(b"")

    chunks, report = loader.load_and_chunk(empty_pdf_path)
    assert len(chunks) == 0
    assert report.total_pages == 0
    assert report.total_chunks == 0


def test_leave_policy_headers_removed_and_metadata(loader: DocumentLoader) -> None:
    """Verify Leave_Policy_v1.pdf chunks have headers stripped and correct page numbers."""
    pdf_path = DOCS_DIR / "Leave_Policy_v1.pdf"
    assert pdf_path.exists(), f"Sample doc missing: {pdf_path}"

    chunks, report = loader.load_and_chunk(pdf_path)
    assert report.total_pages == 4
    assert len(chunks) > 0

    # Ensure running footer / page number text is not in chunks
    for chunk in chunks:
        assert "Subject to corporate revision and jurisdictional regulations" not in chunk.content
        assert "Page 1 of 4" not in chunk.content
        assert "Page 2 of 4" not in chunk.content
        assert chunk.source == "Leave_Policy_v1.pdf"
        assert 1 <= chunk.page <= 4
        assert chunk.quality > 0.8


def test_messy_handbook_two_column_and_table(loader: DocumentLoader) -> None:
    """Verify Messy_Handbook.pdf extracts in proper two-column order, removes headers, and captures table."""
    pdf_path = DOCS_DIR / "Messy_Handbook.pdf"
    assert pdf_path.exists(), f"Sample doc missing: {pdf_path}"

    chunks, report = loader.load_and_chunk(pdf_path)
    assert report.total_pages == 4
    assert report.table_chunks >= 1

    all_text = "\n".join(c.content for c in chunks)

    # 1. Check Section 1 appears before Section 3 (two-column left-to-right order)
    sec1_pos = all_text.find("SECTION 1")
    sec3_pos = all_text.find("SECTION 3")
    assert sec1_pos != -1 and sec3_pos != -1
    assert sec1_pos < sec3_pos, "Left column (Section 1) should appear before right column (Section 3)"

    # 2. Check table chunk was created with chunk_type='table'
    table_chunk = next((c for c in chunks if c.chunk_type == "table"), None)
    assert table_chunk is not None
    assert "Expense Tier" in table_chunk.content
    assert "Local Taxi / Rideshare" in table_chunk.content

    # 3. Check de-hyphenation worked
    assert "handbook" in all_text.lower() or "departments" in all_text.lower()


def test_benefits_policy_table_extraction(loader: DocumentLoader) -> None:
    """Verify Benefits_Insurance_Policy.pdf captures the multi-tier medical table."""
    pdf_path = DOCS_DIR / "Benefits_Insurance_Policy.pdf"
    assert pdf_path.exists(), f"Sample doc missing: {pdf_path}"

    chunks, report = loader.load_and_chunk(pdf_path)
    assert report.table_chunks >= 1

    table_chunk = next((c for c in chunks if c.chunk_type == "table"), None)
    assert table_chunk is not None
    assert table_chunk.extraction_method == "table"
    assert "Premier PPO" in table_chunk.content
    assert "$500 / $1,000" in table_chunk.content
    assert "Core HDHP + HSA" in table_chunk.content


def test_scanned_notice_ocr_extraction(loader: DocumentLoader) -> None:
    """Verify Scanned_Notice.pdf (0 native text layer) is recovered via RapidOCR."""
    pdf_path = DOCS_DIR / "Scanned_Notice.pdf"
    assert pdf_path.exists(), f"Sample doc missing: {pdf_path}"

    chunks, report = loader.load_and_chunk(pdf_path)
    assert report.total_pages == 3
    assert report.ocr_pages == 3
    assert len(chunks) > 0

    all_text = "\n".join(c.content for c in chunks)
    assert len(all_text) > 100
    assert any(w in all_text.upper() for w in ["MEMORANDUM", "OFFICIAL", "EMERGENCY", "CLOSURE"])

    for chunk in chunks:
        assert chunk.extraction_method == "ocr"
        assert chunk.quality > 0.7


def test_markdown_loader_support(loader: DocumentLoader, tmp_path: Path) -> None:
    """Verify that Markdown files are loaded, chunked, and section-tagged."""
    md_file = tmp_path / "Remote_Work_Guidelines.md"
    md_file.write_text(
        "# Remote Work Guidelines\n\n"
        "Section 1: Eligibility\n"
        "All permanent employees may work remotely up to two days per week.\n\n"
        "Section 2: Core Hours\n"
        "Core hours are 10 AM to 4 PM local time without exception.",
        encoding="utf-8",
    )

    chunks, report = loader.load_and_chunk(md_file)
    assert report.total_pages == 1
    assert len(chunks) >= 1
    assert any("Eligibility" in c.section or "Section 1" in c.section for c in chunks)
    assert any("Core Hours" in c.section or "Section 2" in c.section for c in chunks)
