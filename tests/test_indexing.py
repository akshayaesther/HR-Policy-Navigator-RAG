"""Tests for Step 3: Dense ChromaDB indexing, BM25 persistence, and dual search capabilities."""

from pathlib import Path

import pytest

import backend.embeddings.vector_store as vector_store_module
import backend.retrieval.bm25_index as bm25_index_module
from backend.ingestion.document_loader import DocumentLoader
from backend.models.document import DocumentChunk
from backend.embeddings.vector_store import (
    ChromaVectorStore,
    index_chunks,
    delete_by_source,
    search_dense,
)
from backend.retrieval.bm25_index import BM25Index, search_bm25

DOCS_DIR = Path("backend/data/documents")


@pytest.fixture(scope="module", autouse=True)
def setup_indexes(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> None:
    """Index sample documents into isolated temporary ChromaDB and BM25 stores."""
    test_data_dir = tmp_path_factory.mktemp("test_indexing")
    vstore = ChromaVectorStore(
        persist_dir=test_data_dir / "chroma",
        collection_name="test_collection_indexing",
    )
    bm25 = BM25Index(index_path=test_data_dir / "bm25_index.pkl")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(vector_store_module, "get_vector_store", lambda: vstore)
    monkeypatch.setattr(bm25_index_module, "get_bm25_index", lambda: bm25)
    request.addfinalizer(monkeypatch.undo)

    loader = DocumentLoader()
    all_chunks = []

    for pdf_file in sorted(DOCS_DIR.glob("*.pdf")):
        chunks, _ = loader.load_and_chunk(pdf_file)
        all_chunks.extend(chunks)

    assert len(all_chunks) > 0, "No sample document chunks loaded"
    index_chunks(all_chunks)


def test_indexing_sample_docs_populates_both_stores() -> None:
    """Verify that index_chunks stores records in ChromaDB and BM25."""
    vstore = vector_store_module.get_vector_store()
    bm25 = bm25_index_module.get_bm25_index()

    assert vstore.count() > 0, "ChromaDB vector store should not be empty"
    assert bm25.is_indexed(), "BM25 index should be indexed"
    assert len(bm25.chunks) == vstore.count(), "ChromaDB and BM25 should hold matching chunk counts"


def test_exact_keyword_query_hits_bm25() -> None:
    """Verify that precise keyword tokens (exact IDs and statutes) hit BM25 with high scores."""
    # 1. Exact document ID query
    res_id = search_bm25("POL-HR-LEAVE-001", k=3)
    assert len(res_id) > 0
    assert res_id[0].source == "Leave_Policy_v1.pdf"
    assert "POL-HR-LEAVE-001" in res_id[0].content

    # 2. Exact acronym query
    res_fcpa = search_bm25("FCPA", k=3)
    assert len(res_fcpa) > 0
    assert res_fcpa[0].source == "Code_of_Conduct.pdf"
    assert "FCPA" in res_fcpa[0].content

    # 3. Exact hotline number
    res_phone = search_bm25("+1-800-555-ETHIC", k=3)
    assert len(res_phone) > 0
    assert res_phone[0].source == "Code_of_Conduct.pdf"
    assert "555-ETHIC" in res_phone[0].content


def test_paraphrased_query_hits_dense_search() -> None:
    """Verify that natural language semantic queries with zero direct keyword overlap hit dense search."""
    # Paraphrased query for sick absence rules (query avoids the word 'leave')
    query_sick = "rules around missing work when falling ill or needing a doctor note"
    dense_sick = search_dense(query_sick, k=3)

    assert len(dense_sick) > 0
    top_content_sick = dense_sick[0].content.lower()
    # Should identify sick or medical absence sections
    assert "sick" in top_content_sick or "medical" in top_content_sick or "absence" in top_content_sick
    assert dense_sick[0].source in ["Leave_Policy_v1.pdf", "Leave_Policy_v2.pdf"]

    # Paraphrased query for unauthorized out-of-state remote work
    query_wfh = "penalties for doing my job from another jurisdiction without authorization"
    dense_wfh = search_dense(query_wfh, k=3)

    assert len(dense_wfh) > 0
    assert dense_wfh[0].source == "WFH_Policy.pdf"
    assert "unauthorized" in dense_wfh[0].content.lower()


def test_delete_by_source() -> None:
    """Verify that delete_by_source purges chunks from both vector store and BM25 index."""
    test_source = "Temporary_Test_Policy.pdf"
    temp_chunk = DocumentChunk(
        id="temp_chunk_999",
        content="This is temporary test policy content for testing deletion.",
        source=test_source,
        page=1,
        category="hr_policy",
        section="Test Section",
        chunk_type="text",
        extraction_method="text",
        quality=0.9,
    )

    # Insert temporary chunk
    index_chunks([temp_chunk])

    # Verify present in BM25 and Dense
    bm25_before = search_bm25("temporary test policy content", k=5)
    assert any(r.source == test_source for r in bm25_before)

    dense_before = search_dense("temporary test policy content", k=5)
    assert any(r.source == test_source for r in dense_before)

    # Delete
    delete_by_source(test_source)

    # Verify removed from both
    bm25_after = search_bm25("temporary test policy content", k=5)
    assert not any(r.source == test_source for r in bm25_after)

    dense_after = search_dense("temporary test policy content", k=5)
    assert not any(r.source == test_source for r in dense_after)


def test_bm25_index_persistence(tmp_path: Path) -> None:
    """Verify that BM25Index serializes and deserializes cleanly to disk."""
    index_file = tmp_path / "custom_bm25.pkl"
    original_index = BM25Index(index_path=index_file)

    sample_chunks = [
        DocumentChunk(
            id="c1",
            content="Employees must submit receipts within thirty calendar days.",
            source="Expense.pdf",
            page=1,
            category="hr_policy",
            section="Expenses",
            chunk_type="text",
            extraction_method="text",
            quality=0.9,
        ),
        DocumentChunk(
            id="c2",
            content="Standard annual vacation allowance is 24 days per year.",
            source="Leave.pdf",
            page=1,
            category="hr_policy",
            section="Vacation",
            chunk_type="text",
            extraction_method="text",
            quality=0.95,
        ),
    ]

    original_index.index_chunks(sample_chunks, persist=True)
    assert index_file.exists()

    # Create fresh index pointing to the persisted file
    reloaded_index = BM25Index(index_path=index_file)
    assert reloaded_index.is_indexed()
    assert len(reloaded_index.chunks) == 2

    # Query reloaded index
    results = reloaded_index.search("receipts thirty days", k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
