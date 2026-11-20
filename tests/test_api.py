"""API tests using temporary document storage and mocked retrieval/generation."""

from pathlib import Path

import pytest
import httpx
import fitz
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.config import Settings
from backend.models.document import DocumentChunk
from backend.models.search import SearchResult


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Provide an API client whose document directory is isolated per test."""
    settings = Settings(
        base_dir=tmp_path,
        documents_dir=Path("documents"),
        vectorstore_dir=Path("vectorstore"),
        relevance_score_threshold=0.1,
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(main_module, "index_chunks", lambda chunks: None)
    monkeypatch.setattr(main_module, "delete_by_source", lambda source: None)
    monkeypatch.setattr(main_module, "update_policy_status", lambda **kwargs: None)
    return TestClient(main_module.app)


def make_result(score: float = 0.8) -> SearchResult:
    """Build a retrieval result with citation metadata."""
    return SearchResult(
        chunk_id="leave-1",
        content="Full-time employees receive 20 paid annual leave days.",
        source="Leave_Policy.pdf",
        page=2,
        category="hr_policy",
        section="Annual Leave",
        chunk_type="text",
        extraction_method="text",
        score=score,
        rank=1,
        doc_id="leave_policy",
        version="1.0",
        effective_date="2024-01-01",
        metadata={
            "doc_id": "leave_policy",
            "version": "1.0",
            "effective_date": "2024-01-01",
        },
    )


@pytest.mark.parametrize("results", [[], [make_result(0.05)]])
def test_ask_refuses_without_calling_llm_when_no_relevant_context(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    results: list[SearchResult],
) -> None:
    """Missing or below-threshold evidence should trigger the exact refusal."""
    monkeypatch.setattr(main_module, "hybrid_search", lambda **kwargs: results)
    monkeypatch.setattr(
        main_module,
        "get_answer_generator",
        lambda: pytest.fail("LLM must not be called for irrelevant results"),
    )

    response = api_client.post("/ask", json={"query": "What is the leave allowance?"})

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "I cannot answer based on internal documentation."
    assert data["refused"] is True
    assert data["citations"] == []


def test_ask_returns_document_page_section_citations(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relevant results should be passed through with the requested citation fields."""
    monkeypatch.setattr(main_module, "hybrid_search", lambda **kwargs: [make_result()])

    class MockGenerator:
        def generate(self, query: str, chunks: list[SearchResult]) -> str:
            return "Employees receive 20 paid annual leave days."

    monkeypatch.setattr(main_module, "get_answer_generator", lambda: MockGenerator())

    response = api_client.post(
        "/ask",
        json={"query": "How many paid leave days?", "mode": "hybrid_rerank"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"].startswith("Employees receive 20 paid annual leave days.")
    assert "version 1.0, effective 2024-01-01" in data["answer"]
    assert data["citations"] == [
        {
            "document": "Leave_Policy.pdf",
            "page": 2,
            "section": "Annual Leave",
            "version": "1.0",
            "effective_date": "2024-01-01",
        }
    ]
    assert data["retrieved_chunks"][0]["score"] == 0.8
    assert data["mode"] == "hybrid_rerank"
    assert data["latency_ms"] >= 0
    assert data["refused"] is False


def test_upload_rejects_unsupported_file_type(api_client: TestClient) -> None:
    """Only PDF and Markdown uploads are accepted."""
    response = api_client.post(
        "/documents/upload",
        files={"file": ("policy.txt", b"not a supported document", "text/plain")},
    )

    assert response.status_code == 415
    assert api_client.get("/documents").json() == []


def test_upload_ingests_markdown_to_temporary_document_storage(
    api_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Markdown uploads are saved under configured temporary storage and indexed."""
    indexed_chunks: list[DocumentChunk] = []
    monkeypatch.setattr(main_module, "index_chunks", indexed_chunks.extend)

    response = api_client.post(
        "/documents/upload",
        files={
            "file": (
                "remote-work.md",
                b"# Remote Work\n\nEmployees may work remotely two days each week.",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["id"] == "remote-work.md"
    stored_file = tmp_path / "documents" / "remote-work.md"
    assert stored_file.exists()
    assert response.json()["chunks_indexed"] == len(indexed_chunks)
    assert indexed_chunks[0].source == "remote-work.md"

    documents = api_client.get("/documents")
    assert documents.status_code == 200
    assert [document["id"] for document in documents.json()] == ["remote-work.md"]


def test_upload_accepts_pdf(api_client: TestClient) -> None:
    """A valid PDF is accepted and ingested."""
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Employees may work remotely two days each week.")

    response = api_client.post(
        "/documents/upload",
        files={"file": ("remote-work.pdf", pdf.tobytes(), "application/pdf")},
    )
    pdf.close()

    assert response.status_code == 201
    assert response.json()["id"] == "remote-work.pdf"


def test_delete_removes_document_and_index_entries(
    api_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deleting a document removes its temporary file and asks the index to purge it."""
    api_client.post(
        "/documents/upload",
        files={"file": ("delete-me.md", b"Policy text.", "text/markdown")},
    )
    deleted_sources: list[str] = []
    monkeypatch.setattr(main_module, "delete_by_source", deleted_sources.append)

    response = api_client.delete("/documents/delete-me.md")

    assert response.status_code == 204
    assert deleted_sources == ["delete-me.md"]
    assert not (tmp_path / "documents" / "delete-me.md").exists()
    assert api_client.get("/documents").json() == []


def test_upload_skips_duplicate_file_hash(api_client: TestClient) -> None:
    """Re-uploading identical bytes is idempotent and reports that it was skipped."""
    uploaded_file = ("same.md", b"Policy text.", "text/markdown")
    assert api_client.post("/documents/upload", files={"file": uploaded_file}).status_code == 201

    response = api_client.post("/documents/upload", files={"file": uploaded_file})

    assert response.status_code == 200
    assert response.json()["skipped"] is True
    assert response.json()["chunks_indexed"] == 0


def test_upload_removes_file_when_indexing_fails(
    api_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed index operation must not leave an unregistered document behind."""
    deleted_sources: list[str] = []

    def fail_index(chunks: list[DocumentChunk]) -> None:
        raise RuntimeError("Embedding service unavailable")

    monkeypatch.setattr(main_module, "index_chunks", fail_index)
    monkeypatch.setattr(main_module, "delete_by_source", deleted_sources.append)
    response = api_client.post(
        "/documents/upload",
        files={"file": ("failed.md", b"Policy text.", "text/markdown")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Document indexing failed; the uploaded file was removed."
    )
    assert deleted_sources == ["failed.md"]
    assert not (tmp_path / "documents" / "failed.md").exists()
    assert api_client.get("/documents").json() == []


def test_new_version_with_same_filename_is_preserved(api_client: TestClient) -> None:
    """A changed file with the same original filename is stored as a new version."""
    first = (
        "leave.md",
        b"Leave policy\nVersion: 1.0\nEffective Date: 2024-01-01\n\n## Annual Leave\n20 days.",
        "text/markdown",
    )
    second = (
        "leave.md",
        b"Leave policy\nVersion: 2.0\nEffective Date: 2025-04-01\n\n## Annual Leave\n24 days.",
        "text/markdown",
    )
    assert api_client.post("/documents/upload", files={"file": first}).status_code == 201

    response = api_client.post("/documents/upload", files={"file": second})

    assert response.status_code == 201
    data = response.json()
    assert data["doc_id"] == "leave"
    assert data["version"] == "2.0"
    assert data["filename"] != "leave.md"
    documents = api_client.get("/documents").json()
    assert len(documents) == 2
    assert {document["status"] for document in documents} == {"active", "superseded"}


def test_ask_reports_ollama_offline_clearly(
    api_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API should expose a clear service error when Ollama cannot be reached."""
    monkeypatch.setattr(main_module, "hybrid_search", lambda **kwargs: [make_result()])

    def raise_connection_error(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "post", raise_connection_error)

    response = api_client.post("/ask", json={"query": "How many leave days?"})

    assert response.status_code == 503
    assert "Ollama is unavailable" in response.json()["detail"]
