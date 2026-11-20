"""Versioned upload, active/historical search, and idempotency integration tests."""

from pathlib import Path
from typing import List

import pytest
from fastapi.testclient import TestClient

import backend.embeddings.vector_store as vector_store_module
import backend.main as main_module
import backend.retrieval.bm25_index as bm25_index_module
from backend.config import Settings
from backend.embeddings.vector_store import ChromaVectorStore
from backend.models.search import SearchResult
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.hybrid_search import HybridSearchEngine

DOCUMENTS_DIR = Path("backend/data/documents")


class FixedEmbedder:
    """Provide deterministic embeddings without loading a model in tests."""

    def embed_text(self, text: str) -> List[float]:
        return [1.0, 0.0, 0.0]

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]

    def embed_chunks(self, chunks: list) -> List[List[float]]:
        return [self.embed_text(chunk.content) for chunk in chunks]


@pytest.fixture
def versioned_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """Wire the API to temporary Chroma and BM25 stores."""
    settings = Settings(
        base_dir=tmp_path,
        documents_dir=Path("documents"),
        vectorstore_dir=Path("vectorstore"),
        relevance_score_threshold=0.1,
    )
    vector_store = ChromaVectorStore(
        persist_dir=tmp_path / "vectorstore",
        collection_name="test_policy_versions",
        embedder=FixedEmbedder(),  # type: ignore[arg-type]
    )
    bm25_index = BM25Index(index_path=tmp_path / "vectorstore" / "bm25.pkl")
    engine = HybridSearchEngine(
        vector_store=vector_store,
        bm25_index=bm25_index,
        reranker=object(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(vector_store_module, "get_vector_store", lambda: vector_store)
    monkeypatch.setattr(bm25_index_module, "get_bm25_index", lambda: bm25_index)
    monkeypatch.setattr(
        main_module,
        "hybrid_search",
        lambda **kwargs: engine.search(**kwargs),
    )

    class MockGenerator:
        def generate(self, query: str, chunks: list[SearchResult]) -> str:
            answer = "24" if any("twenty-four" in chunk.content.lower() for chunk in chunks) else "20"
            return f"Employees receive {answer} paid annual leave days."

    monkeypatch.setattr(main_module, "get_answer_generator", lambda: MockGenerator())
    return TestClient(main_module.app)


def upload_leave_version(client: TestClient, version: str) -> dict:
    """Upload a bundled leave policy version through the public endpoint."""
    path = DOCUMENTS_DIR / f"Leave_Policy_{version}.pdf"
    return client.post(
        "/documents/upload",
        files={"file": (path.name, path.read_bytes(), "application/pdf")},
    ).json()


def test_active_and_as_of_search_use_the_matching_leave_policy_version(
    versioned_client: TestClient,
) -> None:
    """Current queries use 24-day v2; a 2024 query uses the historical 20-day v1."""
    v1 = upload_leave_version(versioned_client, "v1")
    v2 = upload_leave_version(versioned_client, "v2")

    assert v1["doc_id"] == v2["doc_id"] == "leave_policy"
    assert v1["version"] == "1.0"
    assert v2["version"] == "2.0"
    assert v1["effective_date"] == "2024-01-01"
    assert v2["effective_date"] == "2025-04-01"
    documents = {
        document["version"]: document
        for document in versioned_client.get("/documents").json()
    }
    assert documents["1.0"]["status"] == "superseded"
    assert documents["2.0"]["status"] == "active"

    current = versioned_client.post(
        "/ask",
        json={"query": "How many annual leave days?", "mode": "bm25"},
    ).json()
    historical = versioned_client.post(
        "/ask",
        json={
            "query": "How many annual leave days?",
            "mode": "bm25",
            "as_of_date": "2024-06-01",
        },
    ).json()

    assert "24 paid annual leave days" in current["answer"]
    assert current["citations"][0]["version"] == "2.0"
    assert current["citations"][0]["effective_date"] == "2025-04-01"
    assert "20 paid annual leave days" in historical["answer"]
    assert historical["citations"][0]["version"] == "1.0"
    assert historical["citations"][0]["effective_date"] == "2024-01-01"
    assert "version 2.0, effective 2025-04-01" in current["answer"]
    assert "version 1.0, effective 2024-01-01" in historical["answer"]

    dense_current = main_module.hybrid_search(
        query="annual leave days",
        mode="dense",
        top_k=20,
        filters=None,
        as_of_date=None,
    )
    assert dense_current
    assert all(result.version == "2.0" for result in dense_current)


def test_reuploading_same_version_hash_does_not_change_indexes(
    versioned_client: TestClient,
) -> None:
    """An identical upload is skipped and does not add duplicate indexed chunks."""
    upload_leave_version(versioned_client, "v1")
    v2_result = upload_leave_version(versioned_client, "v2")
    vector_count = vector_store_module.get_vector_store().count()
    bm25_count = len(bm25_index_module.get_bm25_index().chunks)

    path = DOCUMENTS_DIR / "Leave_Policy_v2.pdf"
    duplicate_response = versioned_client.post(
        "/documents/upload",
        files={"file": (path.name, path.read_bytes(), "application/pdf")},
    )
    duplicate = duplicate_response.json()

    assert duplicate_response.status_code == 200
    assert duplicate["skipped"] is True
    assert duplicate["file_hash"] == v2_result["file_hash"]
    assert vector_store_module.get_vector_store().count() == vector_count
    assert len(bm25_index_module.get_bm25_index().chunks) == bm25_count


def test_changes_endpoint_reports_sections_changed_between_versions(
    versioned_client: TestClient,
) -> None:
    """The changes endpoint identifies section-level policy differences."""
    upload_leave_version(versioned_client, "v1")
    upload_leave_version(versioned_client, "v2")

    response = versioned_client.get("/documents/leave_policy/changes")

    assert response.status_code == 200
    data = response.json()
    assert data["from_version"] == "1.0"
    assert data["to_version"] == "2.0"
    assert data["changes"]
    assert all(change["change"] in {"added", "removed", "changed"} for change in data["changes"])
    assert any("Section 5" in change["section"] for change in data["changes"])


def test_indexing_a_new_version_reuses_unchanged_chunk_embeddings(
    tmp_path: Path,
) -> None:
    """Only content not already embedded is sent to the embedding model."""
    embedder = CountingEmbedder()
    store = ChromaVectorStore(
        persist_dir=tmp_path / "vectors",
        collection_name="test_embedding_reuse",
        embedder=embedder,  # type: ignore[arg-type]
    )
    first = make_chunk("v1_common", "Shared policy paragraph.", "1.0")
    store.add_chunks([first])
    assert embedder.embedded_texts == ["Shared policy paragraph."]

    unchanged = make_chunk("v2_common", "Shared policy paragraph.", "2.0")
    changed = make_chunk("v2_changed", "A different policy paragraph.", "2.0")
    store.add_chunks([unchanged, changed])

    assert embedder.embedded_texts == [
        "Shared policy paragraph.",
        "A different policy paragraph.",
    ]
    results = store.search("policy paragraph", k=10, filters={"version": "2.0"})
    assert {result.chunk_id for result in results} == {"v2_common", "v2_changed"}


class CountingEmbedder(FixedEmbedder):
    """Track texts for which a fresh vector was actually generated."""

    def __init__(self) -> None:
        self.embedded_texts: List[str] = []

    def embed_chunks(self, chunks: list) -> List[List[float]]:
        self.embedded_texts.extend(chunk.content for chunk in chunks)
        return super().embed_chunks(chunks)


def make_chunk(chunk_id: str, content: str, version: str):
    """Create one version-tagged chunk for embedding reuse tests."""
    from backend.models.document import DocumentChunk

    return DocumentChunk(
        id=chunk_id,
        content=content,
        source=f"Leave_Policy_v{version}.pdf",
        page=1,
        section="Annual Paid Leave",
        doc_id="leave_policy",
        version=version,
        effective_date="2024-01-01" if version == "1.0" else "2025-04-01",
        file_hash=f"hash-{version}",
    )
