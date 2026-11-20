"""Tests for Reciprocal Rank Fusion, BM25, ChromaDB vector search, and Cross-Encoder reranking."""

import pytest
from pathlib import Path
from backend.models.document import DocumentChunk
from backend.models.search import SearchResult
from backend.retrieval.hybrid_search import (
    reciprocal_rank_fusion,
    HybridSearchEngine,
    hybrid_search,
)
from backend.embeddings.vector_store import ChromaVectorStore
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.reranker import CrossEncoderReranker


def make_dummy_result(chunk_id: str, rank: int, score: float, content: str = "Test content") -> SearchResult:
    """Helper to build dummy SearchResult objects for testing."""
    return SearchResult(
        chunk_id=chunk_id,
        content=f"{content} for {chunk_id}",
        source="test_policy.pdf",
        page=1,
        category="hr_policy",
        section="Section 1",
        chunk_type="text",
        extraction_method="text",
        score=score,
        rank=rank,
        metadata={"source": "test_policy.pdf", "page": 1},
    )


def test_rrf_hand_crafted_example() -> None:
    """Verify Reciprocal Rank Fusion correctness against hand-computed expected values.

    Hand-made scenario:
        Dense ranking:
            Rank 1: Doc A (id: 'doc_a') -> 1/(60 + 1) = 1/61
            Rank 2: Doc B (id: 'doc_b') -> 1/(60 + 2) = 1/62
            Rank 3: Doc C (id: 'doc_c') -> 1/(60 + 3) = 1/63

        BM25 ranking:
            Rank 1: Doc B (id: 'doc_b') -> 1/(60 + 1) = 1/61
            Rank 2: Doc D (id: 'doc_d') -> 1/(60 + 2) = 1/62
            Rank 3: Doc A (id: 'doc_a') -> 1/(60 + 3) = 1/63

        Theoretical RRF scores (k=60):
            Doc B = 1/62 + 1/61 = 0.0161290 + 0.0163934 = 0.0325224
            Doc A = 1/61 + 1/63 = 0.0163934 + 0.0158730 = 0.0322664
            Doc D = 1/62        = 0.0161290             = 0.0161290
            Doc C = 1/63        = 0.0158730             = 0.0158730

        Expected Ranking:
            1. Doc B (0.032522)
            2. Doc A (0.032266)
            3. Doc D (0.016129)
            4. Doc C (0.015873)
    """
    dense_list = [
        make_dummy_result("doc_a", rank=1, score=0.95),
        make_dummy_result("doc_b", rank=2, score=0.88),
        make_dummy_result("doc_c", rank=3, score=0.72),
    ]

    bm25_list = [
        make_dummy_result("doc_b", rank=1, score=12.4),
        make_dummy_result("doc_d", rank=2, score=8.1),
        make_dummy_result("doc_a", rank=3, score=5.6),
    ]

    fused = reciprocal_rank_fusion([dense_list, bm25_list], k=60)

    assert len(fused) == 4

    # 1. Check exact ordering
    assert fused[0].chunk_id == "doc_b"
    assert fused[1].chunk_id == "doc_a"
    assert fused[2].chunk_id == "doc_d"
    assert fused[3].chunk_id == "doc_c"

    # 2. Check 1-based ranks
    assert [f.rank for f in fused] == [1, 2, 3, 4]

    # 3. Check exact computed scores
    expected_b = (1.0 / 62.0) + (1.0 / 61.0)
    expected_a = (1.0 / 61.0) + (1.0 / 63.0)
    expected_d = 1.0 / 62.0
    expected_c = 1.0 / 63.0

    assert fused[0].score == pytest.approx(expected_b, rel=1e-4)
    assert fused[1].score == pytest.approx(expected_a, rel=1e-4)
    assert fused[2].score == pytest.approx(expected_d, rel=1e-4)
    assert fused[3].score == pytest.approx(expected_c, rel=1e-4)


def test_rrf_empty_inputs() -> None:
    """Verify that empty candidate lists produce an empty result."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


@pytest.fixture(scope="module")
def search_engine(tmp_path_factory: pytest.TempPathFactory) -> HybridSearchEngine:
    """Fixture providing an isolated HybridSearchEngine with populated test chunks."""
    temp_vector_dir = tmp_path_factory.mktemp("test_vectorstore")
    vstore = ChromaVectorStore(
        persist_dir=temp_vector_dir,
        collection_name="test_collection_hybrid",
    )
    bm25 = BM25Index(index_path=temp_vector_dir / "bm25_index.pkl")
    reranker = CrossEncoderReranker()

    chunks = [
        DocumentChunk(
            id="leave_p1_c1",
            content="All full-time employees are entitled to 20 paid annual leave days per year.",
            source="Leave_Policy_v1.pdf",
            page=1,
            category="hr_policy",
            section="Section 2: Annual Leave",
            chunk_type="text",
            extraction_method="text",
            quality=0.95,
        ),
        DocumentChunk(
            id="leave_p2_c1",
            content="A maximum of 5 unused annual leave days may be carried over into the next calendar year.",
            source="Leave_Policy_v1.pdf",
            page=2,
            category="hr_policy",
            section="Section 5: Carryover",
            chunk_type="text",
            extraction_method="text",
            quality=0.94,
        ),
        DocumentChunk(
            id="wfh_p2_c1",
            content="Unauthorized out-of-state remote work shall be deducted from accrued paid annual leave.",
            source="WFH_Policy.pdf",
            page=2,
            category="hr_policy",
            section="Section 4: Remote Overrides",
            chunk_type="text",
            extraction_method="text",
            quality=0.92,
        ),
        DocumentChunk(
            id="conduct_p3_c1",
            content="Nominal business gifts under fifty dollars may be accepted if customary.",
            source="Code_of_Conduct.pdf",
            page=3,
            category="hr_policy",
            section="Section 5: Anti-Bribery",
            chunk_type="text",
            extraction_method="text",
            quality=0.96,
        ),
    ]

    engine = HybridSearchEngine(vector_store=vstore, bm25_index=bm25, reranker=reranker)
    engine.index_documents(chunks)
    return engine


def test_hybrid_search_all_modes(search_engine: HybridSearchEngine) -> None:
    """Verify search across dense, bm25, hybrid, and hybrid_rerank modes."""
    query = "How many paid annual leave days do employees get?"

    # 1. Dense mode
    dense_results = search_engine.search(query, mode="dense", top_k=3)
    assert len(dense_results) > 0
    assert any("20 paid annual leave" in r.content for r in dense_results)

    # 2. BM25 mode
    bm25_results = search_engine.search(query, mode="bm25", top_k=3)
    assert len(bm25_results) > 0
    assert any("annual leave" in r.content.lower() for r in bm25_results)

    # 3. Hybrid (RRF) mode
    hybrid_results = search_engine.search(query, mode="hybrid", top_k=3)
    assert len(hybrid_results) > 0
    assert hybrid_results[0].rank == 1
    assert hybrid_results[0].score > 0.0

    # 4. Hybrid + Rerank mode
    rerank_results = search_engine.search(query, mode="hybrid_rerank")
    assert len(rerank_results) > 0
    assert rerank_results[0].rank == 1
    # Cross-encoder should place the exact entitlement chunk at rank 1
    assert "20 paid annual leave" in rerank_results[0].content


def test_hybrid_search_metadata_filter(search_engine: HybridSearchEngine) -> None:
    """Verify that metadata filtering restricts results to specific source documents."""
    query = "leave and remote work"
    results = search_engine.search(
        query=query,
        mode="dense",
        filters={"source": "WFH_Policy.pdf"},
        top_k=5,
    )
    assert len(results) > 0
    for r in results:
        assert r.source == "WFH_Policy.pdf"


def test_hybrid_rerank_respects_top_k_between_one_and_twenty() -> None:
    """Verify reranking accepts requested result counts from one through twenty."""

    class CandidateRetriever:
        def __init__(self, candidates: list[SearchResult]) -> None:
            self.candidates = candidates
            self.requested_k: list[int] = []

        def search(self, query: str, k: int, filters: dict | None = None) -> list[SearchResult]:
            self.requested_k.append(k)
            return self.candidates[:k]

    class RecordingReranker:
        def __init__(self) -> None:
            self.candidate_count = 0
            self.top_k_values: list[int] = []

        def rerank(
            self,
            query: str,
            candidates: list[SearchResult],
            top_k: int = 5,
        ) -> list[SearchResult]:
            self.candidate_count = len(candidates)
            self.top_k_values.append(top_k)
            return candidates[:top_k]

    candidates = [
        make_dummy_result(f"chunk_{index}", rank=index + 1, score=1.0)
        for index in range(20)
    ]
    dense = CandidateRetriever(candidates)
    bm25 = CandidateRetriever(candidates)
    reranker = RecordingReranker()
    engine = HybridSearchEngine(
        vector_store=dense,  # type: ignore[arg-type]
        bm25_index=bm25,  # type: ignore[arg-type]
        reranker=reranker,  # type: ignore[arg-type]
    )

    results = engine.search("policy query", mode="hybrid_rerank", top_k=3)
    assert reranker.candidate_count == 20
    assert reranker.top_k_values == [3]
    assert len(results) == 3

    default_results = engine.search("policy query", mode="hybrid_rerank")
    assert reranker.top_k_values == [3, 5]
    assert len(default_results) == 5
    with pytest.raises(ValueError, match="between 1 and 20"):
        engine.search("policy query", mode="hybrid_rerank", top_k=21)
    with pytest.raises(ValueError, match="between 1 and 20"):
        engine.search("policy query", mode="hybrid_rerank", top_k=0)


def test_hybrid_search_invalid_mode_raises(search_engine: HybridSearchEngine) -> None:
    """Verify that invalid search mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        search_engine.search("test", mode="nonexistent_mode")  # type: ignore
