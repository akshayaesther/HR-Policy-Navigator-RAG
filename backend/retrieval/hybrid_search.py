"""Hybrid search engine combining dense vector search and BM25 keyword search.

Implements hand-written Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking
for high-precision enterprise document retrieval.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict

from backend.config import get_settings
from backend.models.document import DocumentChunk
from backend.models.search import SearchResult, RetrievalMode
from backend.embeddings.vector_store import ChromaVectorStore, get_vector_store
from backend.retrieval.bm25_index import BM25Index, get_bm25_index
from backend.retrieval.reranker import CrossEncoderReranker, get_reranker


def reciprocal_rank_fusion(
    ranking_lists: List[List[SearchResult]],
    k: int = 60,
) -> List[SearchResult]:
    """Combine ranked result lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d in D) = SUM_{m in M} [ 1 / (k + r_m(d)) ]

    Explanation:
        - D is the set of all unique documents across all retrieved lists.
        - M is the set of retrieval systems (e.g., [DenseVectorSearch, BM25KeywordSearch]).
        - r_m(d) is the 1-based rank position of document d in ranking list m.
        - k is a smoothing constant (default: 60) that prevents top-ranked items from
          disproportionately dominating the score while giving an exponential boost
          to documents that appear near the top across multiple independent search algorithms.
        - If document d does not appear in ranking list m, its contribution from system m is 0.

    Args:
        ranking_lists: List of candidate result lists (e.g. [dense_results, bm25_results]).
        k: Smoothing parameter (default: 60).

    Returns:
        List[SearchResult] sorted in descending order of aggregated RRF score.
    """
    # Map chunk_id to its accumulated RRF score
    rrf_scores: Dict[str, float] = defaultdict(float)

    # Store first seen SearchResult object to preserve chunk content and metadata
    doc_registry: Dict[str, SearchResult] = {}

    for ranked_list in ranking_lists:
        for rank_zero_based, item in enumerate(ranked_list):
            chunk_id = item.chunk_id
            rank_one_based = rank_zero_based + 1

            # Reciprocal rank formula: 1 / (k + rank)
            rrf_contribution = 1.0 / (k + rank_one_based)
            rrf_scores[chunk_id] += rrf_contribution

            if chunk_id not in doc_registry:
                doc_registry[chunk_id] = item

    # Sort documents by descending fused RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    fused_results: List[SearchResult] = []
    for rank_idx, (chunk_id, score) in enumerate(sorted_items, start=1):
        original = doc_registry[chunk_id]
        fused_results.append(
            SearchResult(
                chunk_id=original.chunk_id,
                content=original.content,
                source=original.source,
                page=original.page,
                category=original.category,
                section=original.section,
                chunk_type=original.chunk_type,
                extraction_method=original.extraction_method,
                score=round(score, 6),
                rank=rank_idx,
                metadata=original.metadata,
            )
        )

    return fused_results


class HybridSearchEngine:
    """Orchestrates multi-modal retrieval across dense vectors, BM25, and cross-encoders."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.settings = get_settings()
        self.vector_store = vector_store or get_vector_store()
        self.bm25_index = bm25_index or get_bm25_index()
        self.reranker = reranker or get_reranker()

    def index_documents(self, chunks: List[DocumentChunk]) -> None:
        """Index chunks into both ChromaDB vector store and BM25 keyword index."""
        if not chunks:
            return

        # Index into ChromaDB
        self.vector_store.add_chunks(chunks)

        # Index into BM25
        self.bm25_index.index_chunks(chunks)

    def search(
        self,
        query: str,
        mode: RetrievalMode = "hybrid_rerank",
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Execute search under specified retrieval mode.

        Modes:
            - 'dense': Semantic dense vector retrieval via ChromaDB.
            - 'bm25': Lexical keyword retrieval via the persistent BM25Plus index.
            - 'hybrid': Reciprocal Rank Fusion of dense and BM25 results (k=60).
            - 'hybrid_rerank': RRF of top 20 candidates followed by Cross-Encoder reranking
                               returning the top 5 highest-scoring results.

        top_k applies to dense, bm25, and hybrid modes. The hybrid_rerank mode always
        returns up to five results, as required by the retrieval pipeline.
        """
        if mode == "hybrid_rerank" and top_k not in (None, 5):
            raise ValueError("hybrid_rerank always returns top 5; top_k must be omitted or 5")

        target_top_k = top_k or self.settings.vector_top_k

        if mode == "dense":
            return self.vector_store.search(
                query=query,
                k=target_top_k,
                filters=filters,
            )

        elif mode == "bm25":
            return self.bm25_index.search(
                query=query,
                k=target_top_k,
                filters=filters,
            )

        elif mode == "hybrid":
            dense_candidates = self.vector_store.search(
                query=query,
                k=20,
                filters=filters,
            )
            bm25_candidates = self.bm25_index.search(
                query=query,
                k=20,
                filters=filters,
            )
            fused = reciprocal_rank_fusion(
                ranking_lists=[dense_candidates, bm25_candidates],
                k=60,
            )
            return fused[:target_top_k]

        elif mode == "hybrid_rerank":
            # 1. Fetch top 20 candidates from dense and BM25
            dense_candidates = self.vector_store.search(
                query=query,
                k=20,
                filters=filters,
            )
            bm25_candidates = self.bm25_index.search(
                query=query,
                k=20,
                filters=filters,
            )

            # 2. Fuse candidate pools using RRF (k=60)
            fused = reciprocal_rank_fusion(
                ranking_lists=[dense_candidates, bm25_candidates],
                k=60,
            )
            top_fused_candidates = fused[:20]

            # 3. Rerank the top 20 candidates with cross-encoder and return top 5
            reranked = self.reranker.rerank(
                query=query,
                candidates=top_fused_candidates,
                top_k=5,
            )
            return reranked

        else:
            raise ValueError(
                f"Unknown retrieval mode '{mode}'. Supported modes: 'hybrid', 'dense', 'bm25', 'hybrid_rerank'"
            )


# Module-level singleton
_engine_instance: Optional[HybridSearchEngine] = None


def get_search_engine() -> HybridSearchEngine:
    """Return singleton instance of HybridSearchEngine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = HybridSearchEngine()
    return _engine_instance


def hybrid_search(
    query: str,
    mode: RetrievalMode = "hybrid_rerank",
    filters: Optional[Dict[str, Any]] = None,
    top_k: Optional[int] = None,
) -> List[SearchResult]:
    """Unified entrypoint for policy retrieval across comparison modes.

    Args:
        query: Natural language user query.
        mode: Retrieval strategy ('hybrid', 'dense', 'bm25', 'hybrid_rerank').
        filters: Optional metadata filters (e.g. {'category': 'hr_policy', 'source': 'WFH_Policy.pdf'}).
        top_k: Optional result count for dense, bm25, and hybrid modes. For hybrid_rerank,
            omit this or set it to 5.

    Returns:
        List[SearchResult] ordered by relevance.
    """
    engine = get_search_engine()
    return engine.search(query=query, mode=mode, filters=filters, top_k=top_k)
