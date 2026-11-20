"""Hybrid search engine combining dense vector search and BM25 keyword search.

Implements hand-written Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking
for high-precision enterprise document retrieval.
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import date

from backend.config import get_settings
from backend.models.document import DocumentChunk
from backend.models.search import SearchResult, RetrievalMode
from backend.embeddings.vector_store import ChromaVectorStore, get_vector_store, version_key
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
                doc_id=original.doc_id,
                version=original.version,
                effective_date=original.effective_date,
                status=original.status,
                file_hash=original.file_hash,
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
        as_of_date: Optional[date] = None,
    ) -> List[SearchResult]:
        """Execute search under specified retrieval mode.

        Modes:
            - 'dense': Semantic dense vector retrieval via ChromaDB.
            - 'bm25': Lexical keyword retrieval via the persistent BM25Plus index.
            - 'hybrid': Reciprocal Rank Fusion of dense and BM25 results (k=60).
            - 'hybrid_rerank': RRF of top 20 candidates followed by Cross-Encoder reranking.

        For hybrid_rerank, top_k must be between 1 and 20 and defaults to five.
        """
        if mode == "hybrid_rerank" and top_k is not None and not 1 <= top_k <= 20:
            raise ValueError("hybrid_rerank top_k must be between 1 and 20")
        version_filter = self._policy_version_filter(as_of_date)
        if not version_filter:
            return []
        filters = self._combine_filters(filters, version_filter)

        target_top_k = top_k if top_k is not None else (
            5 if mode == "hybrid_rerank" else self.settings.vector_top_k
        )

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

            # 3. Rerank the top 20 candidates and return the requested number.
            reranked = self.reranker.rerank(
                query=query,
                candidates=top_fused_candidates,
                top_k=target_top_k,
            )
            return reranked

        else:
            raise ValueError(
                f"Unknown retrieval mode '{mode}'. Supported modes: 'hybrid', 'dense', 'bm25', 'hybrid_rerank'"
            )

    def _policy_version_filter(self, as_of_date: Optional[date]) -> Dict[str, Any]:
        """Select active versions or the latest version effective on a requested date."""
        if not self.bm25_index.chunks:
            return {"status": "active"}

        versions_by_family: Dict[str, List[DocumentChunk]] = defaultdict(list)
        for chunk in self.bm25_index.chunks:
            if chunk.doc_id:
                versions_by_family[chunk.doc_id].append(chunk)

        selected: List[Dict[str, Any]] = []
        for doc_id, chunks in versions_by_family.items():
            versions: Dict[tuple[str, Optional[str]], DocumentChunk] = {}
            for chunk in chunks:
                if as_of_date is None:
                    if chunk.status != "active":
                        continue
                elif not chunk.effective_date or chunk.effective_date > as_of_date.isoformat():
                    continue
                key = (chunk.version or "", chunk.effective_date)
                versions[key] = chunk

            if as_of_date is None:
                selected.extend(
                    {"$and": [{"doc_id": doc_id}, {"version": version}]}
                    if version
                    else {"doc_id": doc_id}
                    for version, _ in versions
                )
            elif versions:
                selected_version, _ = max(
                    versions,
                    key=lambda item: version_key(item[0], item[1]),
                )
                selected.append(
                    {"$and": [{"doc_id": doc_id}, {"version": selected_version}]}
                    if selected_version
                    else {"doc_id": doc_id}
                )

        if not selected:
            return {}
        return selected[0] if len(selected) == 1 else {"$or": selected}

    @staticmethod
    def _combine_filters(
        filters: Optional[Dict[str, Any]],
        version_filter: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply user metadata constraints together with policy version selection."""
        if filters and version_filter:
            return {"$and": [filters, version_filter]}
        return filters or version_filter


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
    as_of_date: Optional[date] = None,
) -> List[SearchResult]:
    """Unified entrypoint for policy retrieval across comparison modes.

    Args:
        query: Natural language user query.
        mode: Retrieval strategy ('hybrid', 'dense', 'bm25', 'hybrid_rerank').
        filters: Optional metadata filters (e.g. {'category': 'hr_policy', 'source': 'WFH_Policy.pdf'}).
        top_k: Optional result count. hybrid_rerank accepts values from 1 through 20
            and defaults to 5.
        as_of_date: Select the latest policy versions that were effective on this date.

    Returns:
        List[SearchResult] ordered by relevance.
    """
    engine = get_search_engine()
    return engine.search(
        query=query,
        mode=mode,
        filters=filters,
        top_k=top_k,
        as_of_date=as_of_date,
    )
