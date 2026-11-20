"""Cross-encoder reranker for high-precision semantic relevance scoring.

Wraps cross-encoder/ms-marco-MiniLM-L-6-v2 to rescore (query, document)
pairs after initial retrieval.
"""

from typing import List, Optional
from functools import lru_cache
from sentence_transformers import CrossEncoder

from backend.config import get_settings
from backend.models.search import SearchResult


class CrossEncoderReranker:
    """Reranks search candidates using a deep cross-encoder model."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model_name
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy loader: download/cache the model on first use."""
        if self._model is None:
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Score (query, passage) pairs and return top_k by cross-encoder score.

        Args:
            query:      The user's query string.
            candidates: Candidate SearchResult objects from prior fusion step.
            top_k:      How many top results to return.

        Returns:
            List[SearchResult] re-sorted by cross-encoder score, length <= top_k.
        """
        if not candidates:
            return []

        # Build (query, document) pairs for batch inference
        pairs = [[query, c.content] for c in candidates]
        raw_scores = self.model.predict(pairs)

        scored = list(zip(raw_scores, candidates))
        scored.sort(key=lambda x: x[0], reverse=True)

        reranked: List[SearchResult] = []
        for rank, (score, cand) in enumerate(scored[:top_k], start=1):
            reranked.append(
                SearchResult(
                    chunk_id=cand.chunk_id,
                    content=cand.content,
                    source=cand.source,
                    page=cand.page,
                    category=cand.category,
                    section=cand.section,
                    chunk_type=cand.chunk_type,
                    extraction_method=cand.extraction_method,
                    score=round(float(score), 4),
                    rank=rank,
                    metadata=cand.metadata,
                )
            )
        return reranked


@lru_cache()
def get_reranker() -> CrossEncoderReranker:
    """Return cached singleton CrossEncoderReranker."""
    return CrossEncoderReranker()
