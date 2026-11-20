"""BM25 lexical search index with disk persistence."""

import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Plus

from backend.config import get_settings
from backend.models.document import DocumentChunk
from backend.models.search import SearchResult


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """Persistent BM25 lexical keyword search index."""

    def __init__(self, index_path: Optional[Path] = None) -> None:
        settings = get_settings()
        self.index_path = index_path or (settings.get_resolved_vectorstore_dir() / "bm25_index.pkl")
        self.chunks: List[DocumentChunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.bm25: Optional[BM25Plus] = None

        # Auto-load persisted index if it exists on disk
        if self.index_path.exists():
            self.load()

    def is_indexed(self) -> bool:
        """Return True if index contains documents."""
        return self.bm25 is not None and len(self.chunks) > 0

    def index_chunks(self, chunks: List[DocumentChunk], persist: bool = True) -> None:
        """Build or append chunks into the BM25 index and optionally save to disk."""
        if not chunks:
            return

        # Deduplicate against existing chunks by ID
        existing_ids = {c.id for c in self.chunks}
        new_chunks = [c for c in chunks if c.id not in existing_ids]

        if not new_chunks and self.bm25 is not None:
            return

        self.chunks.extend(new_chunks)
        self.corpus_tokens = [tokenize(c.content) for c in self.chunks]
        self.bm25 = BM25Plus(self.corpus_tokens)

        if persist:
            self.save()

    def delete_by_source(self, source: str, persist: bool = True) -> int:
        """Remove all chunks associated with a given source document and rebuild index."""
        initial_count = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.source != source]
        deleted_count = initial_count - len(self.chunks)

        if deleted_count > 0:
            if self.chunks:
                self.corpus_tokens = [tokenize(c.content) for c in self.chunks]
                self.bm25 = BM25Plus(self.corpus_tokens)
            else:
                self.corpus_tokens = []
                self.bm25 = None

            if persist:
                self.save()

        return deleted_count

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Query BM25 index and return ranked SearchResult items."""
        if not self.is_indexed() or not query.strip():
            return []

        assert self.bm25 is not None
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        scored_candidates = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]

            # Filter evaluation
            if filters:
                meta = chunk.to_metadata()
                match = all(meta.get(key) == val for key, val in filters.items())
                if not match:
                    continue

            doc_token_set = set(self.corpus_tokens[idx])
            if score > 0.0 or any(t in doc_token_set for t in query_tokens):
                scored_candidates.append((float(score), chunk))

        # Sort descending by BM25 score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_candidates[:k]

        results: List[SearchResult] = []
        for rank, (score, chunk) in enumerate(top_candidates, start=1):
            results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    source=chunk.source,
                    page=chunk.page,
                    category=chunk.category,
                    section=chunk.section,
                    chunk_type=chunk.chunk_type,
                    extraction_method=chunk.extraction_method,
                    score=round(score, 4),
                    rank=rank,
                    metadata=chunk.to_metadata(),
                )
            )

        return results

    def save(self, path: Optional[Path] = None) -> None:
        """Serialize index state to disk using pickle."""
        target_path = path or self.index_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunks": [c.model_dump() for c in self.chunks],
            "corpus_tokens": self.corpus_tokens,
        }
        with open(target_path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: Optional[Path] = None) -> bool:
        """Deserialize index state from disk."""
        target_path = path or self.index_path
        if not target_path.exists():
            return False

        try:
            with open(target_path, "rb") as f:
                payload = pickle.load(f)

            raw_chunks = payload.get("chunks", [])
            self.chunks = [DocumentChunk(**item) for item in raw_chunks]
            self.corpus_tokens = payload.get("corpus_tokens", [])
            if self.corpus_tokens:
                self.bm25 = BM25Plus(self.corpus_tokens)
            else:
                self.bm25 = None
            return True
        except Exception:
            return False


# Singleton instance
_bm25_instance: Optional[BM25Index] = None


def get_bm25_index() -> BM25Index:
    """Return singleton instance of BM25Index."""
    global _bm25_instance
    if _bm25_instance is None:
        _bm25_instance = BM25Index()
    return _bm25_instance


def search_bm25(
    query: str,
    k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[SearchResult]:
    """Execute BM25 keyword search against the global index."""
    index = get_bm25_index()
    return index.search(query=query, k=k, filters=filters)
