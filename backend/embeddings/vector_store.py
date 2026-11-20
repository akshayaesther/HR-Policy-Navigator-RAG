"""ChromaDB persistent vector store and unified index management for Step 3."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from backend.config import get_settings
from backend.models.document import DocumentChunk
from backend.models.search import SearchResult
from backend.embeddings.embedder import get_embedder, Embedder


class ChromaVectorStore:
    """Persistent ChromaDB vector database manager."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: Optional[str] = None,
        embedder: Optional[Embedder] = None,
    ) -> None:
        settings = get_settings()
        self.persist_dir = persist_dir or settings.get_resolved_vectorstore_dir()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name or settings.chroma_collection_name
        self.embedder = embedder or get_embedder()

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return total number of vector records in collection."""
        return int(self.collection.count())

    def clear(self) -> None:
        """Delete and recreate the current collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Upsert document chunks into ChromaDB with embeddings and metadata."""
        if not chunks:
            return

        ids = [c.id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [c.to_metadata() for c in chunks]

        if embeddings is None:
            embeddings = self.embedder.embed_texts(documents)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks originating from a specified source document."""
        self.collection.delete(where={"source": source})

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Query ChromaDB using dense cosine similarity."""
        if self.count() == 0 or not query.strip():
            return []

        query_embedding = self.embedder.embed_text(query)
        actual_k = min(k, self.count())

        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)

        search_results: List[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for idx, chunk_id in enumerate(ids):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            content = docs[idx] if idx < len(docs) else ""
            dist = distances[idx] if idx < len(distances) else 1.0
            # For cosine distance, similarity is 1.0 - distance
            similarity_score = max(0.0, 1.0 - float(dist))

            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=content,
                    source=str(meta.get("source", "unknown")),
                    page=int(meta.get("page", 1)),
                    category=str(meta.get("category", "hr_policy")),
                    section=str(meta.get("section", "General")),
                    chunk_type=str(meta.get("chunk_type", "text")),
                    extraction_method=str(meta.get("extraction_method", "text")),
                    score=round(similarity_score, 4),
                    rank=idx + 1,
                    metadata=meta,
                )
            )

        return search_results


# Module-level singleton instance
_vector_store_instance: Optional[ChromaVectorStore] = None


def get_vector_store() -> ChromaVectorStore:
    """Return singleton instance of ChromaVectorStore."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = ChromaVectorStore()
    return _vector_store_instance


# Unified Indexing and Retrieval API required for Step 3
def index_chunks(chunks: List[DocumentChunk]) -> None:
    """Index chunks into both ChromaDB vector store and BM25 index."""
    if not chunks:
        return

    from backend.retrieval.bm25_index import get_bm25_index

    # Index into ChromaDB
    vstore = get_vector_store()
    vstore.add_chunks(chunks)

    # Index into BM25
    bm25 = get_bm25_index()
    bm25.index_chunks(chunks, persist=True)


def delete_by_source(source: str) -> None:
    """Remove all chunks of a source document from both ChromaDB and BM25 index."""
    from backend.retrieval.bm25_index import get_bm25_index

    vstore = get_vector_store()
    vstore.delete_by_source(source)

    bm25 = get_bm25_index()
    bm25.delete_by_source(source, persist=True)


def search_dense(
    query: str,
    k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[SearchResult]:
    """Execute dense semantic vector search via ChromaDB."""
    vstore = get_vector_store()
    return vstore.search(query=query, k=k, filters=filters)
