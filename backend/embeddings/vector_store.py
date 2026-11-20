"""ChromaDB persistent vector store and unified index management for Step 3."""

from pathlib import Path
import hashlib
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
            content_hashes = list(dict.fromkeys(str(meta["content_hash"]) for meta in metadatas))
            existing = self.collection.get(
                where={"content_hash": {"$in": content_hashes}},
                include=["metadatas", "embeddings"],
            )
            embeddings_by_hash: Dict[str, List[float]] = {
                str(metadata["content_hash"]): embedding.tolist()
                for metadata, embedding in zip(existing["metadatas"], existing["embeddings"])
            }

            missing_by_hash = {
                str(metadata["content_hash"]): chunk
                for chunk, metadata in zip(chunks, metadatas)
                if metadata["content_hash"] not in embeddings_by_hash
            }
            missing_chunks = list(missing_by_hash.values())
            new_embeddings = self.embedder.embed_chunks(missing_chunks)
            for content_hash, embedding in zip(missing_by_hash, new_embeddings):
                embeddings_by_hash[content_hash] = embedding
            embeddings = [
                embeddings_by_hash[str(metadata["content_hash"])]
                for metadata in metadatas
            ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks originating from a specified source document."""
        self.collection.delete(where={"source": source})

    def update_policy_status(
        self,
        doc_id: str,
        version: Optional[str],
        status: str,
    ) -> None:
        """Update lifecycle metadata without changing stored document embeddings."""
        conditions: List[Dict[str, Any]] = [{"doc_id": doc_id}]
        if version is not None:
            conditions.append({"version": version})
        records = self.collection.get(
            where={"$and": conditions} if len(conditions) > 1 else conditions[0],
            include=["metadatas"],
        )
        metadatas = []
        for metadata in records["metadatas"]:
            updated = dict(metadata)
            updated["status"] = status
            metadatas.append(updated)
        if records["ids"]:
            self.collection.update(ids=records["ids"], metadatas=metadatas)

    def update_document_metadata(
        self,
        source: str,
        *,
        doc_id: str,
        version: Optional[str],
        effective_date: Optional[str],
        status: str,
        file_hash: str,
    ) -> int:
        """Backfill policy metadata in place, preserving each existing embedding."""
        records = self.collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
        updated_metadatas = []
        for document, metadata in zip(records["documents"], records["metadatas"]):
            updated = dict(metadata)
            updated.update(
                doc_id=doc_id,
                status=status,
                file_hash=file_hash,
                content_hash=hashlib.sha256((document or "").encode("utf-8")).hexdigest(),
            )
            if version is not None:
                updated["version"] = version
            else:
                updated.pop("version", None)
            if effective_date is not None:
                updated["effective_date"] = effective_date
            else:
                updated.pop("effective_date", None)
            updated_metadatas.append(updated)
        if records["ids"]:
            self.collection.update(ids=records["ids"], metadatas=updated_metadatas)
        return len(records["ids"])

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
                    doc_id=meta.get("doc_id"),
                    version=meta.get("version"),
                    effective_date=meta.get("effective_date"),
                    status=meta.get("status"),
                    file_hash=meta.get("file_hash"),
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

    vstore = get_vector_store()
    bm25 = get_bm25_index()

    families = {chunk.doc_id for chunk in chunks}
    for doc_id in families:
        family_chunks = [
            chunk for chunk in [*bm25.chunks, *chunks] if chunk.doc_id == doc_id
        ]
        latest_version = max(
            (version_key(chunk.version, chunk.effective_date) for chunk in family_chunks),
            default=((), ""),
        )
        for chunk in chunks:
            if chunk.doc_id == doc_id:
                chunk.status = (
                    "active"
                    if version_key(chunk.version, chunk.effective_date) == latest_version
                    else "superseded"
                )
        for old_version in {
            chunk.version
            for chunk in family_chunks
            if chunk in bm25.chunks
        }:
            status = (
                "active"
                if version_key(old_version, next(
                    (
                        chunk.effective_date
                        for chunk in family_chunks
                        if chunk.version == old_version
                    ),
                    None,
                )) == latest_version
                else "superseded"
            )
            if any(chunk.doc_id == doc_id and chunk.version == old_version for chunk in bm25.chunks):
                bm25.update_policy_status(doc_id, old_version, status, persist=False)
                vstore.update_policy_status(doc_id, old_version, status)
        bm25.save()

    vstore.add_chunks(chunks)
    bm25.index_chunks(chunks, persist=True)


def delete_by_source(source: str) -> None:
    """Remove all chunks of a source document from both ChromaDB and BM25 index."""
    from backend.retrieval.bm25_index import get_bm25_index

    vstore = get_vector_store()
    vstore.delete_by_source(source)

    bm25 = get_bm25_index()
    bm25.delete_by_source(source, persist=True)


def update_policy_status(doc_id: str, version: Optional[str], status: str) -> None:
    """Update a policy version's status in both retrieval indexes."""
    from backend.retrieval.bm25_index import get_bm25_index

    get_vector_store().update_policy_status(doc_id, version, status)
    get_bm25_index().update_policy_status(doc_id, version, status, persist=True)


def update_document_metadata(
    source: str,
    *,
    doc_id: str,
    version: Optional[str],
    effective_date: Optional[str],
    status: str,
    file_hash: str,
) -> int:
    """Backfill policy metadata in both indexes without embedding or tokenizing."""
    updated = get_vector_store().update_document_metadata(
        source,
        doc_id=doc_id,
        version=version,
        effective_date=effective_date,
        status=status,
        file_hash=file_hash,
    )
    get_bm25_index().update_document_metadata(
        source,
        doc_id=doc_id,
        version=version,
        effective_date=effective_date,
        status=status,
        file_hash=file_hash,
        persist=True,
    )
    return updated


def version_key(
    version: Optional[str],
    effective_date: Optional[str] = None,
) -> tuple[tuple[int, ...], str]:
    """Compare numeric policy versions, using effective date to break ties."""
    parts = tuple(int(part) for part in version.split(".") if part.isdigit()) if version else ()
    return parts, effective_date or ""


def search_dense(
    query: str,
    k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[SearchResult]:
    """Execute dense semantic vector search via ChromaDB."""
    vstore = get_vector_store()
    return vstore.search(query=query, k=k, filters=filters)
