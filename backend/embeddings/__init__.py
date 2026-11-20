"""Embeddings package."""

from backend.embeddings.embedder import Embedder, get_embedder
from backend.embeddings.vector_store import (
    ChromaVectorStore,
    get_vector_store,
    index_chunks,
    delete_by_source,
    search_dense,
)

__all__ = [
    "Embedder",
    "get_embedder",
    "ChromaVectorStore",
    "get_vector_store",
    "index_chunks",
    "delete_by_source",
    "search_dense",
]
