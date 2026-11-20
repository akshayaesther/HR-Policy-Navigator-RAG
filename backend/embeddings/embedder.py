"""Embedder module using sentence-transformers all-MiniLM-L6-v2."""

from typing import List, Optional
from functools import lru_cache
from sentence_transformers import SentenceTransformer

from backend.config import get_settings
from backend.models.document import DocumentChunk


class Embedder:
    """Manages text embedding generation via sentence-transformers."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self.device = device or settings.embedding_device
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer model instance."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def get_dimension(self) -> int:
        """Return the output embedding dimension."""
        return int(self.model.get_sentence_embedding_dimension())

    def embed_text(self, text: str) -> List[float]:
        """Generate a normalized dense vector embedding for a query string."""
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate normalized dense vector embeddings for a batch of strings."""
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def embed_chunks(self, chunks: List[DocumentChunk], batch_size: int = 32) -> List[List[float]]:
        """Generate normalized embeddings directly from DocumentChunk instances."""
        texts = [chunk.content for chunk in chunks]
        return self.embed_texts(texts, batch_size=batch_size)


@lru_cache()
def get_embedder() -> Embedder:
    """Return cached singleton instance of Embedder."""
    return Embedder()
