"""Application configuration module.

Loads all system settings, model parameters, retrieval thresholds,
and path definitions from environment variables or .env file.
"""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration parameters for the HR Policy Navigator RAG system."""

    # Server Configuration
    app_name: str = "Enterprise HR & Policy Navigator"
    app_env: str = "development"
    debug: bool = True
    host: str = "127.0.0.1"
    port: int = 8000

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    documents_dir: Path = Path("backend/data/documents")
    vectorstore_dir: Path = Path("backend/data/vectorstore")

    # Ingestion & Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Embedding Configuration
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # ChromaDB Vector Store
    chroma_collection_name: str = "hr_policy_navigator"

    # Hybrid Search & Reranking
    bm25_top_k: int = 10
    vector_top_k: int = 10
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    relevance_score_threshold: float = 0.1

    # Generation (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_temperature: float = 0.0
    ollama_request_timeout: float = 60.0

    # Fallback / Refusal Response
    refusal_response: str = "I cannot answer based on internal documentation."

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def get_resolved_documents_dir(self) -> Path:
        """Return absolute path to documents directory."""
        if self.documents_dir.is_absolute():
            return self.documents_dir
        return self.base_dir / self.documents_dir

    def get_resolved_vectorstore_dir(self) -> Path:
        """Return absolute path to vectorstore directory."""
        if self.vectorstore_dir.is_absolute():
            return self.vectorstore_dir
        return self.base_dir / self.vectorstore_dir


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton instance of system settings."""
    return Settings()
