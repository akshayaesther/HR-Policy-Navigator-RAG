"""CLI script to compare retrieval modes across Dense, BM25, Hybrid, and Hybrid+Rerank."""

import sys
import time
from typing import List

from backend.config import get_settings
from backend.ingestion.document_loader import DocumentLoader
from backend.models.search import SearchResult, RetrievalMode
from backend.retrieval.hybrid_search import get_search_engine, hybrid_search


def ensure_index_populated() -> None:
    """Check if search index has documents; if empty, ingest backend/data/documents."""
    engine = get_search_engine()
    settings = get_settings()

    if engine.vector_store.count() > 0 and engine.bm25_index.is_indexed():
        return

    print("Index is empty. Ingesting documents from backend/data/documents...")
    docs_dir = settings.get_resolved_documents_dir()
    loader = DocumentLoader()

    all_chunks = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        chunks, _ = loader.load_and_chunk(pdf_path)
        all_chunks.extend(chunks)

    engine.index_documents(all_chunks)
    print(f"Successfully indexed {len(all_chunks)} chunks across ChromaDB and BM25.\n")


def print_mode_results(mode: RetrievalMode, results: List[SearchResult], elapsed_ms: float) -> None:
    """Print formatted comparison table for a given retrieval mode."""
    print("=" * 78)
    print(f" MODE: {mode.upper()} (Found {len(results)} results in {elapsed_ms:.2f} ms)")
    print("=" * 78)

    if not results:
        print("  No matching results found.\n")
        return

    for res in results:
        snippet = res.content.replace("\n", " ")[:130].strip()
        print(
            f"  [Rank {res.rank}] Score: {res.score:>8.4f} | "
            f"{res.source} (p.{res.page}) | Section: {res.section}"
        )
        print(f"    Snippet: \"{snippet}...\"\n")


def main() -> None:
    """Run retrieval comparison CLI."""
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "How many days of paid annual leave am I entitled to?"

    print("\n" + "#" * 78)
    print(f" RETRIEVAL COMPARISON BENCHMARK")
    print(f" Query: \"{query}\"")
    print("#" * 78 + "\n")

    ensure_index_populated()

    modes: List[RetrievalMode] = ["dense", "bm25", "hybrid", "hybrid_rerank"]

    for mode in modes:
        start_time = time.perf_counter()
        results = hybrid_search(query=query, mode=mode, top_k=5)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        print_mode_results(mode, results, elapsed_ms)


if __name__ == "__main__":
    main()
