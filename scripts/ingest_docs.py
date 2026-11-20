"""Ingestion CLI utility to process and report on policy documents."""

from pathlib import Path
from backend.config import get_settings
from backend.ingestion.document_loader import DocumentLoader, print_ingestion_report


def run_ingestion() -> None:
    """Run DocumentLoader across all documents in backend/data/documents and print reports."""
    settings = get_settings()
    docs_dir = settings.get_resolved_documents_dir()

    print(f"Loading and chunking documents from: {docs_dir.resolve()}\n")
    loader = DocumentLoader()

    supported_files = sorted(list(docs_dir.glob("*.pdf")) + list(docs_dir.glob("*.md")))
    if not supported_files:
        print("No supported documents (.pdf, .md) found.")
        return

    for doc_file in supported_files:
        _, report = loader.load_and_chunk(doc_file)
        print_ingestion_report(report)


if __name__ == "__main__":
    run_ingestion()
