"""FastAPI application entrypoint for Enterprise HR & Policy Navigator."""

import time
from pathlib import Path
from typing import Any, Dict, List

import fitz
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.embeddings.vector_store import delete_by_source, index_chunks
from backend.generation.answer_generator import (
    AnswerGenerator,
    OllamaResponseError,
    OllamaUnavailableError,
    REFUSAL_RESPONSE,
)
from backend.ingestion.document_loader import DocumentLoader
from backend.models.api import (
    AskRequest,
    AskResponse,
    Citation,
    DocumentInfo,
    DocumentUploadResponse,
    HealthResponse,
    RetrievedChunk,
)
from backend.retrieval.hybrid_search import hybrid_search

MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024
ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".md"}

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Enterprise HR & Policy Navigator with Evaluated Hybrid Search",
    version="1.0.0",
    debug=settings.debug,
)

# Enable CORS for local React/Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Root Endpoint")
def read_root() -> Dict[str, Any]:
    """Provide root status and service metadata."""
    return {
        "message": f"Welcome to {settings.app_name} API",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health", summary="Health Check")
def health_check() -> HealthResponse:
    """Check API server health status."""
    current_settings = get_settings()
    return HealthResponse(
        status="healthy",
        app=current_settings.app_name,
        environment=current_settings.app_env,
    )


def get_answer_generator() -> AnswerGenerator:
    """Create an Ollama generator using current application settings."""
    return AnswerGenerator(settings=get_settings())


def _documents_directory() -> Path:
    """Resolve and create the configured document directory."""
    directory = get_settings().get_resolved_documents_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _validate_document_id(document_id: str) -> str:
    """Reject paths and unsupported extensions in document IDs."""
    if not document_id or "\0" in document_id or Path(document_id).name != document_id:
        raise HTTPException(status_code=400, detail="Invalid document id.")
    if Path(document_id).suffix.lower() not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document_id


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload and index a policy document",
)
async def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    """Validate, persist, ingest, and index one PDF or Markdown document."""
    filename = file.filename or ""
    if (
        not filename
        or "\0" in filename
        or Path(filename).name != filename
        or filename in {".", ".."}
    ):
        raise HTTPException(status_code=400, detail="A plain document filename is required.")

    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Only .pdf and .md files are supported.")

    contents = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Document exceeds the 20 MB upload limit.")

    target_path = _documents_directory() / filename
    try:
        with target_path.open("xb") as uploaded_file:
            uploaded_file.write(contents)
    except FileExistsError as exc:
        raise HTTPException(
            status_code=409,
            detail="A document with this filename already exists.",
        ) from exc

    try:
        chunks, _ = DocumentLoader().load_and_chunk(target_path)
    except (ValueError, fitz.FileDataError, UnicodeDecodeError) as exc:
        target_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is not a valid PDF or UTF-8 Markdown document.",
        ) from exc

    index_chunks(chunks)
    return DocumentUploadResponse(
        id=filename,
        filename=filename,
        chunks_indexed=len(chunks),
    )


@app.get(
    "/documents",
    response_model=List[DocumentInfo],
    summary="List indexed policy documents",
)
def list_documents() -> List[DocumentInfo]:
    """List supported files in the configured document directory."""
    documents_dir = _documents_directory()
    return [
        DocumentInfo(id=path.name, filename=path.name, size_bytes=path.stat().st_size)
        for path in sorted(documents_dir.iterdir())
        if path.is_file() and path.suffix.lower() in ALLOWED_DOCUMENT_EXTENSIONS
    ]


@app.delete(
    "/documents/{document_id}",
    status_code=204,
    summary="Delete a policy document",
)
def remove_document(document_id: str) -> None:
    """Remove a stored document and its chunks from both retrieval indexes."""
    safe_id = _validate_document_id(document_id)
    document_path = _documents_directory() / safe_id
    if not document_path.is_file():
        raise HTTPException(status_code=404, detail="Document not found.")

    delete_by_source(safe_id)
    document_path.unlink()


@app.post("/ask", response_model=AskResponse, summary="Answer a policy question")
def ask_question(request: AskRequest) -> AskResponse:
    """Retrieve policy evidence and generate a cited answer when evidence is relevant."""
    start_time = time.perf_counter()
    results = hybrid_search(
        query=request.query,
        mode=request.mode,
        filters=request.filters,
        top_k=request.top_k,
    )
    retrieved_chunks = [
        RetrievedChunk(
            chunk_id=result.chunk_id,
            content=result.content,
            document=result.source,
            page=result.page,
            section=result.section,
            score=result.score,
            rank=result.rank,
        )
        for result in results
    ]

    refusal = REFUSAL_RESPONSE
    is_reranked = request.mode == "hybrid_rerank"
    if not results or (
        is_reranked
        and max(result.score for result in results) < get_settings().relevance_score_threshold
    ):
        return AskResponse(
            answer=refusal,
            citations=[],
            retrieved_chunks=retrieved_chunks,
            mode=request.mode,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            refused=True,
        )

    try:
        answer = get_answer_generator().generate(request.query, results)
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaResponseError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    citations: List[Citation] = []
    seen_citations = set()
    for result in results:
        citation_key = (result.source, result.page, result.section)
        if citation_key not in seen_citations:
            citations.append(
                Citation(
                    document=result.source,
                    page=result.page,
                    section=result.section,
                )
            )
            seen_citations.add(citation_key)

    refused = answer == refusal
    return AskResponse(
        answer=answer,
        citations=[] if refused else citations,
        retrieved_chunks=retrieved_chunks,
        mode=request.mode,
        latency_ms=(time.perf_counter() - start_time) * 1000,
        refused=refused,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
