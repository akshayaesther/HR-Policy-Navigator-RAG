"""FastAPI application entrypoint for Enterprise HR & Policy Navigator."""

import time
import hashlib
import re
from pathlib import Path
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import fitz
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.embeddings.vector_store import (
    delete_by_source,
    index_chunks,
    update_document_metadata,
    update_policy_status,
)
from backend.generation.answer_generator import (
    AnswerGenerator,
    OllamaResponseError,
    OllamaUnavailableError,
    REFUSAL_RESPONSE,
)
from backend.ingestion.document_loader import DocumentLoader
from backend.ingestion.policy_metadata import (
    add_policy_metadata,
    derive_doc_id,
    extract_policy_metadata,
)
from backend.ingestion.policy_registry import PolicyRegistry
from backend.models.api import (
    AskRequest,
    AskResponse,
    Citation,
    DocumentInfo,
    DocumentUploadResponse,
    HealthResponse,
    PolicyChangesResponse,
    PolicySectionChange,
    RetrievedChunk,
)
from backend.models.policy import PolicyDocument
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


def _policy_registry() -> PolicyRegistry:
    registry = PolicyRegistry(_documents_directory())
    documents = registry.list_documents()
    registered_filenames = {document.filename for document in documents}
    newly_discovered: List[PolicyDocument] = []

    for path in sorted(_documents_directory().iterdir()):
        if (
            not path.is_file()
            or path.suffix.lower() not in ALLOWED_DOCUMENT_EXTENSIONS
            or path.name in registered_filenames
        ):
            continue
        chunks, _ = DocumentLoader().load_and_chunk(path)
        metadata = extract_policy_metadata("\n".join(chunk.content for chunk in chunks))
        newly_discovered.append(
            PolicyDocument(
                doc_id=derive_doc_id(path.name),
                filename=path.name,
                version=metadata["version"],
                effective_date=metadata["effective_date"],
                status="active",
                file_hash=file_sha256_from_bytes(path.read_bytes()),
                size_bytes=path.stat().st_size,
                sections=_sections_from_chunks(chunks),
            )
        )

    all_documents = [*documents, *newly_discovered]
    grouped: Dict[str, List[PolicyDocument]] = {}
    for document in all_documents:
        grouped.setdefault(document.doc_id, []).append(document)

    for family_documents in grouped.values():
        latest = max(
            family_documents,
            key=lambda item: (
                _version_key(item.version),
                item.effective_date or date.min,
                item.filename,
            ),
        )
        for document in family_documents:
            expected_status = "active" if document.filename == latest.filename else "superseded"
            if document.status != expected_status:
                document.status = expected_status  # type: ignore[assignment]
                if document in documents:
                    registry.update_status(document.filename, expected_status)
            if document in newly_discovered:
                update_document_metadata(
                    source=document.filename,
                    doc_id=document.doc_id,
                    version=document.version,
                    effective_date=(
                        document.effective_date.isoformat()
                        if document.effective_date
                        else None
                    ),
                    status=expected_status,
                    file_hash=document.file_hash,
                )
                registry.upsert(document)

    return registry


def _version_key(version: Optional[str]) -> Tuple[int, ...]:
    """Convert a dotted numeric policy version to a comparable tuple."""
    if not version or not re.fullmatch(r"\d+(?:\.\d+)*", version):
        return ()
    return tuple(int(part) for part in version.split("."))


def file_sha256_from_bytes(contents: bytes) -> str:
    """Hash the already size-limited upload bytes."""
    return hashlib.sha256(contents).hexdigest()


def _sections_from_chunks(chunks: List[Any]) -> Dict[str, str]:
    """Collect document text by section for version-to-version comparisons."""
    sections: Dict[str, List[str]] = {}
    for chunk in chunks:
        sections.setdefault(chunk.section, []).append(chunk.content)
    return {section: "\n".join(contents) for section, contents in sections.items()}


def _validate_document_id(document_id: str) -> str:
    """Reject paths and unsupported extensions in document IDs."""
    if not document_id or "\0" in document_id or Path(document_id).name != document_id:
        raise HTTPException(status_code=400, detail="Invalid document id.")
    if Path(document_id).suffix.lower() not in ALLOWED_DOCUMENT_EXTENSIONS:
        raise HTTPException(status_code=404, detail="Document not found.")
    return document_id


def _validate_doc_id(doc_id: str) -> str:
    """Validate a policy-family ID used by the changes endpoint."""
    if not doc_id or Path(doc_id).name != doc_id:
        raise HTTPException(status_code=400, detail="Invalid policy doc_id.")
    try:
        return derive_doc_id("policy.md", doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid policy doc_id.") from exc


@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload and index a policy document",
)
async def upload_document(
    response: Response,
    file: UploadFile = File(...),
    version: Optional[str] = Form(default=None),
    effective_date: Optional[date] = Form(default=None),
    doc_id: Optional[str] = Form(default=None),
) -> DocumentUploadResponse:
    """Validate, version, persist, ingest, and index a policy document."""
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

    content_hash = file_sha256_from_bytes(contents)
    registry = _policy_registry()
    existing_hash = registry.find_by_hash(content_hash)
    if existing_hash:
        response.status_code = 200
        return DocumentUploadResponse(
            id=existing_hash.filename,
            filename=existing_hash.filename,
            doc_id=existing_hash.doc_id,
            version=existing_hash.version,
            effective_date=existing_hash.effective_date,
            status=existing_hash.status,
            file_hash=existing_hash.file_hash,
            chunks_indexed=0,
            skipped=True,
        )

    try:
        policy_doc_id = derive_doc_id(filename, doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    storage_filename = filename
    target_path = _documents_directory() / storage_filename
    if target_path.exists():
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        storage_filename = f"{stem}_{content_hash[:12]}{suffix}"
        target_path = _documents_directory() / storage_filename
        if target_path.exists():
            storage_filename = f"{stem}_{content_hash}{suffix}"
            target_path = _documents_directory() / storage_filename

    try:
        with target_path.open("xb") as uploaded_file:
            uploaded_file.write(contents)
    except FileExistsError as exc:
        raise HTTPException(
            status_code=409,
            detail="An upload with this filename is already being processed.",
        ) from exc
    try:
        chunks, _ = DocumentLoader().load_and_chunk(target_path)
    except (ValueError, fitz.FileDataError, UnicodeDecodeError) as exc:
        target_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="Uploaded file is not a valid PDF or UTF-8 Markdown document.",
        ) from exc

    extracted_metadata = extract_policy_metadata("\n".join(chunk.content for chunk in chunks))
    selected_version = extracted_metadata["version"] or version
    selected_effective_date = extracted_metadata["effective_date"] or effective_date

    previous_versions = [
        document
        for document in registry.list_documents()
        if document.doc_id == policy_doc_id
    ]
    if previous_versions:
        latest_version = max(
            (_version_key(document.version) for document in previous_versions),
            default=(),
        )
        if not _version_key(selected_version) or _version_key(selected_version) <= latest_version:
            target_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=409,
                detail="A new policy version must be higher than the latest existing version.",
            )

    add_policy_metadata(
        chunks,
        doc_id=policy_doc_id,
        version=selected_version,
        effective_date=selected_effective_date,
        file_hash=content_hash,
    )
    try:
        index_chunks(chunks)
    except Exception as exc:
        cleanup_error = None
        try:
            delete_by_source(storage_filename)
        except Exception as cleanup_exc:
            cleanup_error = cleanup_exc
        finally:
            target_path.unlink(missing_ok=True)
        if cleanup_error is not None:
            raise HTTPException(
                status_code=500,
                detail="Document indexing failed and partial index cleanup was unsuccessful.",
            ) from cleanup_error
        raise HTTPException(
            status_code=500,
            detail="Document indexing failed; the uploaded file was removed.",
        ) from exc
    for previous in previous_versions:
        registry.update_status(previous.filename, "superseded")
        update_policy_status(
            doc_id=previous.doc_id,
            version=previous.version,
            status="superseded",
        )

    policy_document = PolicyDocument(
        doc_id=policy_doc_id,
        filename=storage_filename,
        version=selected_version,
        effective_date=selected_effective_date,
        status="active",
        file_hash=content_hash,
        size_bytes=len(contents),
        sections=_sections_from_chunks(chunks),
    )
    registry.upsert(policy_document)
    return DocumentUploadResponse(
        id=storage_filename,
        filename=storage_filename,
        doc_id=policy_doc_id,
        version=selected_version,
        effective_date=selected_effective_date,
        status="active",
        file_hash=content_hash,
        chunks_indexed=len(chunks),
        skipped=False,
    )


@app.get(
    "/documents",
    response_model=List[DocumentInfo],
    summary="List indexed policy documents",
)
def list_documents() -> List[DocumentInfo]:
    """List supported files in the configured document directory."""
    documents = _policy_registry().list_documents()
    return [
        DocumentInfo(
            id=document.filename,
            filename=document.filename,
            size_bytes=document.size_bytes,
            doc_id=document.doc_id,
            version=document.version,
            effective_date=document.effective_date,
            status=document.status,
            file_hash=document.file_hash,
        )
        for document in sorted(documents, key=lambda item: item.filename)
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

    registry = _policy_registry()
    document = registry.get_by_filename(safe_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    delete_by_source(safe_id)
    document_path.unlink()
    registry.delete(safe_id)
    remaining_versions = [
        item for item in registry.list_documents() if item.doc_id == document.doc_id
    ]
    if remaining_versions:
        active_version = max(remaining_versions, key=lambda item: _version_key(item.version))
        registry.update_status(active_version.filename, "active")
        update_policy_status(
            doc_id=active_version.doc_id,
            version=active_version.version,
            status="active",
        )


@app.get(
    "/documents/{doc_id}/changes",
    response_model=PolicyChangesResponse,
    summary="Compare the latest two versions of a policy",
)
def policy_changes(doc_id: str) -> PolicyChangesResponse:
    """Return sections added, removed, or changed between policy versions."""
    safe_doc_id = _validate_doc_id(doc_id)
    versions = [
        document
        for document in _policy_registry().list_documents()
        if document.doc_id == safe_doc_id
    ]
    if len(versions) < 2:
        raise HTTPException(status_code=404, detail="At least two policy versions are required.")

    versions.sort(
        key=lambda item: (_version_key(item.version), item.effective_date or date.min)
    )
    previous, current = versions[-2], versions[-1]
    changes: List[PolicySectionChange] = []
    for section in sorted(set(previous.sections) | set(current.sections)):
        if section not in previous.sections:
            change = "added"
        elif section not in current.sections:
            change = "removed"
        elif previous.sections[section] != current.sections[section]:
            change = "changed"
        else:
            continue
        changes.append(PolicySectionChange(section=section, change=change))

    return PolicyChangesResponse(
        doc_id=safe_doc_id,
        from_version=previous.version,
        to_version=current.version,
        changes=changes,
    )


@app.post("/ask", response_model=AskResponse, summary="Answer a policy question")
def ask_question(request: AskRequest) -> AskResponse:
    """Retrieve policy evidence and generate a cited answer when evidence is relevant."""
    start_time = time.perf_counter()
    _policy_registry()
    results = hybrid_search(
        query=request.query,
        mode=request.mode,
        filters=request.filters,
        top_k=request.top_k,
        as_of_date=request.as_of_date,
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
            doc_id=result.doc_id,
            version=result.version,
            effective_date=result.effective_date,
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
                    version=result.version,
                    effective_date=result.effective_date,
                )
            )
            seen_citations.add(citation_key)

    refused = answer == refusal
    version_citations = {
        (result.source, result.version, result.effective_date)
        for result in results
        if result.version or result.effective_date
    }
    if not refused and version_citations:
        version_details = "; ".join(
            f"{document}, version {version or 'unknown'}, effective "
            f"{effective_date.isoformat() if effective_date else 'unknown'}"
            for document, version, effective_date in sorted(
                version_citations,
                key=lambda item: (
                    item[0],
                    item[1] or "",
                    item[2].isoformat() if item[2] else "",
                ),
            )
        )
        answer = f"{answer.rstrip()}\n\nPolicy version(s) used: {version_details}."
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
