"""Small JSON-backed registry for uploaded policy versions."""

import json
from pathlib import Path
from typing import List, Optional

from backend.models.policy import PolicyDocument


class PolicyRegistry:
    """Persist policy file metadata beside the uploaded documents."""

    def __init__(self, documents_dir: Path) -> None:
        self.path = Path(documents_dir) / ".policy_registry.json"

    def list_documents(self) -> List[PolicyDocument]:
        """Return all registered document versions."""
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as registry_file:
            data = json.load(registry_file)
        if not isinstance(data, list):
            raise ValueError(f"Invalid policy registry format: {self.path}")
        return [PolicyDocument.model_validate(item) for item in data]

    def find_by_hash(self, file_hash: str) -> Optional[PolicyDocument]:
        """Find a previously uploaded document by its byte hash."""
        return next(
            (document for document in self.list_documents() if document.file_hash == file_hash),
            None,
        )

    def get_by_filename(self, filename: str) -> Optional[PolicyDocument]:
        """Find a registered version by its filename."""
        return next(
            (document for document in self.list_documents() if document.filename == filename),
            None,
        )

    def upsert(self, document: PolicyDocument) -> None:
        """Save metadata for a document, replacing its same-filename entry."""
        documents = [
            existing
            for existing in self.list_documents()
            if existing.filename != document.filename
        ]
        documents.append(document)
        self._write(documents)

    def update_status(self, filename: str, status: str) -> None:
        """Update one registered document lifecycle status."""
        documents = self.list_documents()
        updated = False
        for document in documents:
            if document.filename == filename:
                document.status = status
                updated = True
        if updated:
            self._write(documents)

    def delete(self, filename: str) -> None:
        """Remove a document record from the registry."""
        documents = [
            document
            for document in self.list_documents()
            if document.filename != filename
        ]
        self._write(documents)

    def _write(self, documents: List[PolicyDocument]) -> None:
        """Atomically persist registry updates."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as registry_file:
            json.dump(
                [document.model_dump(mode="json") for document in documents],
                registry_file,
                indent=2,
            )
        temporary_path.replace(self.path)
