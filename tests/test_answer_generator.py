"""Tests for Ollama HTTP answer generation."""

from pathlib import Path
from typing import Any

import httpx
import pytest

from backend.config import Settings
from backend.generation.answer_generator import AnswerGenerator, REFUSAL_RESPONSE
from backend.models.search import SearchResult


def test_generator_sends_model_and_cited_context_to_ollama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama receives the configured model and policy evidence with citation metadata."""
    settings = Settings(
        base_dir=Path("."),
        ollama_base_url="http://localhost:11434/",
        ollama_model="test-policy-model",
        ollama_temperature=0.2,
    )
    request_data: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> httpx.Response:
        request_data.update(url=url, body=json, timeout=timeout)
        return httpx.Response(
            200,
            json={"response": "The policy grants 20 days [Leave.pdf, p. 2]."},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    chunk = SearchResult(
        chunk_id="leave-1",
        content="Full-time employees receive 20 paid annual leave days.",
        source="Leave.pdf",
        page=2,
        category="hr_policy",
        section="Annual Leave",
        score=0.8,
        rank=1,
        doc_id="leave_policy",
        version="2.0",
        effective_date="2025-04-01",
        metadata={},
    )

    answer = AnswerGenerator(settings).generate("How many leave days?", [chunk])

    assert answer == "The policy grants 20 days [Leave.pdf, p. 2]."
    assert request_data["url"] == "http://localhost:11434/api/generate"
    assert request_data["body"]["model"] == "test-policy-model"
    assert request_data["body"]["stream"] is False
    assert REFUSAL_RESPONSE in request_data["body"]["system"]
    assert (
        "Document: Leave.pdf; Page: 2; Section: Annual Leave; "
        "Version: 2.0; Effective date: 2025-04-01"
    ) in request_data["body"]["prompt"]
