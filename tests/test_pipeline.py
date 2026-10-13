"""The pipeline's measurements, tested with fakes instead of OpenAI and Qdrant.

monkeypatch swaps a real function for a fake one during a single test, so we can check
our own logic (timing, token counting, cost math) without network calls or an API key.
"""
from types import SimpleNamespace

import pytest

import app.rag.pipeline as pipeline
from app.config import settings
from app.costs import cost_usd


def _fake_openai(answer: str, prompt_tokens: int, completion_tokens: int):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response)))


def test_answer_reports_latency_tokens_and_cost(monkeypatch):
    chunk = SimpleNamespace(payload={"text": "HNSW is a graph index.", "source": "vector_search.txt"})
    monkeypatch.setattr(pipeline, "embed_query", lambda question: ([0.1, 0.2], 8))
    monkeypatch.setattr(pipeline, "search", lambda vector, k=None: [chunk])
    monkeypatch.setattr(pipeline, "get_openai", lambda: _fake_openai("A graph index.", 400, 50))

    out = pipeline.answer_with_context("What is HNSW?")
    m = out["metrics"]

    assert out["answer"] == "A graph index."
    assert out["sources"] == ["vector_search.txt"]
    assert (m["embed_tokens"], m["prompt_tokens"], m["completion_tokens"]) == (8, 400, 50)
    expected = cost_usd(settings.embed_model, 8) + cost_usd(settings.generation_model, 400, 50)
    assert m["cost_usd"] == pytest.approx(expected)
    for stage in ("embed_ms", "search_ms", "generate_ms"):
        assert 0 <= m[stage] <= m["total_ms"]
