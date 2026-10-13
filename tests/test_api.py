"""API tests that need no OpenAI key and no vector store.

TestClient(app) WITHOUT a `with` block does not run the lifespan, so the startup
index build (which calls OpenAI) is skipped. Behaviour that needs the model is
measured by the evals in evals/, not by unit tests.
"""
import logging

from fastapi.testclient import TestClient

import app.api.routes.rag as rag_route

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_rag_rejects_an_empty_prompt():
    # Validation runs before any model call, so this needs no API key.
    assert client.post("/ask-rag", json={"prompt": ""}).status_code == 422


def test_ask_rag_rejects_a_missing_prompt():
    assert client.post("/ask-rag", json={}).status_code == 422


def test_ask_rag_logs_latency_tokens_and_cost(monkeypatch, caplog):
    fake = {"answer": "A graph index.", "contexts": ["..."], "sources": ["vector_search.txt"] * 2,
            "metrics": {"embed_ms": 210.0, "search_ms": 3.0, "generate_ms": 950.0, "total_ms": 1163.0,
                        "embed_tokens": 8, "prompt_tokens": 400, "completion_tokens": 50,
                        "cost_usd": 0.0000903}}
    monkeypatch.setattr(rag_route, "answer_with_context", lambda prompt: fake)

    with caplog.at_level(logging.INFO, logger="app.api.routes.rag"):
        response = client.post("/ask-rag", json={"prompt": "What is HNSW?"})

    assert response.json() == {"answer": "A graph index.", "sources": ["vector_search.txt"]}
    assert "total_ms=1163 " in caplog.text
    assert "tokens_in=408 tokens_out=50 cost_usd=0.000090" in caplog.text
