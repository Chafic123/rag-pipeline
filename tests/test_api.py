"""API tests that need no OpenAI key and no vector store.

TestClient(app) WITHOUT a `with` block does not run the lifespan, so the startup
index build (which calls OpenAI) is skipped. Behaviour that needs the model is
measured by the evals in evals/, not by unit tests.
"""
from fastapi.testclient import TestClient

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
