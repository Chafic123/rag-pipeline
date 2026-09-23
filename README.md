# RAG Doc Assistant

A document question-answering service built with FastAPI, on the way to becoming a
retrieval-augmented (RAG) assistant. The service foundation is in place; retrieval
over documents is the next stage.

## What it does (current)

- `GET /health` — liveness check.
- `POST /ask` — sends a question to an LLM and returns the plain-text answer.
- `POST /ask-structured` — returns a schema-validated JSON answer (`topic`,
  `summary`, `confidence`) using OpenAI Structured Outputs, so the shape is
  guaranteed, not just hoped for.

## Tech stack

- Python, FastAPI, Uvicorn
- OpenAI Python SDK (`gpt-4o-mini`)
- Pydantic for request and response validation
- NumPy for vector math
- Qdrant for vector storage and similarity search

## Project structure

```
main.py            # app entry point, wires the routers together
models.py          # Pydantic schemas (Query, Answer)
llm.py             # LLM calls (ask, ask_structured)
embeddings_demo.py # experiment: embeddings + cosine similarity
ingest.py          # split a document into chunks
rag_store.py       # full RAG: chunk -> embed -> store -> retrieve -> answer
docs/sample.txt    # example document to answer questions from
routers/
  health.py        # GET /health
  ask.py           # POST /ask, POST /ask-structured
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your API key (this file is gitignored
and never committed):

```
OPENAI_API_KEY=sk-your-key
```

## Run

```bash
uvicorn main:app --reload
```

Then open the interactive API docs at http://127.0.0.1:8000/docs

## Example

```bash
curl -X POST http://127.0.0.1:8000/ask-structured \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain what an API key is."}'
```

Returns:

```json
{ "topic": "API keys", "summary": "...", "confidence": 0.9 }
```

## Roadmap

- [x] Document ingestion and chunking (`ingest.py`)
- [x] Explore embeddings and cosine similarity (`embeddings_demo.py`)
- [x] Vector store and semantic retrieval (Qdrant)
- [x] RAG answer as a script: retrieve, augment, generate (`rag_store.py`)
- [ ] Expose RAG through the API endpoint
- [ ] Evaluation harness and metrics
- [ ] Observability: logging, cost, and latency

## Status

Early stage, and actively built. Part of a personal project to combine backend
engineering with AI/ML systems, working toward a production-grade RAG service.
