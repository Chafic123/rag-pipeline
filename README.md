# RAG Doc Assistant

A retrieval-augmented generation (RAG) service that answers questions from a set of
documents, cites the sources it used, and measures its own quality. Built with FastAPI,
OpenAI and Qdrant, packaged with Docker.

## What it does

- `POST /ask-rag` answers a question using only the indexed documents and returns the
  answer plus the source documents it was based on. If the answer is not in the
  documents, it says it does not know.
- `GET /health` is a liveness check.

On startup the service reads every document in `data/docs/`, splits it into overlapping
chunks, embeds them, and stores them in a local Qdrant index (HNSW). Each question is
embedded, matched against the index, and answered by the LLM from the retrieved chunks only.

## Evaluation

Quality is measured, not guessed. `evals/` holds a ground-truth set of 20 questions, each
with the expected source document and a reference answer.

- **Retrieval:** hit@1 and hit@k, did the right document come back?
- **Generation:** faithfulness, answer relevance and correctness, scored 1 to 5 by an
  LLM-as-judge (a stronger model than the generator, one rubric per metric).

Every run is saved to `evals/results/`, so a change such as adding a reranker can be
compared against a baseline.

## Project structure

```
app/                    the service
  main.py               FastAPI app, wires the routers, builds the index at startup
  config.py             all settings in one place, read from the environment
  llm.py                one shared OpenAI client
  schemas.py            request and response models
  api/routes/           HTTP layer: health.py, rag.py
  rag/
    ingest.py           load documents and split them into chunks
    store.py            embeddings, Qdrant index (HNSW), retrieval
    pipeline.py         retrieve -> augment -> generate
evals/                  quality measurement (not shipped in the image)
  eval_set.json         ground truth: questions, expected sources, reference answers
  judge.py              LLM-as-judge
  run_evals.py          runs the evaluation and saves results
tests/                  fast unit tests, no API key needed
data/docs/              the document corpus
Dockerfile              container image
Makefile                common commands
.env.example            required environment variables
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
cp .env.example .env      # then put your OpenAI API key in .env
```

## Commands

| Command | What it does |
|---|---|
| `make run` | start the API, docs at http://127.0.0.1:8000/docs |
| `make test` | run the unit tests |
| `make eval` | full evaluation (stop the API first: Qdrant local mode allows one process) |
| `make eval-quick` | full evaluation on the first 3 questions |
| `make eval-retrieval` | retrieval-only evaluation, almost free |
| `make docker-build` / `make docker-run` | build and run the container |

## Example

```bash
curl -X POST http://127.0.0.1:8000/ask-rag \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What does HNSW stand for?"}'
```

Response shape:

```json
{ "answer": "...", "sources": ["vector_search.txt"] }
```

## Tech stack

Python, FastAPI, Uvicorn, Pydantic, OpenAI (`text-embedding-3-small` for embeddings,
`gpt-4o-mini` for answers, `gpt-4o` as judge), Qdrant, Docker, pytest.

## Roadmap

- [x] Ingestion and chunking with overlap
- [x] Vector store and semantic retrieval (Qdrant, explicit HNSW parameters)
- [x] RAG endpoint with grounded answers and source citations
- [x] Docker image
- [x] Evaluation harness: retrieval hit-rate and LLM-as-judge
- [x] Company-style project layout, central config, unit tests
- [ ] Baseline evaluation numbers in this README
- [ ] RAGAS metrics and Promptfoo regression tests
- [ ] Reranking (cross-encoder) and hybrid search (BM25 + vector)
- [ ] CI: tests on every push
- [ ] Observability: tracing, cost and latency

## Status

Actively built, as part of a project combining backend engineering with AI systems,
working toward a production-grade RAG service.
