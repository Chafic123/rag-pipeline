from contextlib import asynccontextmanager

from fastapi import FastAPI

from routers import health, ask, rag
from rag_store import build_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: runs ONCE when the server boots.
    # Build the vector index so the RAG endpoint has documents to search.
    build_index()
    yield
    # (anything after yield would run on SHUTDOWN; we do not need any)


app = FastAPI(title="RAG Doc Assistant", lifespan=lifespan)

app.include_router(health.router)
app.include_router(ask.router)
app.include_router(rag.router)
