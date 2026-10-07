"""FastAPI entry point: creates the app, wires the routers, builds the index at startup.

Run locally:  make run   (same as: uvicorn app.main:app --reload)
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import health, rag
from app.rag.store import build_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP: runs once when the server boots, not on every request.
    build_index()
    yield
    # SHUTDOWN: nothing to clean up yet.


app = FastAPI(title="RAG Doc Assistant", lifespan=lifespan)
app.include_router(health.router)
app.include_router(rag.router)
