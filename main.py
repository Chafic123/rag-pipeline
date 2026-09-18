from fastapi import FastAPI

from routers import health, ask

app = FastAPI(title="RAG Doc Assistant")

app.include_router(health.router)
app.include_router(ask.router)
