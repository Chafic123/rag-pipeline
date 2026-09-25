from fastapi import APIRouter

from models import Query
from rag_store import answer   # importing does NOT rebuild the index (thanks to the __main__ guard)

router = APIRouter()


@router.post("/ask-rag")
def ask_rag(q: Query):
    # answer() runs the full RAG pipeline: retrieve -> augment -> generate
    return {"answer": answer(q.prompt)}
