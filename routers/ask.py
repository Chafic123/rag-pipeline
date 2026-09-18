from fastapi import APIRouter

from models import Query, Answer
from llm import ask, ask_structured

router = APIRouter()


@router.post("/ask")
def ask_endpoint(q: Query):
    return {"answer": ask(q.prompt)}


@router.post("/ask-structured", response_model=Answer)
def ask_structured_endpoint(q: Query):
    return ask_structured(q.prompt)
