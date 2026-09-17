from fastapi import FastAPI
from pydantic import BaseModel
from llm import ask

class Query(BaseModel):
    prompt: str

app = FastAPI(title="RAG Doc Assistant")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask_endpoint(q: Query):
    answer = ask(q.prompt)
    return {"answer": answer}