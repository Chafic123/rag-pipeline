from fastapi import FastAPI

app = FastAPI(title="RAG Doc Assistant")

@app.get("/health")
def health():
    return {"status": "ok"}