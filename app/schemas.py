from pydantic import BaseModel


class Query(BaseModel):
    prompt: str


class Answer(BaseModel):
    topic: str
    summary: str
    confidence: float
