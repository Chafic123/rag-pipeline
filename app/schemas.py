"""Request and response shapes for the HTTP API, validated by Pydantic."""
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1, description="The question to answer from the documents.")


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list, description="Documents the answer was based on.")
