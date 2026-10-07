from fastapi import APIRouter

from app.rag.pipeline import answer_with_context
from app.schemas import AskRequest, AskResponse

router = APIRouter(tags=["rag"])


@router.post("/ask-rag", response_model=AskResponse)
def ask_rag(request: AskRequest) -> AskResponse:
    """Answer a question from the indexed documents: retrieve -> augment -> generate."""
    result = answer_with_context(request.prompt)
    sources = list(dict.fromkeys(result["sources"]))   # unique, in retrieval order
    return AskResponse(answer=result["answer"], sources=sources)
