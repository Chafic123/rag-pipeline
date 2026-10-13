import logging

from fastapi import APIRouter

from app.rag.pipeline import answer_with_context
from app.schemas import AskRequest, AskResponse

router = APIRouter(tags=["rag"])
logger = logging.getLogger(__name__)


@router.post("/ask-rag", response_model=AskResponse)
def ask_rag(request: AskRequest) -> AskResponse:
    """Answer a question from the indexed documents: retrieve -> augment -> generate."""
    result = answer_with_context(request.prompt)
    sources = list(dict.fromkeys(result["sources"]))   # unique, in retrieval order

    # One key=value line per request: easy to grep now, easy for a log tool
    # (Grafana Loki, Datadog) to parse and chart later.
    m = result["metrics"]
    logger.info(
        "ask-rag total_ms=%.0f embed_ms=%.0f search_ms=%.0f generate_ms=%.0f "
        "tokens_in=%d tokens_out=%d cost_usd=%.6f",
        m["total_ms"], m["embed_ms"], m["search_ms"], m["generate_ms"],
        m["embed_tokens"] + m["prompt_tokens"], m["completion_tokens"], m["cost_usd"],
    )
    return AskResponse(answer=result["answer"], sources=sources)
