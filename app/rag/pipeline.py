"""The RAG pipeline for one question: Retrieve -> Augment -> Generate.

Every call is measured: time per stage, tokens per model, and the cost in dollars.
"""
import time

from app.config import settings
from app.costs import cost_usd
from app.llm import get_openai
from app.rag.store import embed_query, search

PROMPT_TEMPLATE = (
    "Use ONLY the context below to answer the question. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 1)


def answer_with_context(question: str, k: int | None = None) -> dict:
    """Answer a question and return the evidence and the measurements.
    The evaluator needs the retrieved chunks to check faithfulness, and the
    metrics (latency per stage, tokens, cost) to track speed and spend."""
    # perf_counter: a high-resolution clock that never jumps, made for measuring durations
    t0 = time.perf_counter()

    # 1. Retrieve: embed the question, then search the index
    query_vector, embed_tokens = embed_query(question)
    t1 = time.perf_counter()
    results = search(query_vector, k=k)
    t2 = time.perf_counter()
    contexts = [point.payload["text"] for point in results]

    # 2. Augment: put the retrieved context into the prompt
    prompt = PROMPT_TEMPLATE.format(context="\n\n".join(contexts), question=question)

    # 3. Generate a grounded answer
    response = get_openai().chat.completions.create(
        model=settings.generation_model,
        temperature=0,   # same question -> same answer, so evaluation runs are comparable
        messages=[{"role": "user", "content": prompt}],
    )
    t3 = time.perf_counter()

    usage = response.usage   # token counts OpenAI billed for this call
    cost = (cost_usd(settings.embed_model, embed_tokens)
            + cost_usd(settings.generation_model, usage.prompt_tokens, usage.completion_tokens))

    return {
        "answer": response.choices[0].message.content,
        "contexts": contexts,
        "sources": [point.payload["source"] for point in results],
        "metrics": {
            "embed_ms": _ms(t0, t1),
            "search_ms": _ms(t1, t2),
            "generate_ms": _ms(t2, t3),
            "total_ms": _ms(t0, t3),
            "embed_tokens": embed_tokens,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost_usd": cost,
        },
    }


def answer(question: str, k: int | None = None) -> str:
    """Just the answer text."""
    return answer_with_context(question, k=k)["answer"]
