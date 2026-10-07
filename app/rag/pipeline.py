"""The RAG pipeline for one question: Retrieve -> Augment -> Generate."""
from app.config import settings
from app.llm import get_openai
from app.rag.store import retrieve

PROMPT_TEMPLATE = (
    "Use ONLY the context below to answer the question. "
    "If the answer is not in the context, say you don't know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)


def answer_with_context(question: str, k: int | None = None) -> dict:
    """Answer a question and return the evidence it was based on.
    The evaluator needs the retrieved chunks to check faithfulness."""
    # 1. Retrieve the most similar chunks
    results = retrieve(question, k=k)
    contexts = [point.payload["text"] for point in results]

    # 2. Augment: put the retrieved context into the prompt
    prompt = PROMPT_TEMPLATE.format(context="\n\n".join(contexts), question=question)

    # 3. Generate a grounded answer
    response = get_openai().chat.completions.create(
        model=settings.generation_model,
        temperature=0,   # same question -> same answer, so evaluation runs are comparable
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "answer": response.choices[0].message.content,
        "contexts": contexts,
        "sources": [point.payload["source"] for point in results],
    }


def answer(question: str, k: int | None = None) -> str:
    """Just the answer text."""
    return answer_with_context(question, k=k)["answer"]
