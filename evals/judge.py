# judge.py
# LLM-as-judge: a second, stronger model grades the RAG system's answers.
#
# Why: hit-rate only tells us whether the right chunk came back. It says nothing
# about the answer itself. Answer quality has no formula, so we ask a separate
# model to grade it against a written rubric, one metric at a time.
#
#   faithfulness      is every claim in the answer supported by the retrieved context?
#   answer_relevance  does the answer actually address the question?
#   correctness       does it agree with the reference answer we wrote by hand?
#
# Design choices (each one is a real evaluation lesson):
#   - One prompt per metric. Grading everything in one call lets a good impression
#     on one metric leak into the others (the halo effect). RAGAS also does this.
#   - A 1-5 rubric with every level described. Judges are far more consistent on a
#     small labelled scale than on a free 0-1 float.
#   - Reason BEFORE score in the output schema, so the judge justifies first.
#   - Judge stronger than the generator, temperature 0 for repeatable grades.

from pydantic import BaseModel, Field

from app.config import settings
from app.costs import cost_usd
from app.llm import get_openai

# Running token count of every judge call in this process, so run_evals can report
# what the evaluation itself cost, separately from what serving the answers cost.
token_totals = {"calls": 0, "input": 0, "output": 0}


def judge_cost_usd() -> float:
    return cost_usd(settings.judge_model, token_totals["input"], token_totals["output"])

# The judge model (settings.judge_model, gpt-4o by default) is stronger than the
# generator (settings.generation_model, gpt-4o-mini).


class Verdict(BaseModel):
    # Field order matters: the model writes the reason first, then commits to a score.
    reason: str = Field(description="One or two sentences justifying the score.")
    score: int = Field(description="Integer from 1 (worst) to 5 (best), per the rubric.")


def _grade(system_prompt: str, user_prompt: str) -> Verdict:
    """Send one grading request and return a validated Verdict (schema-based Structured Outputs)."""
    completion = get_openai().chat.completions.parse(
        model=settings.judge_model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=Verdict,
    )
    token_totals["calls"] += 1
    token_totals["input"] += completion.usage.prompt_tokens
    token_totals["output"] += completion.usage.completion_tokens

    message = completion.choices[0].message
    if message.refusal:
        raise ValueError(f"Judge refused: {message.refusal}")
    verdict = message.parsed
    verdict.score = min(5, max(1, verdict.score))   # guard against an out-of-range score
    return verdict


FAITHFULNESS_RUBRIC = """You are a strict evaluator of a retrieval-augmented question answering system.
You judge FAITHFULNESS: whether every factual claim in the ANSWER is supported by the CONTEXT.
Do not use outside knowledge. A claim that is true in the real world but absent from the context is unsupported.
Rubric:
5 = every claim is directly supported by the context.
4 = all main claims are supported; one minor detail is not stated explicitly.
3 = some claims are supported and some are not.
2 = most claims are not supported by the context.
1 = the answer contradicts the context or is invented.
If the answer only says it does not know, it makes no unsupported claims: score 5."""

RELEVANCE_RUBRIC = """You are a strict evaluator of a question answering system.
You judge ANSWER RELEVANCE: does the ANSWER directly address the QUESTION? Do not judge factual accuracy.
Rubric:
5 = directly and completely addresses the question.
4 = addresses it, with a minor omission or unnecessary filler.
3 = partially addresses it.
2 = mostly off-topic.
1 = does not address the question, including replies that only say "I don't know"."""

CORRECTNESS_RUBRIC = """You are a strict evaluator of a question answering system.
You judge CORRECTNESS: does the ANSWER convey the same key facts as the REFERENCE answer to the QUESTION?
Wording may differ; meaning must match. Extra correct detail is fine. Missing or wrong key facts lower the score.
Rubric:
5 = same key facts as the reference, nothing wrong.
4 = key facts present, with a minor omission.
3 = partially correct: some key facts missing or slightly wrong.
2 = mostly incorrect, or misses the main point.
1 = wrong, or contradicts the reference."""


def faithfulness(contexts: list[str], answer: str) -> Verdict:
    numbered = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    return _grade(FAITHFULNESS_RUBRIC, f"CONTEXT:\n{numbered}\n\nANSWER:\n{answer}")


def answer_relevance(question: str, answer: str) -> Verdict:
    return _grade(RELEVANCE_RUBRIC, f"QUESTION:\n{question}\n\nANSWER:\n{answer}")


def correctness(question: str, answer: str, reference: str) -> Verdict:
    return _grade(CORRECTNESS_RUBRIC,
                  f"QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\nANSWER:\n{answer}")
