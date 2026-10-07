"""Evaluate the RAG pipeline and keep a history of results.

    make eval              full run: retrieval + LLM judge on every question
    make eval-quick        the same, first 3 questions only
    make eval-retrieval    retrieval only (hit@1, hit@k), no generation, almost free

    (same as: python -m evals.run_evals [--limit N] [--retrieval-only])

Per question it records:
    retrieval    hit@1 and hit@k: did the expected source document come back?
    generation   faithfulness, answer relevance, correctness (LLM judge, 1-5)
    latency      seconds for the RAG call
Results are printed and saved to evals/results/<timestamp>-<mode>.json, so a later
run (for example after adding a reranker) can be compared with this one.

NOTE: stop the API server first. Qdrant local mode allows one process at a time.
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.rag.pipeline import answer_with_context
from app.rag.store import build_index, retrieve
from evals import judge

EVALS_DIR = Path(__file__).resolve().parent
EVAL_SET = EVALS_DIR / "eval_set.json"
RESULTS_DIR = EVALS_DIR / "results"


def _hits(sources: list[str], expected: str) -> dict:
    return {
        "hit@1": bool(sources) and sources[0] == expected,   # right doc is the TOP result
        "hit@k": expected in sources,                          # right doc anywhere in the top k
    }


def evaluate_retrieval(item: dict, k: int) -> dict:
    sources = [point.payload["source"] for point in retrieve(item["question"], k=k)]
    return {
        "question": item["question"],
        "expected_source": item["expected_source"],
        "retrieved_sources": sources,
        **_hits(sources, item["expected_source"]),
    }


def evaluate_full(item: dict, k: int) -> dict:
    question = item["question"]
    start = time.perf_counter()
    out = answer_with_context(question, k=k)
    latency_s = time.perf_counter() - start

    faith = judge.faithfulness(out["contexts"], out["answer"])
    relev = judge.answer_relevance(question, out["answer"])
    corr = judge.correctness(question, out["answer"], item["reference_answer"])

    return {
        "question": question,
        "expected_source": item["expected_source"],
        "retrieved_sources": out["sources"],
        **_hits(out["sources"], item["expected_source"]),
        "answer": out["answer"],
        "latency_s": round(latency_s, 2),
        "faithfulness": faith.score, "faithfulness_reason": faith.reason,
        "relevance": relev.score, "relevance_reason": relev.reason,
        "correctness": corr.score, "correctness_reason": corr.reason,
    }


def summarize(rows: list[dict], retrieval_only: bool) -> dict:
    n = len(rows)
    summary = {
        "n": n,
        "hit_rate@1": sum(r["hit@1"] for r in rows) / n,
        "hit_rate@k": sum(r["hit@k"] for r in rows) / n,
    }
    if not retrieval_only:
        def normalized(key):   # map the 1-5 rubric onto 0-1 so it reads as a percentage
            return sum((r[key] - 1) / 4 for r in rows) / n
        summary.update({
            "faithfulness": normalized("faithfulness"),
            "relevance": normalized("relevance"),
            "correctness": normalized("correctness"),
            "avg_latency_s": sum(r["latency_s"] for r in rows) / n,
        })
    return summary


def print_row(r: dict, retrieval_only: bool) -> None:
    hits = f"{'Y' if r['hit@1'] else '.':^4}{'Y' if r['hit@k'] else '.':^4}"
    if retrieval_only:
        print(f"{hits}  {r['question']}")
        if not r["hit@k"]:
            print(f"          MISS: expected {r['expected_source']}, got {r['retrieved_sources']}")
        return
    print(f"{hits}{r['faithfulness']:^3}{r['relevance']:^3}{r['correctness']:^3}  {r['question']}")
    # Surface anything a human should read, with the judge's reason
    if r["faithfulness"] <= 3:
        print(f"          faithfulness: {r['faithfulness_reason']}")
    if r["correctness"] <= 3:
        print(f"          correctness:  {r['correctness_reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="score retrieval only (hit@1, hit@k), skip generation and the judge")
    args = parser.parse_args()
    k = settings.top_k

    items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    print("Building index...")
    print(f"Indexed {build_index()} chunks.\n")

    header = f"{'@1':^4}{'@' + str(k):^4}"
    if args.retrieval_only:
        print(f"{header}  question")
    else:
        print(f"{header}{'F':^3}{'R':^3}{'C':^3}  question   (F/R/C = faithfulness/relevance/correctness, 1-5)")
    print("-" * 80)

    rows = []
    for item in items:
        row = evaluate_retrieval(item, k) if args.retrieval_only else evaluate_full(item, k)
        rows.append(row)
        print_row(row, args.retrieval_only)

    summary = summarize(rows, args.retrieval_only)
    print("-" * 80)
    for key, value in summary.items():
        if key == "n":
            continue
        print(f"{key:<16} {value:.2f}s" if key == "avg_latency_s" else f"{key:<16} {value:.0%}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "retrieval" if args.retrieval_only else "full"
    out_path = RESULTS_DIR / f"{stamp}-{mode}.json"
    out_path.write_text(json.dumps({
        "timestamp": stamp,
        "mode": mode,
        "config": {
            "top_k": k,
            "embed_model": settings.embed_model,
            "generation_model": settings.generation_model,
            "judge_model": None if args.retrieval_only else settings.judge_model,
        },
        "summary": summary,
        "results": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path.relative_to(EVALS_DIR.parent)}")


if __name__ == "__main__":
    main()
