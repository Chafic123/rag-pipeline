"""Evaluate the RAG pipeline, compare with the baseline, and gate on thresholds.

    make eval              full run: retrieval + LLM judge on every question
    make eval-quick        the same, first 3 questions only (smoke test, never fails the gate)
    make eval-retrieval    retrieval only (hit@1, hit@k), no generation, almost free
    make eval-baseline     full run, then save it as the new baseline

    (same as: python -m evals.run_evals [--limit N] [--retrieval-only] [--set-baseline])

Every run prints, per question:
    retrieval    hit@1 and hit@k: did the expected source document come back?
    generation   faithfulness, answer relevance, correctness (LLM judge, 1-5)
    speed & cost latency per stage (embed, search, generate), tokens, dollars
then one report: each metric, its baseline value, the change, and whether it meets
evals/thresholds.json. A full run that misses a threshold exits with code 1, so CI can
block the change. Every run is also saved to evals/results/<timestamp>-<mode>.json.

NOTE: stop the API server first. Qdrant local mode allows one process at a time.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.rag.pipeline import answer_with_context
from app.rag.store import build_index, retrieve
from evals import judge
from evals.report import (check_thresholds, compare, config_changes, format_report,
                          load_json, stage_breakdown, summarize)

EVALS_DIR = Path(__file__).resolve().parent
EVAL_SET = EVALS_DIR / "eval_set.json"
RESULTS_DIR = EVALS_DIR / "results"
BASELINE = EVALS_DIR / "baseline.json"        # the accepted numbers, committed to git
THRESHOLDS = EVALS_DIR / "thresholds.json"    # the minimum quality to pass


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
    out = answer_with_context(question, k=k)   # the pipeline measures its own latency and cost

    faith = judge.faithfulness(out["contexts"], out["answer"])
    relev = judge.answer_relevance(question, out["answer"])
    corr = judge.correctness(question, out["answer"], item["reference_answer"])

    return {
        "question": question,
        "expected_source": item["expected_source"],
        "retrieved_sources": out["sources"],
        **_hits(out["sources"], item["expected_source"]),
        "answer": out["answer"],
        **out["metrics"],   # embed/search/generate/total ms, tokens, cost_usd
        "faithfulness": faith.score, "faithfulness_reason": faith.reason,
        "relevance": relev.score, "relevance_reason": relev.reason,
        "correctness": corr.score, "correctness_reason": corr.reason,
    }


def print_row(r: dict, retrieval_only: bool) -> None:
    hits = f"{'Y' if r['hit@1'] else '.':^4}{'Y' if r['hit@k'] else '.':^4}"
    if retrieval_only:
        print(f"{hits}  {r['question']}")
        if not r["hit@k"]:
            print(f"          MISS: expected {r['expected_source']}, got {r['retrieved_sources']}")
        return
    print(f"{hits}{r['faithfulness']:^3}{r['relevance']:^3}{r['correctness']:^3}"
          f"{r['total_ms']:>7.0f}  {r['question']}")
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
    parser.add_argument("--set-baseline", action="store_true",
                        help="save this run as the new baseline (needs the full question set)")
    args = parser.parse_args()
    if args.set_baseline and args.limit:
        parser.error("--set-baseline needs the full question set, drop --limit")

    k = settings.top_k
    mode = "retrieval" if args.retrieval_only else "full"
    config = {   # the setup this run measured, saved with it so runs can be compared
        "top_k": k,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "embed_model": settings.embed_model,
        "generation_model": None if args.retrieval_only else settings.generation_model,
        "judge_model": None if args.retrieval_only else settings.judge_model,
    }

    items = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    print("Building index...")
    print(f"Indexed {build_index()} chunks.\n")

    header = f"{'@1':^4}{'@' + str(k):^4}"
    if args.retrieval_only:
        print(f"{header}  question")
    else:
        print(f"{header}{'F':^3}{'R':^3}{'C':^3}{'ms':>7}  question   (F/R/C = faithfulness/relevance/correctness, 1-5)")
    print("-" * 80)

    rows = []
    for item in items:
        row = evaluate_retrieval(item, k) if args.retrieval_only else evaluate_full(item, k)
        rows.append(row)
        print_row(row, args.retrieval_only)

    summary = summarize(rows, args.retrieval_only)
    baselines = load_json(BASELINE) or {}
    baseline = baselines.get(mode)
    deltas = compare(summary, baseline)
    checks = check_thresholds(summary, load_json(THRESHOLDS) or {})
    gating = not args.limit          # a 3-question smoke test is too small to judge on
    failed = gating and any(not c["passed"] for c in checks)

    print("-" * 80 + "\n")
    print(format_report(summary, baseline, deltas, checks,
                        config_changes(config, baseline), k, gating))

    extras = {}
    if not args.retrieval_only:
        stages = stage_breakdown(rows)
        answers_cost = sum(r["cost_usd"] for r in rows)
        judge_cost = judge.judge_cost_usd()
        extras = {"stage_avg_ms": stages,
                  "run_cost_usd": {"answers": answers_cost, "judge": judge_cost}}
        print("\nLatency by stage (avg): " + " | ".join(f"{s} {ms:,.0f} ms" for s, ms in stages.items()))
        print(f"Cost of this run: answers ${answers_cost:.4f} ({len(rows)} queries) + judge "
              f"${judge_cost:.4f} ({judge.token_totals['calls']} calls) = ${answers_cost + judge_cost:.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"{stamp}-{mode}.json"
    out_path.write_text(json.dumps({
        "timestamp": stamp,
        "mode": mode,
        "config": config,
        "summary": summary,
        "gate": {"gating": gating, "passed": not failed, "checks": checks},
        **extras,
        "results": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path.relative_to(EVALS_DIR.parent)}")

    if args.set_baseline:
        baselines[mode] = {"timestamp": stamp, "config": config, "summary": summary}
        BASELINE.write_text(json.dumps(baselines, indent=2) + "\n", encoding="utf-8")
        print(f"Baseline for '{mode}' runs updated. Commit evals/baseline.json "
              "so everyone compares against the same numbers.")

    if failed:
        sys.exit(1)   # non-zero exit = a failed check, the same signal as a failing test


if __name__ == "__main__":
    main()
