# evaluate.py
# Measure RETRIEVAL quality against a fixed ground-truth set (eval_set.json).
#
# Metric: hit-rate@k = fraction of questions where the correct source document
# appears in the top-k retrieved chunks. This measures retrieval ALONE (no LLM),
# so it is cheap, fast, and gives the same number every run.
#
# Why retrieval first? It sets the ceiling: if the right chunk is never retrieved,
# the model cannot possibly answer correctly.

import json

from rag_store import build_index, retrieve

EVAL_PATH = "eval_set.json"


def load_eval():
    with open(EVAL_PATH, encoding="utf-8") as f:
        return json.load(f)


def sources_for(question, k):
    """Return the source filenames of the top-k retrieved chunks, best first."""
    results = retrieve(question, k=k)
    return [r.payload["source"] for r in results]


def evaluate(eval_items, k=3):
    hits_at_1 = 0
    hits_at_k = 0

    print(f"{'@1':^4}{'@'+str(k):^5}  question")
    print("-" * 72)
    for item in eval_items:
        question = item["question"]
        expected = item["expected_source"]
        srcs = sources_for(question, k=k)

        hit1 = bool(srcs) and srcs[0] == expected      # correct doc is the TOP result
        hitk = expected in srcs                          # correct doc is anywhere in top k
        hits_at_1 += hit1
        hits_at_k += hitk

        print(f"{'Y' if hit1 else '.':^4}{'Y' if hitk else '.':^5}  {question}")
        if not hitk:
            print(f"          MISS: expected {expected}, got {srcs}")

    n = len(eval_items)
    print("-" * 72)
    print(f"hit-rate@1  = {hits_at_1}/{n} = {hits_at_1 / n:.0%}")
    print(f"hit-rate@{k}  = {hits_at_k}/{n} = {hits_at_k / n:.0%}")


if __name__ == "__main__":
    print("Building index...")
    count = build_index()
    print(f"Indexed {count} chunks from docs/.\n")
    evaluate(load_eval(), k=3)
