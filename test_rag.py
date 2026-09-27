# test_rag.py
# A tiny evaluation harness: run a fixed set of questions through the RAG
# pipeline and read where it fails. This is the SEED of real evaluation
# (RAGAS, LLM-as-judge) that comes in a few weeks. For now the goal is simple:
# run it, read the answers, and write down the failure modes by hand.
#
# NOTE: Qdrant local mode locks the qdrant_data folder to ONE process at a time.
# So stop the uvicorn server (Ctrl+C) before running this, or you'll hit a lock
# error (the same .lock file you saw earlier).

from rag_store import build_index, answer

# Each item: (question, category)
#   in_doc     -> the answer IS in docs/sample.txt; expect a good grounded answer
#   out_of_doc -> the answer is NOT in the docs; expect an honest "I don't know"
#   borderline -> named in passing but never really explained; the interesting case
TESTS = [
    ("What is an API?", "in_doc"),
    ("What is a backend and why does reliability matter there?", "in_doc"),
    ("How does retrieval-augmented generation work?", "in_doc"),
    ("What format do APIs usually return responses in?", "in_doc"),
    ("Why does a backend engineer have an advantage moving into AI?", "in_doc"),
    ("What is the capital of Japan?", "out_of_doc"),
    ("Who wrote Romeo and Juliet?", "out_of_doc"),
    ("What database should I use for my side project?", "out_of_doc"),
    ("What is load testing and how do I do it?", "borderline"),   # 'load testing' is named, never defined
    ("What exactly is an embedding, mathematically?", "borderline"),  # embeddings named, not defined deeply
]

# Very crude signal, NOT a real metric. Just to draw the eye to likely problems.
REFUSAL_HINTS = ("don't know", "do not know", "not in the context",
                 "cannot find", "no information", "isn't in", "is not in")


def looks_like_refusal(text):
    t = text.lower()
    return any(h in t for h in REFUSAL_HINTS)


def main():
    print("Building index...")
    n = build_index()
    print(f"Indexed {n} chunks.\n")
    print("=" * 72)

    for i, (question, category) in enumerate(TESTS, 1):
        reply = answer(question)
        print(f"\n[{i}] ({category})  {question}")
        print(f"    -> {reply}")

        # Auto-flags: these are hints, not judgments. You still read every answer.
        if category == "out_of_doc" and not looks_like_refusal(reply):
            print("    FLAG: expected a refusal but the model answered -> possible knowledge leak")
        if category == "in_doc" and looks_like_refusal(reply):
            print("    FLAG: refused on an in-doc question -> retrieval may have missed the chunk")

    print("\n" + "=" * 72)
    print("Read every answer above. Note by hand: hallucinations, over-refusals,")
    print("shallow answers, and anything the FLAGs caught.")


if __name__ == "__main__":
    main()
