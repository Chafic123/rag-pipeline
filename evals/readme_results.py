"""Write the baseline numbers into the README results table.

    make readme-results      (run after make eval-baseline)

The table between the RESULTS markers in README.md is generated from
evals/baseline.json, so every number in the README comes from a real run.
"""
from pathlib import Path

from evals.report import format_rule, format_value, load_json

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
BASELINE = ROOT / "evals" / "baseline.json"
THRESHOLDS = ROOT / "evals" / "thresholds.json"
DOCS_DIR = ROOT / "data" / "docs"
START, END = "<!-- results:start -->", "<!-- results:end -->"

ROWS = [  # (summary key, label, what it measures)
    ("hit_rate@1", "Hit rate @1", "the right document is ranked first"),
    ("hit_rate@k", "Hit rate @{k}", "the right document is in the top {k}"),
    ("faithfulness", "Faithfulness", "every claim is supported by the retrieved context (LLM judge)"),
    ("relevance", "Answer relevance", "the answer addresses the question (LLM judge)"),
    ("correctness", "Correctness", "the answer agrees with the hand-written reference (LLM judge)"),
    ("p50_latency_s", "Latency p50", "typical end-to-end time per query"),
    ("p95_latency_s", "Latency p95", "95% of queries finish faster than this"),
    ("cost_per_1k_usd", "Cost per 1,000 queries", "embedding + generation tokens at list price"),
]


def render(baseline: dict, thresholds: dict, n_docs: int) -> str:
    summary, cfg = baseline["summary"], baseline["config"]
    k = cfg["top_k"]
    lines = ["| Metric | Measures | Result | Gate |", "|---|---|---:|---|"]
    for key, label, what in ROWS:
        if key not in summary:
            continue
        gate = format_rule(key, thresholds.get(key, {})).replace(">=", "≥").replace("<=", "≤") or "-"
        lines.append(f"| {label.format(k=k)} | {what.format(k=k)} | "
                     f"**{format_value(key, summary[key])}** | {gate} |")
    ts = baseline["timestamp"]
    lines += ["", f"{summary['n']} hand-written questions over {n_docs} documents. "
              f"Generator `{cfg['generation_model']}`, judge `{cfg['judge_model']}`, top_k = {k}, "
              f"chunks of {cfg['chunk_size']} characters with {cfg['chunk_overlap']} overlap. "
              f"Baseline run {ts[:4]}-{ts[4:6]}-{ts[6:8]}."]
    return "\n".join(lines)


def main() -> None:
    baseline = (load_json(BASELINE) or {}).get("full")
    if baseline is None:
        raise SystemExit("No full baseline yet. Run `make eval-baseline` first.")
    n_docs = sum(1 for p in DOCS_DIR.iterdir()
                 if p.suffix in {".txt", ".md"} and not p.name.startswith("_"))
    table = render(baseline, load_json(THRESHOLDS) or {}, n_docs)

    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"README.md needs the markers {START} and {END}.")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    README.write_text(f"{before}{START}\n{table}\n{END}{after}", encoding="utf-8")
    print("README results table updated from evals/baseline.json")


if __name__ == "__main__":
    main()
