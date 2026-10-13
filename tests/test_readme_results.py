"""The README results table is generated from the baseline, never typed by hand."""
from evals.readme_results import render


def test_render_builds_a_markdown_table_from_the_baseline():
    baseline = {
        "timestamp": "20261013-101500",
        "config": {"top_k": 3, "generation_model": "gpt-4o-mini", "judge_model": "gpt-4o",
                   "chunk_size": 500, "chunk_overlap": 100},
        "summary": {"n": 20, "hit_rate@1": 0.85, "hit_rate@k": 1.0, "p95_latency_s": 2.1,
                    "cost_per_1k_usd": 0.101},
    }
    thresholds = {"hit_rate@1": {"min": 0.7}, "p95_latency_s": {"max": 5.0}}
    table = render(baseline, thresholds, n_docs=7)
    assert "| Hit rate @1 | the right document is ranked first | **85%** | ≥ 70% |" in table
    assert "| Hit rate @3 | the right document is in the top 3 | **100%** | - |" in table
    assert "**2.10s** | ≤ 5.00s |" in table
    assert "**$0.101**" in table
    assert "20 hand-written questions over 7 documents" in table
    assert "Baseline run 2026-10-13." in table
