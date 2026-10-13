"""Unit tests for the eval report math: summary, baseline change, threshold gate.
No API key needed, these never call the model."""
import pytest

from evals.report import (check_thresholds, compare, config_changes, format_report, percentile,
                          stage_breakdown, summarize)


def _row(hit1=True, hitk=True, faith=5, relev=5, corr=5, total_ms=1000.0, cost=0.0001):
    return {"hit@1": hit1, "hit@k": hitk, "faithfulness": faith, "relevance": relev,
            "correctness": corr, "total_ms": total_ms, "cost_usd": cost,
            "embed_ms": 200.0, "search_ms": 5.0, "generate_ms": total_ms - 205.0}


def _baseline(summary=None):
    return {"timestamp": "20261013-090000", "config": {"top_k": 3}, "summary": summary or {}}


def test_summarize_maps_judge_scores_onto_0_to_1():
    rows = [_row(faith=5, hit1=True), _row(faith=3, hit1=False)]
    s = summarize(rows, retrieval_only=False)
    assert s["n"] == 2
    assert s["hit_rate@1"] == 0.5
    assert s["faithfulness"] == 0.75        # (1.0 + 0.5) / 2: score 5 -> 1.0, score 3 -> 0.5


def test_summarize_retrieval_only_has_no_judge_metrics():
    s = summarize([_row()], retrieval_only=True)
    assert set(s) == {"n", "hit_rate@1", "hit_rate@k"}


def test_summarize_rejects_an_empty_run():
    with pytest.raises(ValueError):
        summarize([], retrieval_only=True)


def test_compare_gives_the_change_per_metric():
    deltas = compare({"n": 20, "hit_rate@1": 0.85, "avg_latency_s": 1.5},
                     _baseline({"n": 20, "hit_rate@1": 0.80, "avg_latency_s": 2.0}))
    assert deltas["hit_rate@1"] == pytest.approx(0.05)
    assert deltas["avg_latency_s"] == pytest.approx(-0.5)


def test_compare_refuses_runs_of_different_sizes():
    assert compare({"n": 3, "hit_rate@1": 1.0}, _baseline({"n": 20, "hit_rate@1": 0.8})) == {}


def test_compare_without_a_baseline_is_empty():
    assert compare({"n": 20, "hit_rate@1": 1.0}, None) == {}


def test_config_changes_names_what_moved():
    assert config_changes({"top_k": 5}, _baseline()) == ["top_k: 3 -> 5"]
    assert config_changes({"top_k": 3}, _baseline()) == []


def test_thresholds_handle_min_and_max():
    checks = check_thresholds({"faithfulness": 0.9, "avg_latency_s": 6.0},
                              {"faithfulness": {"min": 0.8}, "avg_latency_s": {"max": 5.0}})
    passed = {c["metric"]: c["passed"] for c in checks}
    assert passed == {"faithfulness": True, "avg_latency_s": False}


def test_thresholds_skip_metrics_the_run_did_not_produce():
    checks = check_thresholds({"hit_rate@1": 0.9}, {"faithfulness": {"min": 0.8}})
    assert checks == []


def test_thresholds_ignore_float_noise_at_the_boundary():
    checks = check_thresholds({"correctness": 0.7999999999}, {"correctness": {"min": 0.8}})
    assert checks[0]["passed"]


def test_report_fails_a_full_run_but_not_a_partial_one():
    summary = {"n": 20, "hit_rate@1": 0.5}
    checks = check_thresholds(summary, {"hit_rate@1": {"min": 0.7}})
    full = format_report(summary, None, {}, checks, [], k=3, gating=True)
    partial = format_report(summary, None, {}, checks, [], k=3, gating=False)
    assert "RESULT: FAIL" in full
    assert "not gating" in partial


def test_percentile_uses_nearest_rank():
    latencies = [float(i) for i in range(1, 21)]      # 1..20 seconds
    assert percentile(latencies, 50) == 10.0
    assert percentile(latencies, 95) == 19.0           # 95% of queries took <= 19s
    assert percentile([3.0], 95) == 3.0


def test_summarize_reports_latency_percentiles_and_cost_per_1k():
    rows = [_row(total_ms=1000, cost=0.0001)] * 19 + [_row(total_ms=9000, cost=0.0003)]
    s = summarize(rows, retrieval_only=False)
    assert s["p50_latency_s"] == 1.0
    assert s["p95_latency_s"] == 1.0                   # one slow query out of 20 is above p95
    assert s["avg_latency_s"] == pytest.approx(1.4)    # but it pulls the average up
    assert s["cost_per_1k_usd"] == pytest.approx(0.11)


def test_stage_breakdown_averages_each_stage():
    stages = stage_breakdown([_row(total_ms=1205), _row(total_ms=2205)])
    assert stages == {"embed": 200.0, "search": 5.0, "generate": 1500.0}


def test_report_shows_dollars_and_a_cost_gate():
    summary = {"n": 20, "cost_per_1k_usd": 0.62}
    checks = check_thresholds(summary, {"cost_per_1k_usd": {"max": 0.50}})
    report = format_report(summary, None, {}, checks, [], k=3, gating=True)
    assert "$0.620" in report and "<= $0.500" in report and "RESULT: FAIL" in report
