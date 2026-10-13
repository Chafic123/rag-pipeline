"""Turn per-question eval rows into numbers you can act on.

    summarize         per-question rows  ->  one summary (hit rates, judge scores, latency, cost)
    stage_breakdown   average milliseconds per pipeline stage (embed, search, generate)
    compare           summary vs the saved baseline  ->  change per metric
    check_thresholds  summary vs evals/thresholds.json  ->  pass / fail per metric
    format_report     all of the above  ->  the table printed after every run

Pure functions only, no API calls, so tests/test_eval_report.py covers them
without an OpenAI key.
"""
import json
import math
from pathlib import Path

SECONDS = {"avg_latency_s", "p50_latency_s", "p95_latency_s"}
DOLLARS = {"cost_per_1k_usd"}
# Metrics where a smaller number is better. For everything else, bigger is better.
LOWER_IS_BETTER = SECONDS | DOLLARS


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the value that p% of the values are less than or equal to.
    p95 of 20 latencies is the 19th fastest: 95% of queries were at least this fast."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))
    return ordered[max(rank, 1) - 1]


def summarize(rows: list[dict], retrieval_only: bool) -> dict:
    n = len(rows)
    if n == 0:
        raise ValueError("no eval rows to summarize")
    summary = {
        "n": n,
        "hit_rate@1": sum(r["hit@1"] for r in rows) / n,
        "hit_rate@k": sum(r["hit@k"] for r in rows) / n,
    }
    if not retrieval_only:
        def normalized(key):   # map the 1-5 rubric onto 0-1 so it reads as a percentage
            return sum((r[key] - 1) / 4 for r in rows) / n
        latencies = [r["total_ms"] / 1000 for r in rows]
        summary.update({
            "faithfulness": normalized("faithfulness"),
            "relevance": normalized("relevance"),
            "correctness": normalized("correctness"),
            "avg_latency_s": sum(latencies) / n,
            "p50_latency_s": percentile(latencies, 50),   # the typical query
            "p95_latency_s": percentile(latencies, 95),   # the slow tail users notice
            # per query is a tiny number like $0.0001, per 1,000 queries reads better
            "cost_per_1k_usd": sum(r["cost_usd"] for r in rows) / n * 1000,
        })
    return summary


def stage_breakdown(rows: list[dict]) -> dict:
    """Average milliseconds per stage: shows WHERE the time goes."""
    n = len(rows)
    return {stage: sum(r[f"{stage}_ms"] for r in rows) / n
            for stage in ("embed", "search", "generate")}


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def compare(summary: dict, baseline: dict | None) -> dict:
    """Change versus the baseline, for every metric both runs have.

    Empty when there is no baseline, or when the two runs used a different number
    of questions: 3 questions against 20 is not a fair comparison.
    """
    if not baseline or baseline["summary"]["n"] != summary["n"]:
        return {}
    base = baseline["summary"]
    return {key: summary[key] - base[key] for key in summary if key != "n" and key in base}


def config_changes(current: dict, baseline: dict | None) -> list[str]:
    """What changed in the setup since the baseline, e.g. 'top_k: 3 -> 5'.
    Explains WHY the numbers moved."""
    if not baseline:
        return []
    before = baseline.get("config", {})
    return [f"{key}: {before.get(key)} -> {value}"
            for key, value in current.items() if before.get(key) != value]


def check_thresholds(summary: dict, thresholds: dict) -> list[dict]:
    """One check per threshold this run can be judged on.

    thresholds look like {"faithfulness": {"min": 0.8}, "avg_latency_s": {"max": 5.0}}.
    Metrics the run did not produce (judge scores in a retrieval-only run) are skipped.
    """
    checks = []
    for metric, rule in thresholds.items():
        if metric not in summary:
            continue
        value = round(summary[metric], 4)   # 0.79999999 from float math still counts as 0.8
        passed = rule.get("min", float("-inf")) <= value <= rule.get("max", float("inf"))
        checks.append({"metric": metric, "value": value, "rule": rule, "passed": passed})
    return checks


# ---------- printing ----------

def format_value(metric: str, value: float) -> str:
    if metric in SECONDS:
        return f"{value:.2f}s"
    if metric in DOLLARS:
        return f"${value:.3f}"
    return f"{value:.0%}"


def _change(metric: str, delta: float | None) -> tuple[str, str]:
    """('+5 pts', 'better') style text for a change. pts = percentage points."""
    if delta is None:
        return "-", ""
    if metric in SECONDS:
        text, size = f"{delta:+.2f}s", round(delta, 2)
    elif metric in DOLLARS:
        size = round(delta, 3)
        text = f"{'+' if delta >= 0 else '-'}${abs(delta):.3f}"
    else:
        size = round(delta * 100)
        text = f"{size:+d} pts"
    if size == 0:
        return "0", ""
    improved = (size < 0) if metric in LOWER_IS_BETTER else (size > 0)
    return text, "better" if improved else "worse"


def format_rule(metric: str, rule: dict) -> str:
    parts = []
    if "min" in rule:
        parts.append(f">= {format_value(metric, rule['min'])}")
    if "max" in rule:
        parts.append(f"<= {format_value(metric, rule['max'])}")
    return " and ".join(parts)


def format_report(summary: dict, baseline: dict | None, deltas: dict, checks: list[dict],
                  changes: list[str], k: int, gating: bool) -> str:
    lines = []

    if baseline is None:
        lines.append("No baseline yet. Save one with --set-baseline (for full runs: make eval-baseline).")
    elif not deltas:
        lines.append(f"Baseline has {baseline['summary']['n']} questions, this run has "
                     f"{summary['n']}: change not shown.")
    else:
        lines.append(f"Compared with baseline from {baseline['timestamp']}.")
    if deltas and changes:
        lines.append("Setup changed since baseline: " + ", ".join(changes))
    lines.append("")

    by_metric = {c["metric"]: c for c in checks}
    lines.append(f"{'metric':<16}{'this run':>9}{'baseline':>10}{'change':>10}  {'':<7}{'gate':<17}")
    lines.append("-" * 72)
    for metric, value in summary.items():
        if metric == "n":
            continue
        base = baseline["summary"].get(metric) if (baseline and deltas) else None
        change, direction = _change(metric, deltas.get(metric))
        check = by_metric.get(metric)
        gate = f"{format_rule(metric, check['rule']):<9}  {'PASS' if check['passed'] else 'FAIL'}" if check else ""
        name = metric.replace("@k", f"@{k}")
        lines.append(f"{name:<16}{format_value(metric, value):>9}"
                     f"{format_value(metric, base) if base is not None else '-':>10}"
                     f"{change:>10}  {direction:<7}{gate}")
    lines.append("-" * 72)

    failed = [c["metric"].replace("@k", f"@{k}") for c in checks if not c["passed"]]
    if not checks:
        lines.append("RESULT: no thresholds apply to this run.")
    elif not gating:
        lines.append(f"RESULT: {len(checks) - len(failed)}/{len(checks)} checks met "
                     "(partial run with --limit, not gating).")
    elif failed:
        lines.append(f"RESULT: FAIL, {len(failed)}/{len(checks)} below threshold: {', '.join(failed)}")
    else:
        lines.append(f"RESULT: PASS, all {len(checks)} checks met.")
    return "\n".join(line.rstrip() for line in lines)
