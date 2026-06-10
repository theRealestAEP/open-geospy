"""Pure metric helpers for locator evals: categories, confidence intervals,
and baseline comparison. Kept free of I/O so they are unit-testable."""

import math
import random
from typing import Dict, List, Optional, Sequence

# Case categories (derived, no manifest changes needed):
# - negative:       expected_reject=1 (system should refuse).
# - in_index_pano:  query pano exists in the index (different render of an
#                   indexed panorama; tests recognition).
# - novel_pano:     query pano NOT in the index but has ground-truth coords
#                   (tests spatial generalization to unseen panoramas).
CATEGORY_NEGATIVE = "negative"
CATEGORY_IN_INDEX = "in_index_pano"
CATEGORY_NOVEL = "novel_pano"


def categorize_row(row: dict) -> str:
    if int(row.get("expected_reject") or 0):
        return CATEGORY_NEGATIVE
    if row.get("expected_panorama_id") is not None:
        return CATEGORY_IN_INDEX
    return CATEGORY_NOVEL


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Dict[str, float]:
    """95% Wilson score interval for a proportion, in percent."""
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    n = float(total)
    p = float(successes) / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / denom
    return {
        "low": round(100.0 * max(0.0, center - half), 2),
        "high": round(100.0 * min(1.0, center + half), 2),
    }


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    z_alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile-bootstrap CI for the median. Deterministic for a given seed."""
    data = sorted(float(v) for v in values)
    if not data:
        return {"low": 0.0, "high": 0.0}
    if len(data) == 1:
        return {"low": round(data[0], 2), "high": round(data[0], 2)}
    rng = random.Random(seed)
    n = len(data)
    medians = []
    for _ in range(max(100, int(n_boot))):
        sample = sorted(rng.choice(data) for _ in range(n))
        medians.append(sample[n // 2])
    medians.sort()
    lo_idx = int((z_alpha / 2.0) * len(medians))
    hi_idx = min(len(medians) - 1, int((1.0 - z_alpha / 2.0) * len(medians)))
    return {"low": round(medians[lo_idx], 2), "high": round(medians[hi_idx], 2)}


def summarize_positive(rows: List[dict], *, seed: int = 42) -> Dict[str, object]:
    total = len(rows)
    if total <= 0:
        return {
            "positive_cases": 0,
            "within_25m": 0.0,
            "within_50m": 0.0,
            "within_100m": 0.0,
            "panorama_top1": 0.0,
            "capture_top1": 0.0,
            "median_error_m": 0.0,
        }
    errors = sorted(
        float(row["error_m"]) for row in rows if row.get("error_m") is not None
    )

    def within(threshold: float) -> Dict[str, object]:
        hits = sum(
            1 for row in rows if float(row.get("error_m") or 1e18) <= threshold
        )
        return {
            "pct": round(100.0 * hits / total, 2),
            "ci95": wilson_interval(hits, total),
        }

    def top1_pct(field: str) -> float:
        eligible = [row for row in rows if row.get(field) is not None]
        if not eligible:
            return 0.0
        return round(
            100.0 * sum(1 for row in eligible if int(row.get(field) or 0)) / len(eligible),
            2,
        )

    w25, w50, w100 = within(25.0), within(50.0), within(100.0)
    median_error = errors[len(errors) // 2] if errors else 0.0
    return {
        "positive_cases": total,
        "within_25m": w25["pct"],
        "within_25m_ci95": w25["ci95"],
        "within_50m": w50["pct"],
        "within_50m_ci95": w50["ci95"],
        "within_100m": w100["pct"],
        "within_100m_ci95": w100["ci95"],
        "panorama_top1": top1_pct("panorama_top1"),
        "capture_top1": top1_pct("capture_top1"),
        "median_error_m": round(float(median_error), 2),
        "median_error_m_ci95": bootstrap_median_ci(errors, seed=seed),
    }


def summarize_negative(
    rows: List[dict], reject_family_score_threshold: Optional[float]
) -> Dict[str, object]:
    total = len(rows)
    if total <= 0:
        return {
            "negative_cases": 0,
            "reject_rate": 0.0,
            "false_accept_rate": 0.0,
        }
    correct_rejects = 0
    for row in rows:
        has_match = bool(row.get("top_family_id"))
        top_score = row.get("top_family_score")
        rejected = not has_match
        if (
            not rejected
            and reject_family_score_threshold is not None
            and top_score is not None
            and float(top_score) < float(reject_family_score_threshold)
        ):
            rejected = True
        if rejected:
            correct_rejects += 1
    reject_rate = round(100.0 * correct_rejects / total, 2)
    return {
        "negative_cases": total,
        "reject_rate": reject_rate,
        "reject_rate_ci95": wilson_interval(correct_rejects, total),
        "false_accept_rate": round(100.0 - reject_rate, 2),
    }


def summarize_by_category(
    rows: List[dict],
    reject_family_score_threshold: Optional[float],
    *,
    seed: int = 42,
) -> Dict[str, dict]:
    grouped: Dict[str, List[dict]] = {}
    for row in rows:
        grouped.setdefault(categorize_row(row), []).append(row)
    out: Dict[str, dict] = {}
    for category, group in sorted(grouped.items()):
        if category == CATEGORY_NEGATIVE:
            out[category] = summarize_negative(group, reject_family_score_threshold)
        else:
            out[category] = summarize_positive(group, seed=seed)
    return out


# Metrics where larger is better / smaller is better, used for baseline deltas.
HIGHER_IS_BETTER = ("within_25m", "within_50m", "within_100m", "panorama_top1", "capture_top1", "reject_rate")
LOWER_IS_BETTER = ("median_error_m", "false_accept_rate")


def compare_to_baseline(current: dict, baseline: dict) -> Dict[str, dict]:
    """Per-metric deltas vs. a baseline summary. Positive `delta` means the
    raw value increased; `improved` interprets direction per metric."""
    comparison: Dict[str, dict] = {}
    for metric in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER):
        cur = current.get(metric)
        base = baseline.get(metric)
        if cur is None or base is None:
            continue
        delta = round(float(cur) - float(base), 2)
        improved = delta > 0 if metric in HIGHER_IS_BETTER else delta < 0
        comparison[metric] = {
            "current": float(cur),
            "baseline": float(base),
            "delta": delta,
            "improved": bool(improved) if delta != 0 else None,
        }
    return comparison
