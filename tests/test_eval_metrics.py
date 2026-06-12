"""Tests for eval/metrics.py: categories, confidence intervals, baseline deltas."""

import pytest

from eval.metrics import (
    CATEGORY_IN_INDEX,
    CATEGORY_NEGATIVE,
    CATEGORY_NOVEL,
    bootstrap_median_ci,
    categorize_row,
    compare_to_baseline,
    summarize_by_category,
    summarize_negative,
    summarize_positive,
    wilson_interval,
)


class TestCategorizeRow:
    def test_negative(self):
        assert categorize_row({"expected_reject": 1}) == CATEGORY_NEGATIVE

    def test_in_index(self):
        row = {"expected_reject": 0, "expected_panorama_id": 55}
        assert categorize_row(row) == CATEGORY_IN_INDEX

    def test_novel(self):
        row = {"expected_reject": 0, "expected_panorama_id": None, "expected_lat": 37.7}
        assert categorize_row(row) == CATEGORY_NOVEL


class TestWilsonInterval:
    def test_empty(self):
        assert wilson_interval(0, 0) == {"low": 0.0, "high": 0.0}

    def test_bounds_contain_point_estimate(self):
        ci = wilson_interval(80, 100)
        assert ci["low"] < 80.0 < ci["high"]
        assert 0.0 <= ci["low"] and ci["high"] <= 100.0

    def test_narrower_with_more_samples(self):
        narrow = wilson_interval(800, 1000)
        wide = wilson_interval(8, 10)
        assert (narrow["high"] - narrow["low"]) < (wide["high"] - wide["low"])


class TestBootstrapMedianCI:
    def test_deterministic_for_seed(self):
        values = [5.0, 10.0, 12.0, 30.0, 100.0]
        assert bootstrap_median_ci(values, seed=7) == bootstrap_median_ci(values, seed=7)

    def test_contains_median(self):
        values = list(range(1, 101))
        ci = bootstrap_median_ci([float(v) for v in values], seed=42)
        assert ci["low"] <= 50.0 <= ci["high"] + 1.0

    def test_single_value(self):
        assert bootstrap_median_ci([4.2]) == {"low": 4.2, "high": 4.2}


def positive_row(error_m, panorama_top1=None, expected_panorama_id=None):
    return {
        "expected_reject": 0,
        "error_m": error_m,
        "panorama_top1": panorama_top1,
        "expected_panorama_id": expected_panorama_id,
    }


class TestSummarizePositive:
    def test_empty(self):
        summary = summarize_positive([])
        assert summary["positive_cases"] == 0
        assert summary["within_50m"] == 0.0

    def test_within_percentages(self):
        rows = [positive_row(e) for e in [10.0, 20.0, 60.0, 500.0]]
        summary = summarize_positive(rows)
        assert summary["within_25m"] == 50.0
        assert summary["within_50m"] == 50.0
        assert summary["within_100m"] == 75.0
        assert summary["median_error_m"] == 60.0  # upper median of sorted list
        assert summary["within_50m_ci95"]["low"] < 50.0 < summary["within_50m_ci95"]["high"]

    def test_missing_error_counts_as_miss(self):
        rows = [positive_row(10.0), positive_row(None)]
        summary = summarize_positive(rows)
        assert summary["within_25m"] == 50.0

    def test_zero_error_counts_as_hit(self):
        rows = [positive_row(0.0), positive_row(500.0)]
        summary = summarize_positive(rows)
        assert summary["within_25m"] == 50.0


class TestSummarizeNegative:
    def test_reject_counted_when_no_match(self):
        rows = [{"top_family_id": ""}, {"top_family_id": "family-1", "top_family_score": 0.9}]
        summary = summarize_negative(rows, None)
        assert summary["reject_rate"] == 50.0
        assert summary["false_accept_rate"] == 50.0

    def test_threshold_rejects_low_scores(self):
        rows = [{"top_family_id": "family-1", "top_family_score": 0.2}]
        assert summarize_negative(rows, 0.5)["reject_rate"] == 100.0
        assert summarize_negative(rows, 0.1)["reject_rate"] == 0.0


class TestSummarizeByCategory:
    def test_groups_and_summarizes(self):
        rows = [
            positive_row(10.0, expected_panorama_id=5),
            positive_row(500.0, expected_panorama_id=None),
            {"expected_reject": 1, "top_family_id": ""},
        ]
        out = summarize_by_category(rows, None)
        assert set(out) == {CATEGORY_IN_INDEX, CATEGORY_NOVEL, CATEGORY_NEGATIVE}
        assert out[CATEGORY_IN_INDEX]["positive_cases"] == 1
        assert out[CATEGORY_IN_INDEX]["within_25m"] == 100.0
        assert out[CATEGORY_NOVEL]["within_100m"] == 0.0
        assert out[CATEGORY_NEGATIVE]["reject_rate"] == 100.0


class TestCompareToBaseline:
    def test_improvement_directions(self):
        current = {"within_50m": 60.0, "median_error_m": 40.0}
        baseline = {"within_50m": 50.0, "median_error_m": 80.0}
        cmp = compare_to_baseline(current, baseline)
        assert cmp["within_50m"]["improved"] is True
        assert cmp["within_50m"]["delta"] == pytest.approx(10.0)
        assert cmp["median_error_m"]["improved"] is True  # lower is better

    def test_regression_flagged(self):
        cmp = compare_to_baseline({"within_50m": 40.0}, {"within_50m": 50.0})
        assert cmp["within_50m"]["improved"] is False

    def test_missing_metrics_skipped_and_zero_delta_neutral(self):
        cmp = compare_to_baseline({"within_50m": 50.0}, {"within_50m": 50.0})
        assert cmp["within_50m"]["improved"] is None
        assert "median_error_m" not in cmp
