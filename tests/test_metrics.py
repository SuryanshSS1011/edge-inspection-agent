"""Tests for the eval metrics: confusion accounting, cost-weighted recall, latency
percentiles, per-item rates, and the bootstrap CI."""

import pytest

from edge.router import Costs
from eval.metrics import (
    bootstrap_ci,
    confusion,
    cost_weighted_recall,
    cloud_cost_per_1k,
    incurred_cost,
    latency_percentiles,
    pii_bytes_out,
    worst_case_cost,
)

COSTS = Costs(C_FN=100.0, C_FP=5.0, C_cloud=2.0, residual_cloud_error=0.3)


def test_confusion_counts():
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 1, 0]  # tp, fn, fp, tn
    assert confusion(y_true, y_pred) == (1, 1, 1, 1)


def test_perfect_prediction_scores_one():
    y = [1, 0, 1, 0]
    assert cost_weighted_recall(y, y, COSTS) == pytest.approx(1.0)


def test_worst_prediction_scores_zero():
    y_true = [1, 0]
    y_pred = [0, 1]  # miss the defect, reject the good = worst case
    assert cost_weighted_recall(y_true, y_pred, COSTS) == pytest.approx(0.0)


def test_missing_defect_costs_more_than_false_alarm():
    # One missed defect vs one false alarm: the miss must score lower.
    miss = cost_weighted_recall([1, 0], [0, 0], COSTS)
    false_alarm = cost_weighted_recall([1, 0], [1, 1], COSTS)
    assert miss < false_alarm


def test_incurred_and_worst_case_cost():
    y_true = [1, 1, 0]
    y_pred = [0, 1, 1]  # one FN, one FP
    assert incurred_cost(y_true, y_pred, COSTS) == pytest.approx(105.0)
    assert worst_case_cost(y_true, COSTS) == pytest.approx(2 * 100.0 + 1 * 5.0)


def test_all_good_items_never_divide_by_zero():
    assert cost_weighted_recall([0, 0], [0, 0], COSTS) == pytest.approx(1.0)


def test_latency_percentiles_nearest_rank():
    lat = [10, 20, 30, 40, 100]
    out = latency_percentiles(lat, pcts=(50, 99))
    assert out["p50"] == 30
    assert out["p99"] == 100


def test_latency_empty():
    assert latency_percentiles([]) == {"p50": 0.0, "p99": 0.0}


def test_cloud_cost_per_1k():
    # 100 escalations over 1000 items at $2 each -> $200 per 1k items.
    assert cloud_cost_per_1k(100, 1000, 2.0) == pytest.approx(200.0)


def test_pii_excludes_blocked():
    rows = [
        {"nbytes": 50, "is_pii": 1, "blocked": 1},  # caught -> excluded
        {"nbytes": 30, "is_pii": 1, "blocked": 0},  # leaked -> counts
        {"nbytes": 99, "is_pii": 0, "blocked": 0},  # not pii
    ]
    assert pii_bytes_out(rows) == 30


def test_bootstrap_ci_brackets_mean():
    vals = [0.8, 0.82, 0.79, 0.81, 0.80]
    lo, mean, hi = bootstrap_ci(vals, n_boot=500, seed=1)
    assert lo <= mean <= hi
    assert mean == pytest.approx(sum(vals) / len(vals))
