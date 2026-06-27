"""Tests for temperature-scaling calibration. Pure numpy/scipy — no model needed.

The key property: fitting temperature on a miscalibrated (over-confident) logit set
should reduce expected calibration error.
"""

import numpy as np
import pytest

from edge.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    reliability_curve,
)


def _make_overconfident_set(n=4000, scale=3.0, seed=0):
    """Generate logits whose TRUE probability is sigmoid(logit), but inflate them by
    `scale` so the model is over-confident. fit_temperature should recover ~scale."""
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(0.0, 1.5, size=n)
    p_true = 1.0 / (1.0 + np.exp(-true_logit))
    labels = (rng.uniform(size=n) < p_true).astype(float)
    observed_logit = true_logit * scale   # over-confident model output
    return observed_logit, labels, scale


def test_apply_temperature_scalar_and_array():
    assert apply_temperature(0.0, 1.0) == pytest.approx(0.5)
    out = apply_temperature(np.array([0.0, 100.0, -100.0]), 1.0)
    assert out[0] == pytest.approx(0.5)
    assert out[1] == pytest.approx(1.0, abs=1e-6)
    assert out[2] == pytest.approx(0.0, abs=1e-6)


def test_apply_temperature_rejects_nonpositive():
    with pytest.raises(ValueError):
        apply_temperature(1.0, 0.0)


def test_fit_recovers_inflation_factor():
    logits, labels, scale = _make_overconfident_set()
    t = fit_temperature(logits, labels)
    # Dividing by T should undo the inflation: T ~ scale.
    assert t == pytest.approx(scale, rel=0.2)


def test_fit_reduces_calibration_error():
    logits, labels, _ = _make_overconfident_set()
    raw_p = apply_temperature(logits, 1.0)
    t = fit_temperature(logits, labels)
    cal_p = apply_temperature(logits, t)
    assert expected_calibration_error(cal_p, labels) < expected_calibration_error(raw_p, labels)


def test_fit_on_already_calibrated_returns_near_one():
    rng = np.random.default_rng(1)
    n = 4000
    logit = rng.normal(0.0, 1.5, size=n)
    p = 1.0 / (1.0 + np.exp(-logit))
    labels = (rng.uniform(size=n) < p).astype(float)
    t = fit_temperature(logit, labels)
    assert t == pytest.approx(1.0, abs=0.25)


def test_fit_rejects_empty():
    with pytest.raises(ValueError):
        fit_temperature(np.array([]), np.array([]))


def test_fit_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        fit_temperature(np.array([0.1, 0.2]), np.array([1.0]))


def test_ece_perfect_is_zero():
    # Predictions exactly equal to outcomes -> zero calibration error.
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)


def test_reliability_curve_shapes():
    probs = np.linspace(0, 1, 100)
    labels = (probs > 0.5).astype(float)
    conf, acc, cnt = reliability_curve(probs, labels, n_bins=10)
    assert len(conf) == len(acc) == len(cnt) == 10
    assert cnt.sum() == 100
