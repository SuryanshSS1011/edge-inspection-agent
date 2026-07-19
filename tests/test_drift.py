"""Tests for calibration drift detection."""
import json
import tempfile

import numpy as np
import pytest

from edge.drift import DriftConfig, DriftDetector, DriftState


# unit tests for DriftDetector

def _ref():
    rng = np.random.default_rng(0)
    return rng.beta(2, 5, size=300).tolist()  # reference skewed toward 0


def test_ok_when_recent_matches_reference():
    ref = _ref()
    det = DriftDetector(ref, DriftConfig(window=100, check_every=1, ks_threshold=0.15))
    # recent drawn from same distribution
    rng = np.random.default_rng(1)
    recent = rng.beta(2, 5, size=100).tolist()
    state = det.check(recent)
    assert state == DriftState.OK


def test_alert_when_distribution_shifts():
    ref = _ref()
    det = DriftDetector(ref, DriftConfig(window=100, check_every=1, ks_threshold=0.15))
    # recent drawn from a very different distribution (uniform high confidence)
    rng = np.random.default_rng(2)
    recent = rng.beta(8, 2, size=100).tolist()  # shifted toward 1
    state = det.check(recent)
    assert state == DriftState.ALERT


def test_no_check_before_enough_data():
    ref = _ref()
    det = DriftDetector(ref, DriftConfig(window=100, check_every=1, ks_threshold=0.15))
    # only 20 samples, below window//2=50 threshold
    recent = [0.9] * 20
    state = det.check(recent)
    assert state == DriftState.OK  # not enough data, stays OK


def test_record_and_maybe_check_fires_on_schedule():
    ref = _ref()
    cfg = DriftConfig(window=50, check_every=10, ks_threshold=0.05)
    det = DriftDetector(ref, cfg)
    rng = np.random.default_rng(3)
    shifted = rng.beta(8, 2, size=50).tolist()

    # feed 9 samples, no check yet
    for p in shifted[:9]:
        state = det.record_and_maybe_check(p, shifted)
    assert det.state == DriftState.OK  # hasn't checked yet

    # 10th sample triggers the check
    state = det.record_and_maybe_check(shifted[9], shifted)
    assert state == DriftState.ALERT


# integration test that the store saves and retrieves confidences

def test_store_confidence_history():
    from edge.store import Store
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        store = Store(f.name)
        for i, p in enumerate([0.1, 0.5, 0.9]):
            store.append_confidence(float(i), p)
        recent = store.recent_confidences(10)
        assert recent == [0.1, 0.5, 0.9]


def test_store_recent_confidences_capped():
    from edge.store import Store
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        store = Store(f.name)
        for i in range(20):
            store.append_confidence(float(i), float(i) / 20)
        recent = store.recent_confidences(5)
        assert len(recent) == 5
        assert recent[-1] == pytest.approx(19 / 20)


# integration test that calibration save/load round-trips reference_confidences

def test_calibration_save_includes_reference():
    from edge.calibration import save
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
        ref = [0.1, 0.2, 0.3]
        save(1.5, f.name, reference_confidences=ref)
        data = json.loads(open(f.name).read())
        assert data["temperature"] == pytest.approx(1.5)
        assert data["reference_confidences"] == pytest.approx(ref)


def test_calibration_save_without_reference():
    from edge.calibration import save
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
        save(1.2, f.name)
        data = json.loads(open(f.name).read())
        assert "reference_confidences" not in data
