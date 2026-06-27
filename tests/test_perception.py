"""Tests for perception logic that doesn't need a real ONNX session or cv2:
output->logit reduction, calibration wiring, and the uncertainty function.
"""

import numpy as np
import pytest

from edge.perception import (
    OnnxClassifier,
    logit_from_output,
    uncertainty_of,
)


def test_logit_from_single_output():
    assert logit_from_output(np.array([[2.0]])) == pytest.approx(2.0)
    assert logit_from_output(np.array([1.5])) == pytest.approx(1.5)


def test_logit_from_two_class():
    # [good, defect] logits -> defect minus good
    assert logit_from_output(np.array([[1.0, 3.0]])) == pytest.approx(2.0)


def test_logit_rejects_unexpected_shape():
    with pytest.raises(ValueError):
        logit_from_output(np.array([[1.0, 2.0, 3.0]]))


def test_uncertainty_peaks_at_half():
    assert uncertainty_of(0.5) == pytest.approx(1.0)
    assert uncertainty_of(0.0) == pytest.approx(0.0)
    assert uncertainty_of(1.0) == pytest.approx(0.0)
    assert uncertainty_of(0.75) == pytest.approx(0.5)


def test_predict_from_logit_uncalibrated():
    clf = OnnxClassifier("unused.onnx", temperature=1.0)
    out = clf.predict_from_logit(0.0)
    assert out.p == pytest.approx(0.5)
    assert out.uncertainty == pytest.approx(1.0)


def test_predict_from_logit_temperature_softens():
    # Higher temperature pulls a confident logit back toward 0.5.
    hot = OnnxClassifier("unused.onnx", temperature=3.0)
    cold = OnnxClassifier("unused.onnx", temperature=1.0)
    p_hot = hot.predict_from_logit(3.0).p
    p_cold = cold.predict_from_logit(3.0).p
    assert 0.5 < p_hot < p_cold
