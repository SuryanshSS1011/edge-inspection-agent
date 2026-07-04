"""Temperature scaling so the classifier's p is trustworthy (§5).

The router's thresholds assume p is a *calibrated* probability. A raw classifier is
usually over/under-confident, so we fit a single scalar temperature T on a held-out
validation split that minimizes binary cross-entropy of sigmoid(logit / T) vs labels,
then divide every inference logit by T.

Pure numpy/scipy — no model/ONNX dependency, so this is fully testable offline.
Persist T to models/temperature.json.
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np  # type: ignore

_EPS = 1e-7


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _bce(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs, _EPS, 1.0 - _EPS)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))


def fit_temperature(logits, labels, bounds: Tuple[float, float] = (0.05, 10.0)) -> float:
    """Return the scalar temperature minimizing validation BCE.

    logits: raw pre-sigmoid defect scores (positive => defect).
    labels: 0/1 ground truth (1 => defect).
    The objective is convex in 1/T; a bounded 1-D search is robust and dependency-light.
    """
    from scipy.optimize import minimize_scalar  # lazy import keeps module load cheap

    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have the same shape")
    if logits.size == 0:
        raise ValueError("cannot calibrate on an empty validation set")

    def objective(t: float) -> float:
        return _bce(_sigmoid(logits / t), labels)

    res = minimize_scalar(objective, bounds=bounds, method="bounded")
    return float(res.x)


def apply_temperature(logit, temperature: float):
    """Map a defect logit (scalar or array) through temperature scaling to a probability."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return _sigmoid(np.asarray(logit, dtype=float) / temperature)


def _bin_mask(probs: np.ndarray, lo: float, hi: float, first: bool) -> np.ndarray:
    """Bins partition [0, 1] as [0, e1], (e1, e2], ..., (e_{n-1}, 1]. The first bin is
    closed on the left so a probability of exactly 0.0 is counted."""
    upper = probs <= hi
    lower = probs >= lo if first else probs > lo
    return lower & upper


def expected_calibration_error(probs, labels, n_bins: int = 10) -> float:
    """ECE: weighted gap between confidence and accuracy across probability bins.

    The headline number for the M2 exit check — lower is better-calibrated.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = _bin_mask(probs, lo, hi, first=(i == 0))
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        avg_conf = float(np.mean(probs[in_bin]))
        avg_acc = float(np.mean(labels[in_bin]))
        ece += (count / n) * abs(avg_conf - avg_acc)
    return ece


def reliability_curve(probs, labels, n_bins: int = 10):
    """Return (bin_confidence, bin_accuracy, bin_count) arrays for plotting/inspection."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf, acc, cnt = [], [], []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_bin = _bin_mask(probs, lo, hi, first=(i == 0))
        count = int(np.sum(in_bin))
        conf.append(float(np.mean(probs[in_bin])) if count else 0.0)
        acc.append(float(np.mean(labels[in_bin])) if count else 0.0)
        cnt.append(count)
    return np.array(conf), np.array(acc), np.array(cnt)


def save(temperature: float, path: str = "models/temperature.json",
         reference_confidences=None) -> None:
    data = {"temperature": temperature}
    if reference_confidences is not None:
        data["reference_confidences"] = [float(p) for p in reference_confidences]
    Path(path).write_text(json.dumps(data))


def load(path: str = "models/temperature.json") -> float:
    return float(json.loads(Path(path).read_text())["temperature"])
