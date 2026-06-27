"""Temperature scaling so the classifier's p is trustworthy (§5).

Fit a single scalar T on a held-out validation split by minimizing NLL of
softmax(logits / T) vs labels. Persist T to models/temperature.json. M2.
"""

import json
from pathlib import Path

import numpy as np  # type: ignore


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:  # M2
    """Return the scalar temperature minimizing validation NLL.

    Use scipy.optimize.minimize_scalar over T in (0, 10]; convex in 1/T.
    """
    raise NotImplementedError


def apply_temperature(logit: float, temperature: float) -> float:
    """Map a defect logit through temperature scaling to a calibrated probability."""
    return 1.0 / (1.0 + np.exp(-logit / temperature))


def save(temperature: float, path: str = "models/temperature.json") -> None:
    Path(path).write_text(json.dumps({"temperature": temperature}))


def load(path: str = "models/temperature.json") -> float:
    return float(json.loads(Path(path).read_text())["temperature"])
