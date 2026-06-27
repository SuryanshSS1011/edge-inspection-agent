"""Edge perception: quantized ONNX classifier -> calibrated defect probability (§3.1).

Outputs a *calibrated* p (the router's thresholds assume p is meaningful) plus an
uncertainty estimate. Calibration scalar comes from calibration.py. Lands in M2.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np  # type: ignore


@dataclass
class Perception:
    p: float            # calibrated defect probability
    uncertainty: float  # e.g. entropy or 1 - |2p - 1|


class OnnxClassifier:
    def __init__(self, model_path: str, temperature: float = 1.0):
        self.model_path = model_path
        self.temperature = temperature  # temperature-scaling scalar from calibration.py
        self._session = None  # lazily create onnxruntime.InferenceSession in M2

    def predict(self, frame: np.ndarray) -> Perception:  # M2
        """Preprocess -> run ONNX -> temperature-scale logits -> calibrated p."""
        raise NotImplementedError

    @staticmethod
    def uncertainty_of(p: float) -> float:
        # 1.0 at p=0.5 (max ambiguity), 0.0 at p in {0,1}.
        return 1.0 - abs(2.0 * p - 1.0)
