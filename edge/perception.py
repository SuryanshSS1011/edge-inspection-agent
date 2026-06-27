"""Edge perception: quantized ONNX classifier -> calibrated defect probability (§3.1).

Runs the common case locally in milliseconds (the reflex layer) and outputs a
*calibrated* defect probability p plus an uncertainty estimate. The temperature scalar
comes from calibration.py; without it (temperature defaults to 1.0) p is the raw model
probability.

onnxruntime and cv2 are imported lazily so this module loads (and the pure logic is
testable) even where those heavy deps aren't installed.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np  # type: ignore

from edge.calibration import apply_temperature

# Standard ImageNet-style preprocessing; override per model if needed.
_INPUT_SIZE = (224, 224)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Perception:
    p: float            # calibrated defect probability
    uncertainty: float  # 1 - |2p - 1|: peaks at p=0.5, zero at the extremes


def preprocess(frame: np.ndarray, size: Tuple[int, int] = _INPUT_SIZE) -> np.ndarray:
    """BGR HxWx3 uint8 frame -> normalized NCHW float32 tensor for the ONNX model."""
    import cv2  # lazy

    img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    chw = np.transpose(img, (2, 0, 1))            # HWC -> CHW
    return np.expand_dims(chw, 0).astype(np.float32)  # add batch dim


def logit_from_output(output: np.ndarray) -> float:
    """Reduce a model output to a single defect logit.

    Supports:
      - shape (1, 1) or (1,): a raw defect logit, used directly.
      - shape (1, 2): two-class logits [good, defect] -> logit(defect) - logit(good).
    """
    arr = np.asarray(output, dtype=float).reshape(-1)
    if arr.size == 1:
        return float(arr[0])
    if arr.size == 2:
        return float(arr[1] - arr[0])
    raise ValueError(f"unexpected model output shape with {arr.size} values")


def uncertainty_of(p: float) -> float:
    """1.0 at p=0.5 (max ambiguity), 0.0 at p in {0, 1}."""
    return 1.0 - abs(2.0 * p - 1.0)


class OnnxClassifier:
    def __init__(self, model_path: str, temperature: float = 1.0):
        self.model_path = model_path
        self.temperature = temperature  # from calibration.py; 1.0 = uncalibrated
        self._session = None

    def _ensure_session(self):
        if self._session is None:
            import onnxruntime as ort  # lazy

            self._session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
        return self._session

    def predict(self, frame: np.ndarray) -> Perception:
        session = self._ensure_session()
        tensor = preprocess(frame)
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: tensor})[0]
        return self.predict_from_logit(logit_from_output(output))

    def predict_from_logit(self, logit: float) -> Perception:
        """Calibrate a raw defect logit -> Perception. Separated so it's unit-testable
        without a real ONNX session."""
        p = float(apply_temperature(logit, self.temperature))
        return Perception(p=p, uncertainty=uncertainty_of(p))
