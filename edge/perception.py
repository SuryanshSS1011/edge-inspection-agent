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
    p: float  # calibrated defect probability
    uncertainty: float  # 1 - |2p - 1|: peaks at p=0.5, zero at the extremes


def preprocess(frame: np.ndarray, size: Tuple[int, int] = _INPUT_SIZE) -> np.ndarray:
    """BGR HxWx3 uint8 frame -> normalized NCHW float32 tensor for the ONNX model."""
    import cv2  # lazy

    img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    chw = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(chw, 0).astype(np.float32)  # add batch dim


def logit_from_output(output: np.ndarray) -> float:
    """Reduce a model output to a single defect logit (temperature scaling is applied to it).

    Supports:
      - shape (1, 1) or (1,): a raw defect logit, used directly.
      - shape (1, 2): two values. If they look like a probability pair [P(good), P(defect)]
        (non-negative and summing to ~1), recover the logit as log(P_def / P_good);
        otherwise treat them as raw class logits [good, defect] -> defect - good.
    """
    arr = np.asarray(output, dtype=float).reshape(-1)
    if arr.size == 1:
        return float(arr[0])
    if arr.size == 2:
        is_prob_pair = (arr >= 0).all() and abs(arr.sum() - 1.0) < 1e-3
        if is_prob_pair:
            eps = 1e-7
            p_def = min(max(arr[1], eps), 1 - eps)
            return float(np.log(p_def / (1.0 - p_def)))
        return float(arr[1] - arr[0])
    raise ValueError(f"unexpected model output shape with {arr.size} values")


def uncertainty_of(p: float) -> float:
    """1.0 at p=0.5 (max ambiguity), 0.0 at p in {0, 1}."""
    return 1.0 - abs(2.0 * p - 1.0)


def _pick_score_output(outputs):
    """Choose the scores output among possibly several ONNX outputs (one row assumed).

    sklearn-onnx emits [label, probabilities]; a CNN may emit a single logit tensor.
    Prefer a 2-value probabilities/logits output, then a single-value logit, else the
    first output."""
    for out in outputs:
        if np.asarray(out).size == 2:
            return out
    for out in outputs:
        if np.asarray(out).size == 1:
            return out
    return outputs[0]


class OnnxClassifier:
    def __init__(self, model_path: str, temperature: float = 1.0):
        self.model_path = model_path
        self.temperature = temperature  # from calibration.py; 1.0 = uncalibrated
        self._session = None
        self._input_dim: Optional[int] = None  # cached feature width the model expects

    def _ensure_session(self):
        if self._session is None:
            import onnxruntime as ort  # lazy

            self._session = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            shape = self._session.get_inputs()[0].shape
            # Last dim is the feature width for the LR/MLP heads; None for a 4-D CNN.
            last = shape[-1]
            self._input_dim = last if isinstance(last, int) else None
        return self._session

    def _features_for(self, frame: np.ndarray) -> np.ndarray:
        """Build the input the loaded model expects from a live BGR frame.

        The head trained in eval determines the input width, so we dispatch on it:
          * 23  -> hand-crafted color/edge/grid features (legacy classifier.onnx)
          * 384 -> DINOv2 ViT-S/14 CLS embedding (the default core backbone)
          * 1000 -> MobileNetV2 embedding (grid_mobilenet.onnx backbone head)
          * else (a 4-D CNN taking a raw image) -> ImageNet-normalized NCHW tensor
        This keeps the live path in lockstep with whatever config.yaml points `model` at,
        using the exact same feature code as training."""
        self._ensure_session()
        if self._input_dim == 23:
            from eval.features import features_from_bgr

            return features_from_bgr(frame).reshape(1, -1)
        if self._input_dim == 384:
            from eval.dinov2_features import extract_from_bgr

            return extract_from_bgr(frame).reshape(1, -1)
        if self._input_dim == 1000:
            from eval.mobilenet_features import extract_from_bgr

            return extract_from_bgr(frame).reshape(1, -1)
        return preprocess(frame)  # raw-image CNN

    def predict(self, frame: np.ndarray) -> Perception:
        return self.predict_from_features(self._features_for(frame))

    def predict_from_features(self, tensor: np.ndarray) -> Perception:
        """Run the ONNX model on a ready input tensor and calibrate. Used directly when the
        model takes a feature vector rather than a raw image (e.g. the LR baseline)."""
        session = self._ensure_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: tensor})
        return self.predict_from_logit(logit_from_output(_pick_score_output(outputs)))

    def predict_from_logit(self, logit: float) -> Perception:
        """Calibrate a raw defect logit -> Perception. Separated so it's unit-testable
        without a real ONNX session."""
        p = float(apply_temperature(logit, self.temperature))
        return Perception(p=p, uncertainty=uncertainty_of(p))
