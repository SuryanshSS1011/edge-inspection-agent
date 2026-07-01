"""MobileNetV2 as a frozen ONNX feature extractor (no torch install).

The reviewer's suggested upgrade: run a small pretrained backbone as an ONNX feature
extractor and put the same LogisticRegression + temperature-scaling head on top of its
embeddings. This raises the local model's floor — especially on texture categories like
grid where hand-crafted color/edge features are weakest — with ZERO change to the router,
privacy filter, or outbox. It's a drop-in behind the same OnnxClassifier interface.

The 1000-d pre-softmax logits are used as a general-purpose image embedding. Preprocessing
is standard ImageNet (resize 224, center, per-channel normalize). Pillow-based (no cv2).
"""

import io
from pathlib import Path

import numpy as np  # type: ignore

_MODEL = "models/mobilenetv2.onnx"
_SIZE = (224, 224)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FEATURE_DIM = 1000

_session = None  # lazily created, reused across calls


def _sess():
    global _session
    if _session is None:
        import onnxruntime as ort  # lazy

        if not Path(_MODEL).is_file():
            raise FileNotFoundError(
                f"{_MODEL} not found — download MobileNetV2 ONNX (see docs) before using "
                "the mobilenet backbone."
            )
        _session = ort.InferenceSession(_MODEL, providers=["CPUExecutionProvider"])
    return _session


def _preprocess(path: str) -> np.ndarray:
    from PIL import Image  # lazy

    img = Image.open(path).convert("RGB").resize(_SIZE)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
    chw = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(chw, 0).astype(np.float32)


def extract(path: str) -> np.ndarray:
    """Return the 1000-d MobileNetV2 embedding for one image."""
    sess = _sess()
    out = sess.run(None, {sess.get_inputs()[0].name: _preprocess(path)})[0]
    return np.asarray(out, dtype=np.float32).reshape(-1)


def extract_many(paths) -> np.ndarray:
    paths = list(paths)
    if not paths:
        return np.empty((0, FEATURE_DIM), np.float32)
    return np.stack([extract(p) for p in paths])
