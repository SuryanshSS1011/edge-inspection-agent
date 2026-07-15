"""DINOv2 as a frozen ONNX feature extractor (no torch at inference).

The SOTA frozen backbone for anomaly detection: DINOv2's self-supervised ViT features are
what modern methods (PatchCore et al.) build on. Same drop-in role as MobileNetV2 here, a
stronger frozen embedding under the SAME LogisticRegression + temperature head, so the
router / privacy / outbox never change. This is the third ablation arm alongside
handcrafted and mobilenet: it shows the router absorbs local-model variance even when the
backbone is best-in-class.

The CLS-token embedding is used as a global image descriptor. ViT-S/14 gives 384 dims,
ViT-B/14 gives 768; FEATURE_DIM is read from the ONNX model's output so either works.
Export the ONNX on ROAR with `python -m eval.export_dinov2` (needs torch, one-time).
Preprocessing is standard ImageNet at 224 (a multiple of the /14 patch size). Pillow-based.
"""

import io  # noqa: F401 - kept for parity with mobilenet_features API surface
from pathlib import Path

import numpy as np  # type: ignore

_MODEL = "models/dinov2.onnx"
_SIZE = (224, 224)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = None  # lazily created, reused across calls
_feature_dim = None  # read from the model on first use


def _sess():
    global _session, _feature_dim
    if _session is None:
        import onnxruntime as ort  # lazy

        if not Path(_MODEL).is_file():
            raise FileNotFoundError(
                f"{_MODEL} not found. Export DINOv2 to ONNX first: "
                "`python -m eval.export_dinov2` (needs torch, run once on ROAR)."
            )
        _session = ort.InferenceSession(_MODEL, providers=_providers())
        out_shape = _session.get_outputs()[0].shape
        last = out_shape[-1]
        _feature_dim = last if isinstance(last, int) else None
    return _session


def _providers():
    """Prefer CUDA on the cluster, fall back to CPU. onnxruntime picks the first available."""
    import onnxruntime as ort  # lazy

    avail = ort.get_available_providers()
    order = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return [p for p in order if p in avail] or ["CPUExecutionProvider"]


# FEATURE_DIM is resolved from the model; expose a sensible default (ViT-S/14) for callers
# that read it before a session exists (train_classifier's _backbone). It is corrected to the
# real value once the session loads, and export_dinov2 records the true dim next to the model.
FEATURE_DIM = 384


def _resolve_dim() -> int:
    global FEATURE_DIM
    _sess()
    if _feature_dim:
        FEATURE_DIM = _feature_dim
    return FEATURE_DIM


def _preprocess(path: str) -> np.ndarray:
    from PIL import Image  # lazy

    img = Image.open(path).convert("RGB").resize(_SIZE)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - _MEAN) / _STD
    chw = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(chw, 0).astype(np.float32)


def _preprocess_bgr(frame: np.ndarray) -> np.ndarray:
    """ImageNet preprocessing from an in-memory BGR frame (the live camera path)."""
    import cv2  # lazy

    rgb = cv2.cvtColor(cv2.resize(frame, _SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    arr = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    chw = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(chw, 0).astype(np.float32)


def _embed(tensor: np.ndarray) -> np.ndarray:
    sess = _sess()
    out = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]
    return np.asarray(out, dtype=np.float32).reshape(-1)


def extract(path: str) -> np.ndarray:
    """Return the DINOv2 CLS-token embedding for one image."""
    return _embed(_preprocess(path))


def extract_from_bgr(frame: np.ndarray) -> np.ndarray:
    """Return the DINOv2 CLS-token embedding for an in-memory BGR frame (live cam)."""
    return _embed(_preprocess_bgr(frame))


def extract_many(paths) -> np.ndarray:
    paths = list(paths)
    if not paths:
        return np.empty((0, _resolve_dim()), np.float32)
    return np.stack([extract(p) for p in paths])
