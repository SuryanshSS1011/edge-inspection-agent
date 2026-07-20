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

    rgb = cv2.cvtColor(
        cv2.resize(frame, _SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB
    )
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


# --- optional torch-CUDA fast path -----------------------------------------------------
# onnxruntime-gpu on some clusters can't find cuDNN and silently falls back to CPU, which
# makes full-dataset extraction glacial. When TOLLGATE_DINOV2_TORCH=1, run the same DINOv2
# weights through torch on the GPU in batches instead. Produces the same embedding; used only
# for eval throughput, the ONNX path stays the default (no torch at inference).

_torch_model = None
_torch_device = None


def _torch_backbone():
    global _torch_model, _torch_device, FEATURE_DIM
    if _torch_model is None:
        import os
        import torch

        _torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        os.environ.setdefault("TORCH_HOME", "/scratch/sss6371/torch_hub")
        _torch_model = (
            torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
            .eval()
            .to(_torch_device)
        )
        for p in _torch_model.parameters():
            p.requires_grad_(False)
        FEATURE_DIM = 384
    return _torch_model, _torch_device


def _extract_many_torch(paths, batch_size: int = 64) -> np.ndarray:
    import torch

    model, device = _torch_backbone()
    tensors = [torch.from_numpy(_preprocess(p)[0]) for p in paths]  # each [3,224,224]
    out = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            batch = torch.stack(tensors[i : i + batch_size]).to(device)
            emb = model(batch)  # [B, 384]
            out.append(emb.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def extract_many(paths) -> np.ndarray:
    import os

    paths = list(paths)
    if not paths:
        return np.empty((0, _resolve_dim()), np.float32)
    if os.environ.get("TOLLGATE_DINOV2_TORCH") == "1":
        return _extract_many_torch(paths)
    return np.stack([extract(p) for p in paths])
