"""PointNet as a frozen ONNX feature extractor for the 3D (point-cloud) modality.

The 3D analogue of dinov2_features / mobilenet_features: an organized point cloud -> a fixed
global descriptor -> the SAME LogisticRegression + temperature head. The router, privacy
filter, and outbox never change, so 3D-AD is a drop-in new modality behind the same
interface. This is what makes the cost-routing thesis modality-agnostic in practice, not just
in principle: the router bands a calibrated p exactly as it does for 2D.

Preprocessing doubles as the privacy step (see privacy note in `prepare`):
  1. drop background/zero points,
  2. sample/pad to a fixed N,
  3. CENTER to local coordinates (subtract the centroid) and scale to unit radius, which
     strips the absolute sensor-frame position/pose so only local surface shape leaves.
"""

from pathlib import Path

import numpy as np  # type: ignore

_MODEL = "models/pointnet.onnx"
_N = 2048  # points per cloud; must match export_pointnet --points

_session = None
_feature_dim = None
FEATURE_DIM = 256  # advisory default; the real dim is read from the ONNX


def _sess():
    global _session, _feature_dim
    if _session is None:
        import onnxruntime as ort  # lazy

        if not Path(_MODEL).is_file():
            raise FileNotFoundError(
                f"{_MODEL} not found. Export it first: `python -m eval.export_pointnet` "
                "(needs torch, run once on ROAR)."
            )
        avail = ort.get_available_providers()
        prov = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        _session = ort.InferenceSession(_MODEL, providers=prov or ["CPUExecutionProvider"])
        last = _session.get_outputs()[0].shape[-1]
        _feature_dim = last if isinstance(last, int) else None
    return _session


def _resolve_dim() -> int:
    global FEATURE_DIM
    _sess()
    if _feature_dim:
        FEATURE_DIM = _feature_dim
    return FEATURE_DIM


def prepare(points: np.ndarray, n: int = _N, rng=None) -> np.ndarray:
    """Sample/pad to n points, then center + unit-scale to LOCAL coordinates.

    Centering removes the absolute sensor-frame position (the privacy-relevant part: where
    the object physically is), leaving only its local surface geometry. Returns [n, 3]."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    pts = pts[np.any(pts != 0.0, axis=1)]  # drop background/missing
    if len(pts) == 0:
        pts = np.zeros((1, 3), np.float32)
    rng = rng or np.random.default_rng(0)
    if len(pts) >= n:
        idx = rng.choice(len(pts), n, replace=False)
    else:
        idx = rng.choice(len(pts), n, replace=True)
    pts = pts[idx]
    pts = pts - pts.mean(axis=0, keepdims=True)          # center -> strips world position
    scale = np.linalg.norm(pts, axis=1).max() + 1e-9
    return (pts / scale).astype(np.float32)               # unit radius -> scale-invariant


def _embed(pts_n3: np.ndarray) -> np.ndarray:
    sess = _sess()
    out = sess.run(None, {sess.get_inputs()[0].name: pts_n3[None, ...]})[0]
    return np.asarray(out, dtype=np.float32).reshape(-1)


def extract_from_cloud(organized_or_points: np.ndarray) -> np.ndarray:
    """Global PointNet descriptor for one cloud (organized HxWx3 or a flat Nx3 point list)."""
    return _embed(prepare(organized_or_points))


def extract(xyz_path: str) -> np.ndarray:
    """Descriptor for one MVTec 3D-AD xyz TIFF."""
    from eval.datasets_3d import load_organized_pointcloud

    return extract_from_cloud(load_organized_pointcloud(xyz_path))


def extract_many(xyz_paths) -> np.ndarray:
    xyz_paths = list(xyz_paths)
    if not xyz_paths:
        return np.empty((0, _resolve_dim()), np.float32)
    return np.stack([extract(p) for p in xyz_paths])
