"""Lightweight image features for the modest local classifier (PIL + numpy only).

These are deliberately simple. The router needs a *calibrated, uncertain-near-the-
boundary* local model, not a strong one (the cloud catches the hard cases). The features
capture the cues that separate a clean bottle from broken-glass / contamination defects:
color statistics, edge density, and grid-wise intensity variance (spatial irregularity).

The same fixed-length vector is produced at train and inference time, so the ONNX model
trained on these takes exactly this vector as input.
"""

from typing import List

import numpy as np  # type: ignore

GRID = 4              # 4x4 cells for spatial variance
INPUT_SIZE = (128, 128)
FEATURE_DIM = 3 + 3 + 1 + GRID * GRID  # means + stds + edge density + per-cell variance


def _load_gray_and_rgb(path: str):
    from PIL import Image  # lazy

    img = Image.open(path).convert("RGB").resize(INPUT_SIZE)
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    return rgb, gray


def features_from_array(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    means = rgb.reshape(-1, 3).mean(axis=0)            # 3
    stds = rgb.reshape(-1, 3).std(axis=0)              # 3

    # Edge density: mean gradient magnitude (defects add edges).
    gy, gx = np.gradient(gray)
    edge_density = np.array([np.sqrt(gx ** 2 + gy ** 2).mean()])  # 1

    # Grid-wise intensity variance: contamination/breakage = local irregularity.
    h, w = gray.shape
    ch, cw = h // GRID, w // GRID
    cell_var = []
    for i in range(GRID):
        for j in range(GRID):
            cell = gray[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            cell_var.append(cell.var())
    cell_var = np.array(cell_var)                      # GRID*GRID

    return np.concatenate([means, stds, edge_density, cell_var]).astype(np.float32)


def features_from_bgr(frame: np.ndarray) -> np.ndarray:
    """Same 23-d feature vector as `extract`, but from an in-memory BGR uint8 frame on the
    live camera path instead of a file. Resizes to INPUT_SIZE and reuses the array core so
    train-time and live features stay identical."""
    import cv2  # lazy

    resized = cv2.resize(frame, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = rgb.mean(axis=2)
    return features_from_array(rgb, gray)


def extract(path: str) -> np.ndarray:
    rgb, gray = _load_gray_and_rgb(path)
    return features_from_array(rgb, gray)


def extract_many(paths: List[str]) -> np.ndarray:
    return np.stack([extract(p) for p in paths]) if paths else np.empty((0, FEATURE_DIM), np.float32)
