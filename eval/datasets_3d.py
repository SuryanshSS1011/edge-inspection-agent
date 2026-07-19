"""MVTec 3D-AD loader (organized point clouds + RGB).

Layout:
    <root>/<category>/test/good/{rgb,xyz,gt}/NNN.{png,tiff,png}   label 0
    <root>/<category>/test/<defect>/{rgb,xyz,gt}/...              label 1
    <root>/<category>/train/good/{rgb,xyz}/...                   defect-free training

Each sample is a triplet keyed by frame id NNN:
    rgb/NNN.png   color image (HxWx3 uint8)
    xyz/NNN.tiff  ORGANIZED point cloud (HxWx3 float32): each pixel holds its X,Y,Z in the
                  sensor frame; (0,0,0) marks a missing/background point.
    gt/NNN.png    anomaly mask (good samples have none)

The point cloud is the signal the 3D perception path consumes. This loader yields the xyz
path (the depth/3D modality) plus the rgb path for reference.

Yields (xyz_path, rgb_path, label, category).
"""

from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np  # type: ignore

MVTEC3D_CATEGORIES = [
    "bagel", "cable_gland", "carrot", "cookie", "dowel",
    "foam", "peach", "potato", "rope", "tire",
]


def load_mvtec3d(
    root: str, category: str = "bagel", split: str = "test"
) -> Iterator[Tuple[str, Optional[str], int, str]]:
    """Yield (xyz_path, rgb_path, label, category) for one 3D-AD category.

    split "test" yields good + all defect types; "train"/"validation" are good-only.
    good -> 0, any other subdir -> 1. rgb_path is None if the rgb sibling is absent.
    """
    split_dir = Path(root) / category / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"MVTec 3D-AD split not found: {split_dir}")

    for subdir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = 0 if subdir.name == "good" else 1
        xyz_dir = subdir / "xyz"
        rgb_dir = subdir / "rgb"
        if not xyz_dir.is_dir():
            continue
        for xyz_path in sorted(xyz_dir.glob("*.tiff")):
            rgb_path = rgb_dir / f"{xyz_path.stem}.png"
            yield (
                str(xyz_path),
                str(rgb_path) if rgb_path.is_file() else None,
                label,
                category,
            )


def load_organized_pointcloud(xyz_path: str) -> np.ndarray:
    """Read an organized point cloud TIFF as an (H, W, 3) float32 array of XYZ per pixel.

    MVTec 3D-AD stores the cloud as a 3-channel float32 TIFF; PIL cannot decode that layout,
    so we use OpenCV (IMREAD_UNCHANGED preserves the float channels). cv2 loads channels as
    BGR-order, but here the three channels are X,Y,Z (not colour), so no swap is applied.
    """
    import cv2  # lazy

    arr = cv2.imread(xyz_path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"could not read organized cloud: {xyz_path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 organized cloud, got {arr.shape} in {xyz_path}")
    return arr


def to_point_list(organized: np.ndarray, drop_zero: bool = True) -> np.ndarray:
    """Flatten an organized (H, W, 3) cloud to an (N, 3) point list.

    Points at exactly (0, 0, 0) are background/missing in the MVTec 3D convention and are
    dropped by default so downstream models see only real surface points.
    """
    pts = organized.reshape(-1, 3)
    if drop_zero:
        mask = np.any(pts != 0.0, axis=1)
        pts = pts[mask]
    return pts
