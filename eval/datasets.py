"""MVTec AD loader (build plan §5). Confirm the dataset license is acceptable for
the hackathon submission before relying on it (CC BY-NC-SA 4.0, non-commercial).

Directory layout (MVTec AD convention):
    root/<category>/train/good/*.png            # defect-free training images
    root/<category>/test/good/*.png             # defect-free test images   (label 0)
    root/<category>/test/<defect>/*.png         # defective test images     (label 1)
    root/<category>/ground_truth/<defect>/*_mask.png   # pixel-precise anomaly masks

A subdir named "good" => label 0; any other subdir => label 1 (defective).
`ground_truth` holds segmentation masks, NOT samples, so it is not a valid split here.

Yields (image_path, label, category).
"""

from pathlib import Path
from typing import Iterator, Optional, Tuple

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
_VALID_SPLITS = {"train", "test"}


def load_mvtec(
    root: str, category: str = "bottle", split: str = "test"
) -> Iterator[Tuple[str, int, str]]:
    if split not in _VALID_SPLITS:
        # ground_truth holds masks, not labeled samples; reading it would mislabel data.
        raise ValueError(f"split must be one of {sorted(_VALID_SPLITS)}, got {split!r}")

    split_dir = Path(root) / category / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"MVTec split not found: {split_dir}")

    for subdir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = 0 if subdir.name == "good" else 1
        for image_path in sorted(subdir.iterdir()):
            if image_path.suffix.lower() in _IMAGE_EXTS:
                yield str(image_path), label, category


def ground_truth_mask(image_path: str) -> Optional[str]:
    """Return the pixel-precise anomaly mask for a defective test image, or None.

    MVTec names masks `<frame>_mask.png` under ground_truth/<defect>/. Good images and
    images without a mask return None. Useful later to derive a tight ROI for the privacy
    filter instead of the fixed centered crop.
    """
    p = Path(image_path)
    defect = p.parent.name
    if defect == "good":
        return None
    category_dir = p.parent.parent.parent          # .../<category>/test/<defect>/x.png
    mask = category_dir / "ground_truth" / defect / f"{p.stem}_mask{p.suffix}"
    return str(mask) if mask.is_file() else None
