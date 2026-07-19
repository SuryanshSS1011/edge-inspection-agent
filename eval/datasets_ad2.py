"""MVTec AD 2 loader (the 2024 successor to MVTec AD).

AD 2 is harder and more realistic (challenging lighting, more subtle defects). Its layout
differs from the original AD:

    <root>/mvtec_ad_2/<category>/train/good/*.png          defect-free training
    <root>/mvtec_ad_2/<category>/validation/good/*.png     defect-free validation
    <root>/mvtec_ad_2/<category>/test_public/good/*.png    label 0
    <root>/mvtec_ad_2/<category>/test_public/bad/*.png      label 1
    <root>/mvtec_ad_2/<category>/test_private*             UNLABELED benchmark, skipped

Only test_public carries public labels (good/bad subdirs), so that is the eval split. The
test_private and test_private_mixed splits are the held-out benchmark with no released
ground truth and must NOT be used for scored evaluation.

Yields (image_path, label, category).
"""

from pathlib import Path
from typing import Iterator, Tuple

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

AD2_CATEGORIES = [
    "can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts",
]

# The extracted archive nests everything under a top-level mvtec_ad_2/ directory.
_INNER = "mvtec_ad_2"


def _category_dir(root: str, category: str) -> Path:
    """Resolve <category> whether root points at the archive parent or the inner dir."""
    direct = Path(root) / category
    if direct.is_dir():
        return direct
    return Path(root) / _INNER / category


def load_ad2(
    root: str, category: str = "can", split: str = "test_public"
) -> Iterator[Tuple[str, int, str]]:
    """Yield (image_path, label, category) for one AD 2 category.

    split defaults to test_public (the only labeled test split). good -> 0, bad -> 1.
    train/validation contain only good (label 0). Raises FileNotFoundError if missing,
    ValueError if pointed at an unlabeled private split.
    """
    if split.startswith("test_private"):
        raise ValueError(
            f"{split!r} has no released labels; use 'test_public' for scored eval"
        )
    split_dir = _category_dir(root, category) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"AD 2 split not found: {split_dir}")

    for subdir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        if subdir.name == "ground_truth":
            continue  # masks, not samples
        label = 0 if subdir.name == "good" else 1
        for image_path in sorted(subdir.iterdir()):
            if image_path.suffix.lower() in _IMAGE_EXTS:
                yield str(image_path), label, category
