"""MVTec LOCO AD loader (logical + structural anomalies).

LOCO is the dataset that motivates *why* an edge inspector needs a reasoning cloud model.
Its test split separates two anomaly kinds:

    <category>/test/good/*.png                 label 0
    <category>/test/logical_anomalies/*.png    label 1, kind "logical"
    <category>/test/structural_anomalies/*.png label 1, kind "structural"

Structural anomalies (scratches, dents, contamination) are LOCAL and a texture model can
catch them. Logical anomalies (wrong count, wrong arrangement, a missing or extra object)
are GLOBAL: every surface looks fine, so a local feature model is near-blind to them. The
hypothesis under test: the router should be uncertain on logical anomalies and escalate them
to Qwen-VL, which can reason about count and arrangement. This loader yields the kind so the
harness can score logical and structural subsets separately.

Yields (image_path, label, kind) where kind in {"good", "logical", "structural"}.
"""

from pathlib import Path
from typing import Iterator, Tuple

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# test subdir name -> (label, kind)
_TEST_DIRS = {
    "good": (0, "good"),
    "logical_anomalies": (1, "logical"),
    "structural_anomalies": (1, "structural"),
}

LOCO_CATEGORIES = [
    "breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors",
]


def load_loco(
    root: str, category: str = "breakfast_box", split: str = "test"
) -> Iterator[Tuple[str, int, str]]:
    """Yield (image_path, label, kind) for one LOCO category.

    split "test" yields good + both anomaly kinds; "train"/"validation" are good-only
    (kind "good"). Raises FileNotFoundError if the category/split is missing.
    """
    split_dir = Path(root) / category / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"LOCO split not found: {split_dir}")

    for subdir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label, kind = _TEST_DIRS.get(subdir.name, (1, "structural"))
        # train/validation only contain "good"; test has all three.
        for image_path in sorted(subdir.iterdir()):
            if image_path.suffix.lower() in _IMAGE_EXTS:
                yield str(image_path), label, kind
