"""MVTec AD loader (build plan §5). Confirm the dataset license is acceptable for
the hackathon submission before relying on it.

Directory layout (MVTec AD convention):
    root/<category>/<split>/<subdir>/*.png
where <split> is "train" (only a "good" subdir) or "test" ("good" + defect subdirs).
A subdir named "good" => label 0; any other subdir => label 1 (defective).

Yields (image_path, label, category).
"""

from pathlib import Path
from typing import Iterator, Tuple

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_mvtec(
    root: str, category: str = "bottle", split: str = "test"
) -> Iterator[Tuple[str, int, str]]:
    split_dir = Path(root) / category / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"MVTec split not found: {split_dir}")

    for subdir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = 0 if subdir.name == "good" else 1
        for image_path in sorted(subdir.iterdir()):
            if image_path.suffix.lower() in _IMAGE_EXTS:
                yield str(image_path), label, category
