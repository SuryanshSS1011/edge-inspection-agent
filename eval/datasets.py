"""MVTec AD loader (build plan §5). Confirm the dataset license is acceptable for
the hackathon submission before relying on it. Lands in M7.

Yields (image_path, label, category) where label is 1 for defective, 0 for good.
"""

from pathlib import Path
from typing import Iterator, Tuple


def load_mvtec(root: str, category: str = "bottle", split: str = "test") -> Iterator[Tuple[str, int, str]]:  # M7
    """Iterate an MVTec category. 'good' subdir -> label 0; any other defect subdir -> 1."""
    raise NotImplementedError
