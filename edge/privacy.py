"""Privacy filter: nothing raw leaves the device (§3.3).

On escalation, transmit only a cropped ROI or an abstracted embedding — never the
full frame. The boundary logger records exactly what bytes/fields cross so
"zero PII egress" is a measured claim. Lands in M5.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np  # type: ignore


@dataclass
class CrossingRecord:
    field: str
    nbytes: int
    is_pii: bool


@dataclass
class FilteredPayload:
    roi_png: Optional[bytes]      # cropped region of interest, encoded
    embedding: Optional[list]     # abstracted embedding (alternative to ROI)
    crossings: list               # list[CrossingRecord] for the boundary log


class PrivacyFilter:
    def __init__(self, mode: str = "roi"):  # "roi" | "embedding"
        self.mode = mode

    def filter(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> FilteredPayload:  # M5
        """Crop to bbox (ROI) or produce an embedding; log every crossing.

        Must guarantee no full-frame bytes and no PII fields ever enter the payload.
        """
        raise NotImplementedError
