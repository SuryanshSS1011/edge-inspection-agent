"""Privacy filter: nothing raw leaves the device (§3.3).

On escalation, transmit only a cropped ROI or an abstracted embedding — never the full
frame. Every byte/field that crosses the device boundary is recorded as a CrossingRecord
so "zero PII egress" is a *measured* claim (see docs/privacy_model.md), not an assertion.

The cropping and measurement are pure numpy; PNG encoding imports cv2 lazily so the
boundary logic is testable without it.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np  # type: ignore

# Context keys that are allowed to leave the device with an escalation. Anything else is
# dropped and never crosses the boundary. Keep this allowlist deliberately small.
ALLOWED_CONTEXT_KEYS = {"category"}

# Context keys that are explicitly PII / identifying — if one ever appears it is dropped
# AND recorded as a blocked crossing so the audit shows the filter caught it.
PII_CONTEXT_KEYS = {"operator", "serial", "lot_id", "timestamp", "location", "frame"}


@dataclass
class CrossingRecord:
    field: str
    nbytes: int
    is_pii: bool
    blocked: bool = False  # True if the filter refused to let this field cross


@dataclass
class FilteredPayload:
    roi_png: Optional[bytes]      # cropped region of interest, encoded (roi mode)
    embedding: Optional[list]     # abstracted embedding (embedding mode)
    crossings: List[CrossingRecord] = field(default_factory=list)
    context: dict = field(default_factory=dict)  # scrubbed, allowlisted context

    @property
    def pii_bytes(self) -> int:
        """Bytes of PII that actually left the device (blocked crossings don't count)."""
        return sum(c.nbytes for c in self.crossings if c.is_pii and not c.blocked)

    @property
    def total_bytes(self) -> int:
        return sum(c.nbytes for c in self.crossings if not c.blocked)


class PrivacyViolation(RuntimeError):
    """Raised when the requested escalation would expose the full frame / raw data."""


class PrivacyFilter:
    def __init__(self, mode: str = "roi", embedder=None):
        if mode not in ("roi", "embedding"):
            raise ValueError("mode must be 'roi' or 'embedding'")
        self.mode = mode
        self._embedder = embedder  # callable(roi_array) -> list[float], for embedding mode

    def filter(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        context: Optional[dict] = None,
    ) -> FilteredPayload:
        """Crop to bbox (ROI) or produce an embedding; scrub context; log every crossing.

        Guarantees no full-frame bytes and no PII fields enter the payload. Raises
        PrivacyViolation if the ROI is not strictly smaller than the frame in ROI mode.
        Skin-tone regions within the ROI are blurred before encoding to prevent accidental
        PII egress when a human hand or face crosses into the inspection zone.
        """
        roi = self._crop(frame, bbox)
        roi = self._anonymize_skin(roi)
        scrubbed, ctx_crossings = self._scrub_context(context or {})

        if self.mode == "roi":
            self._assert_not_full_frame(roi, frame)
            roi_png = self._encode_png(roi)
            crossings = [CrossingRecord(field="roi_png", nbytes=len(roi_png), is_pii=False)]
            payload = FilteredPayload(roi_png=roi_png, embedding=None, context=scrubbed)
        else:  # embedding mode — no pixels leave at all
            embedding = self._embed(roi)
            nbytes = len(embedding) * 8  # float64 wire size
            crossings = [CrossingRecord(field="embedding", nbytes=nbytes, is_pii=False)]
            payload = FilteredPayload(roi_png=None, embedding=embedding, context=scrubbed)

        payload.crossings = crossings + ctx_crossings
        return payload

    # --- internals -----------------------------------------------------------

    @staticmethod
    def _crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"empty ROI for bbox {bbox} on frame {frame.shape}")
        return frame[y0:y1, x0:x1]

    @staticmethod
    def _assert_not_full_frame(roi: np.ndarray, frame: np.ndarray) -> None:
        if roi.shape[:2] == frame.shape[:2]:
            raise PrivacyViolation(
                "ROI equals the full frame — refusing to escalate raw frame bytes. "
                "Provide a tighter bbox (a detector should localize the part)."
            )

    @staticmethod
    def _anonymize_skin(roi: np.ndarray) -> np.ndarray:
        """Blur skin-tone regions so hands/faces entering the inspection zone never leave
        the device as recoverable biometric data. Uses an HSV skin-tone mask (robust to
        lighting variation) and a box-blur replacement — pure numpy, no cv2 dependency.
        """
        if roi.ndim != 3 or roi.shape[2] != 3:
            return roi
        roi_f = roi.astype(np.float32) / 255.0
        b, g, r = roi_f[:, :, 0], roi_f[:, :, 1], roi_f[:, :, 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        delta = cmax - cmin
        v = cmax
        s = np.where(cmax > 0, np.divide(delta, cmax, where=cmax > 0, out=np.zeros_like(delta)), 0.0)
        h = np.zeros_like(v)
        dr = (cmax == r) & (delta > 0)
        dg = (cmax == g) & (delta > 0)
        db = (cmax == b) & (delta > 0)
        h[dr] = 60.0 * (((g[dr] - b[dr]) / delta[dr]) % 6)
        h[dg] = 60.0 * (((b[dg] - r[dg]) / delta[dg]) + 2)
        h[db] = 60.0 * (((r[db] - g[db]) / delta[db]) + 4)
        skin = (
            ((h <= 25) | (h >= 335)) &
            (s >= 0.1) & (s <= 0.9) &
            (v >= 0.2) & (v <= 0.95)
        )
        if not np.any(skin):
            return roi
        k = max(3, min(roi.shape[0], roi.shape[1]) // 8)
        k = k if k % 2 == 1 else k + 1
        pad = k // 2
        padded = np.pad(roi.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, (k, k, 3))[:, :, 0]
        blurred = windows.reshape(*roi.shape[:2], k * k, 3).mean(axis=2).astype(roi.dtype)
        out = roi.copy()
        out[skin] = blurred[skin]
        return out

    @staticmethod
    def _encode_png(roi: np.ndarray) -> bytes:
        import cv2  # lazy

        ok, buf = cv2.imencode(".png", roi)
        if not ok:
            raise RuntimeError("failed to PNG-encode ROI")
        return buf.tobytes()

    def _embed(self, roi: np.ndarray) -> list:
        if self._embedder is not None:
            return list(self._embedder(roi))
        # Default abstracted embedding: per-channel mean/std — carries no recoverable image.
        chans = roi.reshape(-1, roi.shape[2]) if roi.ndim == 3 else roi.reshape(-1, 1)
        means = chans.mean(axis=0)
        stds = chans.std(axis=0)
        return [float(v) for v in np.concatenate([means, stds])]

    @staticmethod
    def _scrub_context(context: dict) -> Tuple[dict, List[CrossingRecord]]:
        """Keep only allowlisted, non-PII keys. Record any PII key as a blocked crossing
        so the audit log proves the filter caught it."""
        scrubbed = {}
        crossings: List[CrossingRecord] = []
        for key, value in context.items():
            nbytes = len(str(value).encode("utf-8"))
            if key in PII_CONTEXT_KEYS:
                crossings.append(CrossingRecord(field=f"context.{key}", nbytes=nbytes,
                                                is_pii=True, blocked=True))
                continue
            if key in ALLOWED_CONTEXT_KEYS:
                scrubbed[key] = value
                crossings.append(CrossingRecord(field=f"context.{key}", nbytes=nbytes,
                                                is_pii=False))
                continue
            # Unlisted keys are dropped (default-deny) AND logged as blocked, so the audit
            # trail proves the filter caught an unanticipated identifier — not just known
            # PII. is_pii=True because an unknown field could carry PII; blocked=True so it
            # never counts as egress.
            crossings.append(CrossingRecord(field=f"context.{key}", nbytes=nbytes,
                                            is_pii=True, blocked=True))
        return scrubbed, crossings
