"""Frame sources behind one interface so the orchestrator is camera-agnostic.

WebcamSource for live (M4), FileSource for eval replay (M7), MockSource for tests.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional

import numpy as np  # type: ignore


class FrameSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]:
        """Yield BGR frames as numpy arrays until exhausted/stopped."""
        raise NotImplementedError


class WebcamSource(FrameSource):
    """Streams BGR frames from a webcam via OpenCV. Yields one frame per `interval_s`
    (default: sample every 0.5s so each inspected item is a distinct part, not 30fps of the
    same one). Stops after `max_frames` if set, else runs until the camera closes."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.5, max_frames=None):
        self.device_index = device_index
        self.interval_s = interval_s
        self.max_frames = max_frames

    def frames(self) -> Iterator[np.ndarray]:
        import time

        import cv2  # lazy: only needed for a real camera

        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera {self.device_index}")
        n = 0
        try:
            while self.max_frames is None or n < self.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
                n += 1
                if self.interval_s:
                    time.sleep(self.interval_s)
        finally:
            cap.release()


class FileSource(FrameSource):
    """Yields BGR frames decoded from image files on disk (e.g. an MVTec split)."""

    def __init__(self, image_paths: list):
        self.image_paths = list(image_paths)

    def frames(self) -> Iterator[np.ndarray]:
        for path in self.image_paths:
            yield self._read(path)

    @staticmethod
    def _read(path: str) -> np.ndarray:
        # Pillow decode (cv2-free), returned as BGR to match the webcam's channel order.
        from PIL import Image  # lazy

        rgb = np.asarray(Image.open(path).convert("RGB"))
        return rgb[:, :, ::-1].copy()  # RGB -> BGR


class MockSource(FrameSource):
    """Yields a fixed list of pre-built arrays — for tests."""

    def __init__(self, frames: Optional[list] = None):
        self._frames = frames or []

    def frames(self) -> Iterator[np.ndarray]:
        return iter(self._frames)
