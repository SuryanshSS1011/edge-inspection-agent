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


class WebcamSource(FrameSource):  # M4
    def __init__(self, device_index: int = 0):
        self.device_index = device_index

    def frames(self) -> Iterator[np.ndarray]:
        raise NotImplementedError


class FileSource(FrameSource):  # M7 — replay an MVTec category for eval
    def __init__(self, image_paths: list):
        self.image_paths = image_paths

    def frames(self) -> Iterator[np.ndarray]:
        raise NotImplementedError


class MockSource(FrameSource):
    """Yields a fixed list of pre-built arrays — for tests."""

    def __init__(self, frames: Optional[list] = None):
        self._frames = frames or []

    def frames(self) -> Iterator[np.ndarray]:
        return iter(self._frames)
