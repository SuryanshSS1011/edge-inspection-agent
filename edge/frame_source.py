"""Frame sources behind one interface so the orchestrator is camera-agnostic.

WebcamSource for live capture (laptop, USB, or a phone streaming over the network),
FileSource for eval replay, MockSource for tests.
"""

import logging
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Union

import numpy as np  # type: ignore

_log = logging.getLogger(__name__)


class FrameSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]:
        """Yield BGR frames as numpy arrays until exhausted/stopped."""
        raise NotImplementedError


class CameraUnavailable(RuntimeError):
    """No capture device could be opened (bad index, unplugged, or unreachable URL)."""


class WebcamSource(FrameSource):
    """Streams BGR frames from a camera via OpenCV.

    ``device`` may be:
      * an int index (0 = built-in, 1+ = USB / Continuity / OBS virtual cam), or
      * ``"auto"`` to scan indices 0..3 and pick the first that opens, or
      * a stream URL string for an IP-webcam / MJPEG / RTSP feed from a phone, e.g.
        ``http://192.168.1.42:8080/video`` (Android "IP Webcam") or an ``rtsp://`` URL.

    Yields one frame per ``interval_s`` (default 0.5s) so each inspected item is a
    distinct part rather than 30fps of the same one. A short warm-up discards the first
    few frames, which are often black while the sensor exposes. Opening is retried a few
    times before giving up so slow-to-initialize devices (and phone streams still coming
    up) don't fail the run. Runs until the device closes, or ``max_frames`` if set.
    """

    _AUTO_SCAN_RANGE = 4  # indices 0..3 tried by "auto"

    def __init__(
        self,
        device: Union[int, str] = 0,
        interval_s: float = 0.5,
        max_frames: Optional[int] = None,
        warmup_frames: int = 5,
        open_retries: int = 5,
        retry_delay_s: float = 0.5,
    ):
        self.device = device
        self.interval_s = interval_s
        self.max_frames = max_frames
        self.warmup_frames = warmup_frames
        self.open_retries = open_retries
        self.retry_delay_s = retry_delay_s

    # --- opening -----------------------------------------------------------------

    def _open_one(self, target):
        """Open a single VideoCapture target (int index or URL). Returns an opened cap
        or None. URL streams get the FFMPEG backend; local indices use the default."""
        import cv2  # lazy: only needed for a real camera

        if isinstance(target, str) and not target.isdigit():
            cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(int(target))
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
        return None

    def _open(self):
        """Open the configured device with retries. Raises CameraUnavailable on failure."""
        import time

        targets = self._resolve_targets()
        for attempt in range(1, self.open_retries + 1):
            for target in targets:
                cap = self._open_one(target)
                if cap is not None:
                    if target != self.device:
                        _log.info("camera: opened %r (from %r)", target, self.device)
                    return cap
            if attempt < self.open_retries:
                _log.warning(
                    "camera: could not open %r (attempt %d/%d), retrying...",
                    self.device, attempt, self.open_retries,
                )
                time.sleep(self.retry_delay_s)
        raise CameraUnavailable(f"could not open camera {self.device!r}")

    def _resolve_targets(self) -> list:
        """The ordered list of capture targets to try for this device setting."""
        if isinstance(self.device, str) and self.device.lower() == "auto":
            return list(range(self._AUTO_SCAN_RANGE))
        return [self.device]

    # --- streaming ---------------------------------------------------------------

    def frames(self) -> Iterator[np.ndarray]:
        import time

        cap = self._open()
        try:
            # Warm up by draining a few frames so the first inspected part isn't black.
            for _ in range(self.warmup_frames):
                cap.read()

            n = 0
            misses = 0
            while self.max_frames is None or n < self.max_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    # A dropped frame on a network stream is transient; tolerate a few
                    # before concluding the feed has ended.
                    misses += 1
                    if misses > 30:
                        _log.warning("camera: stream ended (%d consecutive misses)", misses)
                        break
                    time.sleep(0.05)
                    continue
                misses = 0
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


class FallbackSource(FrameSource):
    """Tries a primary source; if it can't produce frames (e.g. no camera at demo time),
    transparently streams from a fallback instead so a live run never dies on stage.

    The switch is decided lazily on the first frame. If the primary raises before
    yielding anything, we log a clear warning and hand off to the fallback. A failure
    *after* frames have started (a webcam unplugged mid-run) is not masked, since that
    is a real fault the operator should see."""

    def __init__(self, primary: FrameSource, fallback: FrameSource):
        self.primary = primary
        self.fallback = fallback

    def frames(self) -> Iterator[np.ndarray]:
        try:
            gen = self.primary.frames()
            first = next(gen)
        except StopIteration:
            return
        except CameraUnavailable as exc:
            _log.warning("%s, falling back to file replay", exc)
            yield from self.fallback.frames()
            return
        yield first
        yield from gen


class MockSource(FrameSource):
    """Yields a fixed list of pre-built arrays for tests."""

    def __init__(self, frames: Optional[list] = None):
        self._frames = frames or []

    def frames(self) -> Iterator[np.ndarray]:
        return iter(self._frames)
