"""Tests for the hardware I/O paths. Real camera/relay can't run in CI, so cv2 and
pyserial are mocked; FileSource is tested for real against on-disk images.
"""

import sys
import types

import numpy as np
import pytest

from edge.actuator import UsbRelayActuator
from edge.frame_source import (
    CameraUnavailable,
    FallbackSource,
    FileSource,
    MockSource,
    WebcamSource,
)
from edge.router import Action


# --- FileSource: real decode, no mocks ---------------------------------------

def test_filesource_reads_images_as_bgr(tmp_path):
    from PIL import Image
    # write two small RGB images with a known top-left pixel
    for i, color in enumerate([(255, 0, 0), (0, 0, 255)]):  # red, blue in RGB
        img = Image.new("RGB", (4, 4), color)
        img.save(tmp_path / f"{i}.png")

    frames = list(FileSource([str(tmp_path / "0.png"), str(tmp_path / "1.png")]).frames())
    assert len(frames) == 2
    # BGR order: a pure-red RGB pixel becomes (0,0,255) in BGR.
    assert tuple(frames[0][0, 0]) == (0, 0, 255)
    assert tuple(frames[1][0, 0]) == (255, 0, 0)  # blue RGB -> (255,0,0) BGR


def test_filesource_empty_list():
    assert list(FileSource([]).frames()) == []


# --- UsbRelayActuator: mock pyserial ------------------------------------------

class _FakeSerial:
    def __init__(self, *a, **k):
        self.written = []
    def write(self, data):
        self.written.append(bytes(data))
    def flush(self):
        pass
    def close(self):
        pass


@pytest.fixture()
def fake_serial(monkeypatch):
    mod = types.ModuleType("serial")
    mod.Serial = _FakeSerial
    monkeypatch.setitem(sys.modules, "serial", mod)
    return mod


def test_relay_reject_sends_on_command(fake_serial):
    act = UsbRelayActuator("/dev/ttyUSB0")
    label = act.fire(Action.REJECT)
    assert label == "relay:REJECT"
    assert act._serial.written[-1] == UsbRelayActuator.DEFAULT_ON


def test_relay_pass_sends_off_command(fake_serial):
    act = UsbRelayActuator("/dev/ttyUSB0")
    act.fire(Action.PASS)
    assert act._serial.written[-1] == UsbRelayActuator.DEFAULT_OFF


def test_relay_constructor_does_not_open_port(fake_serial):
    # Constructing must not touch hardware; the port opens only on first fire().
    act = UsbRelayActuator("/dev/ttyUSB0")
    assert act._serial is None


def test_relay_custom_commands(fake_serial):
    act = UsbRelayActuator("/dev/ttyUSB0", on_cmd=b"\x01", off_cmd=b"\x00")
    act.fire(Action.REJECT)
    assert act._serial.written[-1] == b"\x01"


# --- WebcamSource: mock cv2 ---------------------------------------------------

class _FakeCap:
    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0
    def isOpened(self):
        return True
    def read(self):
        if self._i < len(self._frames):
            f = self._frames[self._i]; self._i += 1
            return True, f
        return False, None
    def release(self):
        pass


def _fake_cv2(frames, on_open=None):
    """A fake cv2 module. VideoCapture accepts the (target[, backend]) signature the
    source now uses for URL streams, and records each target opened in `.opened`."""
    mod = types.ModuleType("cv2")
    mod.CAP_FFMPEG = 1900
    mod.opened = []

    def _video_capture(target, *args):
        mod.opened.append(target)
        cap = _FakeCap(frames)
        if on_open:
            on_open(target, cap)
        return cap

    mod.VideoCapture = _video_capture
    return mod


def test_webcam_yields_frames(monkeypatch):
    frames = [np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)]
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(frames))
    src = WebcamSource(interval_s=0, max_frames=5, warmup_frames=0)
    got = list(src.frames())
    assert len(got) == 2  # camera ran dry after 2


def test_webcam_respects_max_frames(monkeypatch):
    frames = [np.zeros((4, 4, 3), np.uint8)] * 10
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(frames))
    src = WebcamSource(interval_s=0, max_frames=3, warmup_frames=0)
    assert len(list(src.frames())) == 3


def test_webcam_warmup_discards_leading_frames(monkeypatch):
    # 5 frames total; warm-up drains 2, so only 3 are yielded.
    frames = [np.full((2, 2, 3), i, np.uint8) for i in range(5)]
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(frames))
    src = WebcamSource(interval_s=0, warmup_frames=2)
    got = list(src.frames())
    assert len(got) == 3
    assert got[0][0, 0, 0] == 2  # first yielded frame is the 3rd captured


def test_webcam_url_uses_ffmpeg_backend(monkeypatch):
    frames = [np.zeros((2, 2, 3), np.uint8)]
    fake = _fake_cv2(frames)
    monkeypatch.setitem(sys.modules, "cv2", fake)
    src = WebcamSource("http://phone.local:8080/video", interval_s=0, warmup_frames=0)
    list(src.frames())
    assert fake.opened == ["http://phone.local:8080/video"]


def test_webcam_auto_scans_indices(monkeypatch):
    # index 0 fails to open; auto should try 1 next and succeed.
    frames = [np.zeros((2, 2, 3), np.uint8)]
    fake = _fake_cv2(frames, on_open=lambda t, cap: setattr(
        cap, "isOpened", (lambda: t != 0)))
    monkeypatch.setitem(sys.modules, "cv2", fake)
    src = WebcamSource("auto", interval_s=0, warmup_frames=0, retry_delay_s=0)
    list(src.frames())
    assert fake.opened[:2] == [0, 1]  # tried 0, then 1


def test_webcam_raises_if_camera_unavailable(monkeypatch):
    fake = _fake_cv2([], on_open=lambda t, cap: setattr(cap, "isOpened", lambda: False))
    monkeypatch.setitem(sys.modules, "cv2", fake)
    with pytest.raises(CameraUnavailable):
        list(WebcamSource(open_retries=2, retry_delay_s=0).frames())


def test_fallback_uses_fallback_when_camera_unavailable(monkeypatch):
    fake = _fake_cv2([], on_open=lambda t, cap: setattr(cap, "isOpened", lambda: False))
    monkeypatch.setitem(sys.modules, "cv2", fake)
    fb_frames = [np.zeros((2, 2, 3), np.uint8), np.ones((2, 2, 3), np.uint8)]
    src = FallbackSource(
        WebcamSource(open_retries=1, retry_delay_s=0), MockSource(fb_frames)
    )
    got = list(src.frames())
    assert len(got) == 2  # camera died at open -> fell back to the 2 mock frames


def test_fallback_prefers_primary_when_camera_works(monkeypatch):
    cam_frames = [np.full((2, 2, 3), 7, np.uint8)]
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(cam_frames))
    src = FallbackSource(
        WebcamSource(interval_s=0, warmup_frames=0), MockSource([np.zeros((2, 2, 3), np.uint8)])
    )
    got = list(src.frames())
    assert len(got) == 1
    assert got[0][0, 0, 0] == 7  # primary frame, not the fallback
