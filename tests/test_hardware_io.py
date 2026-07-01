"""Tests for the hardware I/O paths. Real camera/relay can't run in CI, so cv2 and
pyserial are mocked; FileSource is tested for real against on-disk images.
"""

import sys
import types

import numpy as np
import pytest

from edge.actuator import UsbRelayActuator
from edge.frame_source import FileSource, WebcamSource
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


def _fake_cv2(frames):
    mod = types.ModuleType("cv2")
    mod.VideoCapture = lambda idx: _FakeCap(frames)
    return mod


def test_webcam_yields_frames(monkeypatch):
    frames = [np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)]
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(frames))
    src = WebcamSource(interval_s=0, max_frames=5)
    got = list(src.frames())
    assert len(got) == 2  # camera ran dry after 2


def test_webcam_respects_max_frames(monkeypatch):
    frames = [np.zeros((4, 4, 3), np.uint8)] * 10
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2(frames))
    src = WebcamSource(interval_s=0, max_frames=3)
    assert len(list(src.frames())) == 3


def test_webcam_raises_if_camera_unavailable(monkeypatch):
    mod = types.ModuleType("cv2")
    class _Closed(_FakeCap):
        def isOpened(self):
            return False
    mod.VideoCapture = lambda idx: _Closed([])
    monkeypatch.setitem(sys.modules, "cv2", mod)
    with pytest.raises(RuntimeError):
        list(WebcamSource().frames())
