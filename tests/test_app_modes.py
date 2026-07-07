"""camera / data / demo CLI modes (edge/app.py).

camera and data run the *same* real orchestrator pipeline and differ only in the frame
source; demo is the separate scripted walkthrough (covered in test_demo.py). These tests
drive the app wiring directly with a tiny config, a mock cloud, and tmp images.
"""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from edge import app
from edge.config import Config, Paths
from edge.frame_source import CameraUnavailable, FileSource
from edge.router import Costs, NetworkMode


def _write_images(d, n, color=(120, 120, 120)):
    for i in range(n):
        Image.new("RGB", (32, 32), color).save(d / f"img_{i:02d}.png")


def _config(tmp_path):
    return Config(
        costs=Costs(C_FN=100.0, C_FP=5.0, C_cloud=2.0, residual_cloud_error=0.3),
        default_mode=NetworkMode.FULL,
        paths=Paths(db=f"{tmp_path}/e.db", model="", calibration=""),
    )


def _args(tmp_path, **over):
    base = dict(config="config.yaml", category="bottle", data=str(tmp_path), limit=None,
                camera="0", relay_port=None, cloud_url=None)
    base.update(over)
    return SimpleNamespace(**base)


class _FakeClassifier:
    """Stands in for OnnxClassifier so the pipeline runs without a real ONNX model."""
    def __init__(self, *a, **k):
        pass

    def predict(self, frame):
        from edge.perception import Perception
        return Perception(p=0.30, uncertainty=1.0 - abs(2 * 0.30 - 1))  # in-band -> escalate


class _FakeCloud:
    def __init__(self, *a, **k):
        pass

    def diagnose(self, roi_png_b64="", embedding=None, context=None):
        return {"defect_present": True, "defect_type": "crack", "confidence": 0.9,
                "root_cause": "x", "recommended_action": "reject"}

    def healthz(self):
        return True


# --- data mode: real pipeline over a folder ----------------------------------

def test_data_mode_runs_pipeline_over_images(tmp_path, monkeypatch, capsys):
    d = tmp_path / "bottle"
    d.mkdir()
    _write_images(d, 3)
    monkeypatch.setattr(app, "OnnxClassifier", _FakeClassifier)
    monkeypatch.setattr(app, "CloudClient", _FakeCloud)

    args = _args(tmp_path, cloud_url="http://fake")
    source = app._data_source(args)
    assert isinstance(source, FileSource)
    orch = app.build_orchestrator(args, _config(tmp_path), source)
    app._run_pipeline(orch, "data")

    out = capsys.readouterr().out
    assert "summary: 3 items" in out
    assert "PII 0" in out            # privacy claim holds through the real path
    assert "ESCALATE" in out         # p=0.30 is in-band


def test_data_mode_limit_bounds_frames(tmp_path):
    d = tmp_path / "bottle"
    d.mkdir()
    _write_images(d, 10)
    paths = app._image_paths(str(tmp_path), "bottle", limit=4)
    assert len(paths) == 4


def test_data_mode_errors_when_no_images(tmp_path):
    (tmp_path / "bottle").mkdir()
    with pytest.raises(SystemExit):
        app._data_source(_args(tmp_path))


# --- camera mode: falls back to data when no device --------------------------

def _fake_cv2_closed():
    mod = types.ModuleType("cv2")
    mod.CAP_FFMPEG = 1900

    class _Closed:
        def isOpened(self):
            return False
        def release(self):
            pass

    mod.VideoCapture = lambda *a: _Closed()
    return mod


def test_camera_mode_falls_back_to_data(tmp_path, monkeypatch):
    d = tmp_path / "bottle"
    d.mkdir()
    _write_images(d, 2)
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2_closed())
    monkeypatch.setattr(app, "OnnxClassifier", _FakeClassifier)

    # A camera with no fallback path would just be a bare WebcamSource; with images present
    # it wraps in a FallbackSource that replays them when the device won't open.
    args = _args(tmp_path)
    source = app._camera_source(args)
    frames = list(source.frames())
    assert len(frames) == 2  # device dead -> fell back to the 2 data images


def test_camera_mode_no_fallback_when_no_images(tmp_path, monkeypatch, caplog):
    (tmp_path / "bottle").mkdir()
    monkeypatch.setitem(sys.modules, "cv2", _fake_cv2_closed())
    source = app._camera_source(_args(tmp_path))
    # No fallback images -> a bare webcam that raises when the missing device is opened.
    with pytest.raises(CameraUnavailable):
        list(source.frames())


# --- camera parsing ----------------------------------------------------------

def test_parse_camera_index_vs_url():
    assert app._parse_camera("0") == 0
    assert app._parse_camera("2") == 2
    assert app._parse_camera("auto") == "auto"
    assert app._parse_camera("http://phone:8080/video") == "http://phone:8080/video"
