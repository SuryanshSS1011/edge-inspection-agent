"""Tests for the pluggable feature backbone (handcrafted vs. mobilenet vs. dinov2).

The MobileNet and DINOv2 ONNX files are large artifacts (gitignored / exported on the
cluster), so live-extraction tests are skipped when absent; the registry wiring is always
tested.
"""

import os

import pytest

from eval.train_classifier import _backbone


def test_backbone_registry_handcrafted():
    extract_many, dim = _backbone("handcrafted")
    assert dim == 23
    assert callable(extract_many)


def test_backbone_registry_mobilenet():
    extract_many, dim = _backbone("mobilenet")
    assert dim == 1000
    assert callable(extract_many)


def test_backbone_registry_dinov2():
    # The dim is the variant default (advisory); the real width is confirmed from the ONNX at
    # extract time. Registry wiring must resolve WITHOUT the ONNX present (no FileNotFoundError).
    extract_many, dim = _backbone("dinov2")
    assert isinstance(dim, int) and dim > 0
    assert callable(extract_many)


def test_backbone_registry_rejects_unknown():
    with pytest.raises(ValueError):
        _backbone("resnet-9000")


@pytest.mark.skipif(not os.path.isfile("models/mobilenetv2.onnx"),
                    reason="MobileNetV2 ONNX not present (download it to run)")
def test_mobilenet_extract_shape(tmp_path):
    from PIL import Image
    from eval.mobilenet_features import extract, FEATURE_DIM

    img = Image.new("RGB", (128, 128), (120, 90, 60))
    p = tmp_path / "x.png"
    img.save(p)
    feat = extract(str(p))
    assert feat.shape == (FEATURE_DIM,)


@pytest.mark.skipif(not os.path.isfile("models/dinov2.onnx"),
                    reason="DINOv2 ONNX not present (export it on ROAR to run)")
def test_dinov2_extract_shape(tmp_path):
    from PIL import Image
    from eval.dinov2_features import extract, _resolve_dim

    img = Image.new("RGB", (128, 128), (60, 120, 90))
    p = tmp_path / "x.png"
    img.save(p)
    feat = extract(str(p))
    assert feat.shape == (_resolve_dim(),)
