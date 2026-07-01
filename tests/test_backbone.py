"""Tests for the pluggable feature backbone (handcrafted vs. mobilenet).

The MobileNet ONNX is a large downloaded artifact (gitignored), so the live-extraction
test is skipped when it's absent; the registry wiring is always tested.
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
