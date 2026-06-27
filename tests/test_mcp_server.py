"""Tests for the diagnose_defect tool entrypoint — input handling and decoding,
with the Qwen call patched out (no network/API key needed).
"""

import base64

import pytest

import cloud.mcp_server as srv

GOOD = {
    "defect_present": True,
    "defect_type": "crack",
    "confidence": 0.91,
    "root_cause": "thermal stress",
    "recommended_action": "reject",
}


def test_requires_roi_or_embedding():
    with pytest.raises(ValueError):
        srv.diagnose_defect()


def test_decodes_roi_and_passes_bytes(monkeypatch):
    seen = {}

    def fake_diagnose(roi_png, embedding, context):
        seen["roi"] = roi_png
        seen["embedding"] = embedding
        return dict(GOOD)

    monkeypatch.setattr(srv, "diagnose", fake_diagnose)
    out = srv.diagnose_defect(roi_png_b64=base64.b64encode(b"PNGDATA").decode(), context={"c": 1})
    assert seen["roi"] == b"PNGDATA"
    assert out == GOOD


def test_embedding_path(monkeypatch):
    seen = {}

    def fake_diagnose(roi_png, embedding, context):
        seen["roi"] = roi_png
        seen["embedding"] = embedding
        return dict(GOOD)

    monkeypatch.setattr(srv, "diagnose", fake_diagnose)
    srv.diagnose_defect(embedding=[0.1, 0.2])
    assert seen["roi"] is None
    assert seen["embedding"] == [0.1, 0.2]
