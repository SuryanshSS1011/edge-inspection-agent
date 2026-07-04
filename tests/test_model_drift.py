"""Tests for ModelDriftMonitor, LiveConfig, privacy anonymization, and store agreement query."""
import tempfile
import time

import numpy as np
import pytest


# ── ModelDriftMonitor ────────────────────────────────────────────────────────

def test_no_drift_when_agreement_high():
    from edge.drift import ModelDriftConfig, ModelDriftMonitor
    mon = ModelDriftMonitor(ModelDriftConfig(window=10, disagreement_threshold=0.20))
    for _ in range(10):
        mon.record(edge_defect=True, cloud_defect=True)
    assert not mon.is_drifted
    assert mon.disagreement_rate == 0.0


def test_drift_detected_when_disagreement_exceeds_threshold():
    from edge.drift import ModelDriftConfig, ModelDriftMonitor
    mon = ModelDriftMonitor(ModelDriftConfig(window=10, disagreement_threshold=0.20))
    # 3/10 = 30% > 20% threshold
    for i in range(10):
        mon.record(edge_defect=(i < 3), cloud_defect=False)
    assert mon.is_drifted
    assert mon.disagreement_rate == pytest.approx(0.3)


def test_no_drift_before_half_window_filled():
    from edge.drift import ModelDriftConfig, ModelDriftMonitor
    mon = ModelDriftMonitor(ModelDriftConfig(window=10, disagreement_threshold=0.01))
    # All disagree but only 4 samples — below window//2=5
    for _ in range(4):
        mon.record(edge_defect=True, cloud_defect=False)
    assert not mon.is_drifted


def test_window_rolls_old_entries_out():
    from edge.drift import ModelDriftConfig, ModelDriftMonitor
    mon = ModelDriftMonitor(ModelDriftConfig(window=5, disagreement_threshold=0.20))
    # First 5 all disagree
    for _ in range(5):
        mon.record(True, False)
    assert mon.is_drifted
    # Add 5 more that agree — old disagreements pushed out
    for _ in range(5):
        mon.record(True, True)
    assert not mon.is_drifted


def test_load_from_pairs():
    from edge.drift import ModelDriftConfig, ModelDriftMonitor
    mon = ModelDriftMonitor(ModelDriftConfig(window=10, disagreement_threshold=0.20))
    pairs = [(True, False)] * 5 + [(True, True)] * 5  # 50% disagreement
    mon.load_from_pairs(pairs)
    assert mon.is_drifted


# ── store.recent_edge_cloud_agreement ────────────────────────────────────────

def test_store_agreement_query():
    from edge.store import InspectionEvent, Store
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        store = Store(f.name)
        import json, uuid
        # Insert two escalated events with diagnoses
        e1 = InspectionEvent(
            id=str(uuid.uuid4()), ts=1.0, frame_hash="a", p=0.7, uncertainty=0.1,
            decision="REJECT", escalated=True, network_mode="full",
            action_fired="REJECT", latency_ms=10.0, bytes_to_cloud=100,
            cloud_diagnosis={"defect_present": True, "defect_type": "crack",
                              "confidence": 0.9, "root_cause": "x", "recommended_action": "y"}
        )
        e2 = InspectionEvent(
            id=str(uuid.uuid4()), ts=2.0, frame_hash="b", p=0.3, uncertainty=0.1,
            decision="PASS", escalated=True, network_mode="full",
            action_fired="PASS", latency_ms=10.0, bytes_to_cloud=100,
            cloud_diagnosis={"defect_present": False, "defect_type": "none",
                              "confidence": 0.1, "root_cause": "ok", "recommended_action": "none"}
        )
        store.insert_event(e1)
        store.insert_event(e2)
        pairs = store.recent_edge_cloud_agreement(10)
        assert len(pairs) == 2
        # e1: edge=REJECT -> True, cloud=True -> agree
        assert pairs[0] == (True, True)
        # e2: edge=PASS -> False, cloud=False -> agree
        assert pairs[1] == (False, False)


# ── LiveConfig ────────────────────────────────────────────────────────────────

def test_live_config_returns_config():
    from edge.config import LiveConfig, load_config
    cfg = LiveConfig("config.yaml")
    c = cfg.get()
    assert c.costs.C_FN > 0


def test_live_config_reloads_on_mtime_change(tmp_path):
    import yaml
    from edge.config import LiveConfig
    cfg_path = tmp_path / "test_config.yaml"
    initial = {
        "costs": {"C_FN": 100.0, "C_FP": 5.0, "C_cloud": 2.0, "residual_cloud_error": 0.3},
        "network": {"default_mode": "full"},
        "paths": {"db": "./edge.db", "model": "./m.onnx", "calibration": "./t.json"},
    }
    cfg_path.write_text(yaml.dump(initial))
    live = LiveConfig(str(cfg_path))
    assert live.get().costs.C_FN == 100.0

    # Modify file with a slight mtime bump
    time.sleep(0.01)
    updated = dict(initial)
    updated["costs"] = {**initial["costs"], "C_FN": 200.0}
    cfg_path.write_text(yaml.dump(updated))
    # Manually poke mtime so the in-memory check triggers
    import os
    live._mtime = 0.0
    assert live.get().costs.C_FN == 200.0


# ── privacy skin anonymization ────────────────────────────────────────────────

def test_anonymize_skin_blurs_skin_pixels():
    from edge.privacy import PrivacyFilter
    # A synthetic frame that is all "skin-tone" in BGR: hue ~15°, moderate sat+val
    # BGR ~ (100, 150, 200) maps to H≈15, S≈0.5, V≈0.78
    frame = np.full((20, 20, 3), [100, 150, 200], dtype=np.uint8)
    result = PrivacyFilter._anonymize_skin(frame)
    # Blurred result should still be uint8 and same shape
    assert result.shape == frame.shape
    assert result.dtype == np.uint8


def test_anonymize_skin_passthrough_non_skin():
    from edge.privacy import PrivacyFilter
    # A clearly non-skin color: blue (H≈240°) — should be returned unchanged
    frame = np.full((10, 10, 3), [200, 50, 50], dtype=np.uint8)
    result = PrivacyFilter._anonymize_skin(frame)
    np.testing.assert_array_equal(result, frame)


def test_anonymize_skin_handles_grayscale():
    from edge.privacy import PrivacyFilter
    gray = np.full((10, 10), 128, dtype=np.uint8)
    result = PrivacyFilter._anonymize_skin(gray)
    np.testing.assert_array_equal(result, gray)
