"""M4 tests: the full-mode loop wires perceive -> route -> act -> log correctly, and
actuation never depends on the cloud.

Uses lightweight fakes (no ONNX model, no cv2, no network) injected behind the same
interfaces the real components implement.
"""

import numpy as np
import pytest

from edge.actuator import MockActuator
from edge.cloud_client import CloudUnreachable
from edge.config import Config, Paths
from edge.frame_source import MockSource
from edge.network import NetworkController
from edge.perception import Perception
from edge.privacy import CrossingRecord, FilteredPayload
from edge.router import Action, Costs, NetworkMode
from edge.store import Store

# Default costs: p* ≈ 0.048, escalation band ≈ [0.023, 0.54].
COSTS = Costs(C_FN=100.0, C_FP=5.0, C_cloud=2.0, residual_cloud_error=0.3)


def make_config(tmp_path):
    return Config(
        costs=COSTS,
        default_mode=NetworkMode.FULL,
        paths=Paths(db=str(tmp_path / "edge.db"), model="m.onnx", calibration="t.json"),
    )


class FakePerception:
    """Returns a fixed p per frame instead of running a model."""

    def __init__(self, p):
        self._p = p

    def predict(self, frame):
        return Perception(p=self._p, uncertainty=1.0 - abs(2 * self._p - 1))


class FakeCloud:
    def __init__(self, defect_present=True, raises=False):
        self.defect_present = defect_present
        self.raises = raises
        self.calls = 0

    def diagnose(self, roi_png_b64="", embedding=None, context=None):
        self.calls += 1
        if self.raises:
            raise CloudUnreachable("down")
        return {
            "defect_present": self.defect_present,
            "defect_type": "crack" if self.defect_present else "none",
            "confidence": 0.9,
            "root_cause": "x",
            "recommended_action": "reject" if self.defect_present else "pass",
        }


class FakePrivacy:
    """Returns a tiny ROI payload and logs the crossing, no real cropping."""

    def filter(self, frame, bbox):
        return FilteredPayload(
            roi_png=b"PNGDATA",
            embedding=None,
            crossings=[CrossingRecord(field="roi_png", nbytes=7, is_pii=False)],
        )


def _frame():
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _orchestrator(tmp_path, p, mode=NetworkMode.FULL, cloud=None, privacy=None):
    from edge.orchestrator import Orchestrator

    return Orchestrator(
        config=make_config(tmp_path),
        source=MockSource([_frame()]),
        perception=FakePerception(p),
        actuator=MockActuator(),
        store=Store(str(tmp_path / "edge.db")),
        network=NetworkController(mode),
        cloud=cloud,
        privacy=privacy,
    )


def test_confident_good_passes_locally(tmp_path):
    # p below the band -> local PASS, no escalation.
    orch = _orchestrator(tmp_path, p=0.001)
    [event] = orch.run()
    assert event.decision == Action.PASS.value
    assert event.escalated is False
    assert event.bytes_to_cloud == 0


def test_confident_defect_rejects_locally(tmp_path):
    # p above the band -> local REJECT, no escalation.
    orch = _orchestrator(tmp_path, p=0.9)
    [event] = orch.run()
    assert event.decision == Action.REJECT.value
    assert event.escalated is False


def test_in_band_escalates_and_uses_cloud_verdict(tmp_path):
    cloud = FakeCloud(defect_present=True)
    orch = _orchestrator(tmp_path, p=0.3, cloud=cloud, privacy=FakePrivacy())
    [event] = orch.run()
    assert event.escalated is True
    assert cloud.calls == 1
    assert event.decision == Action.REJECT.value
    assert event.cloud_diagnosis["defect_type"] == "crack"
    assert event.bytes_to_cloud == 7


def test_escalation_without_privacy_acts_locally(tmp_path):
    # No privacy filter => never transmit a raw frame; fall back to local action.
    cloud = FakeCloud()
    orch = _orchestrator(tmp_path, p=0.3, cloud=cloud, privacy=None)
    [event] = orch.run()
    assert cloud.calls == 0
    assert event.escalated is False
    assert event.bytes_to_cloud == 0


def test_actuation_survives_cloud_outage(tmp_path):
    # In-band escalation but cloud is down: must still fire a local action and log it.
    cloud = FakeCloud(raises=True)
    orch = _orchestrator(tmp_path, p=0.3, cloud=cloud, privacy=FakePrivacy())
    [event] = orch.run()
    assert event.escalated is True            # we tried
    assert event.cloud_diagnosis is None      # but got nothing back
    assert event.decision in (Action.PASS.value, Action.REJECT.value)
    assert event.action_fired.startswith("mock:")


def test_event_persisted_and_readable(tmp_path):
    orch = _orchestrator(tmp_path, p=0.9)
    [event] = orch.run()
    stored = orch.store.get_event(event.id)
    assert stored is not None
    assert stored.decision == event.decision
    assert stored.frame_hash == event.frame_hash
