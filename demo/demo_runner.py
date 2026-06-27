"""Scripted demo (build plan §8). Narrates each beat to the console so the video shows
clear on-screen state:

  1. good part        -> confident local PASS (no cloud)
  2. ambiguous part   -> escalate -> cloud diagnosis -> relay REJECTS
  3. CUT THE NETWORK  -> offline: conservative local REJECT, escalation queued
  4. RECONNECT        -> outbox drains, the late cloud diagnosis is back-filled
  5. summary          -> the log proves zero PII egress and the deferred-then-reconciled item

Self-contained and deterministic: a modeled cloud + scripted local probabilities, a mock
relay, and an in-memory DB. Record this now; swap in the real webcam / ONNX model /
deployed cloud (build_live in edge.app) for the final cut — the loop code is identical.
"""

import tempfile
from dataclasses import dataclass

import numpy as np  # type: ignore

from demo.net_toggle import set_mode
from edge.actuator import MockActuator
from edge.config import Config, Paths
from edge.frame_source import MockSource
from edge.network import NetworkController
from edge.orchestrator import Orchestrator
from edge.outbox import Outbox
from edge.perception import Perception
from edge.privacy import PrivacyFilter
from edge.router import Costs, NetworkMode
from edge.store import Store


@dataclass
class Beat:
    p: float       # local calibrated defect probability for this staged part
    label: int     # ground truth (1 = defect) — drives the modeled cloud verdict
    caption: str


class _ScriptedPerception:
    def __init__(self, beats):
        self._beats = list(beats)
        self.i = -1

    def predict(self, frame) -> Perception:
        self.i += 1
        p = self._beats[self.i].p
        return Perception(p=p, uncertainty=1.0 - abs(2 * p - 1))


class _ScriptedCloud:
    """Deterministic, accurate cloud verdict keyed to the beat being inspected."""

    def __init__(self, perception: "_ScriptedPerception", beats):
        self._perception = perception
        self._beats = list(beats)
        self.calls = 0

    def diagnose(self, roi_png_b64="", embedding=None, context=None):
        self.calls += 1
        label = self._beats[self._perception.i].label
        present = bool(label)
        return {
            "defect_present": present,
            "defect_type": "hairline_crack" if present else "none",
            "confidence": 0.94,
            "root_cause": "thermal stress during cure" if present else "n/a",
            "recommended_action": "reject_and_flag_batch" if present else "pass",
        }

    def healthz(self):
        return True


def _say(msg: str) -> None:
    print(msg, flush=True)


def run_demo(config_path: str = "config.yaml") -> dict:
    costs = Costs(C_FN=100.0, C_FP=5.0, C_cloud=2.0, residual_cloud_error=0.3)

    # Three staged parts. p* ≈ 0.048, escalation band ≈ [0.023, 0.54].
    good = Beat(p=0.005, label=0, caption="clean part")
    ambiguous = Beat(p=0.30, label=1, caption="ambiguous part (real defect)")
    cut = Beat(p=0.30, label=1, caption="ambiguous part during the outage")
    beats = [good, ambiguous, cut]

    with tempfile.TemporaryDirectory() as d:
        store = Store(f"{d}/demo.db")
        perception = _ScriptedPerception(beats)
        cloud = _ScriptedCloud(perception, beats)
        orch = Orchestrator(
            config=Config(costs=costs, default_mode=NetworkMode.FULL,
                          paths=Paths(db=f"{d}/demo.db", model="", calibration="")),
            source=MockSource([np.zeros((16, 16, 3), dtype=np.uint8) for _ in beats]),
            perception=perception,
            actuator=MockActuator(),
            store=store,
            network=NetworkController(NetworkMode.FULL),
            cloud=cloud,
            privacy=PrivacyFilter(mode="embedding"),
            outbox=Outbox(store),
            category="bottle",
        )

        _say("=" * 64)
        _say("EdgeAgent — live inspection demo")
        _say("=" * 64)

        # Beat 1: good part -> confident local PASS.
        _say("\n[1] A clean part enters the frame...")
        e1 = orch.process_frame(next_frame(orch))
        _say(f"    p={e1.p:.3f}  -> {e1.decision}  (local, no cloud call)  relay: {e1.action_fired}")

        # Beat 2: ambiguous part -> escalate -> cloud diagnosis -> REJECT.
        _say("\n[2] An ambiguous part enters — the router escalates to the cloud...")
        e2 = orch.process_frame(next_frame(orch))
        diag = e2.cloud_diagnosis or {}
        _say(f"    p={e2.p:.3f}  -> ESCALATE  ({e2.bytes_to_cloud} bytes crossed, PII={e2.pii_bytes})")
        _say(f"    cloud: defect={diag.get('defect_present')} type={diag.get('defect_type')} "
             f"cause='{diag.get('root_cause')}'")
        _say(f"    -> {e2.decision}  relay: {e2.action_fired}")

        # Beat 3: CUT THE NETWORK — the killer beat.
        _say("\n[3] *** NETWORK CUT *** the cloud is now unreachable...")
        set_mode(orch.network, "offline")
        e3 = orch.process_frame(next_frame(orch))
        _say(f"    p={e3.p:.3f}  -> {e3.decision}  (conservative local policy: reject when unsure)")
        _say(f"    escalation deferred to outbox: {orch.outbox.pending_count()} queued; line never stalled")

        # Beat 4: RECONNECT — drain + reconcile.
        _say("\n[4] *** NETWORK RESTORED *** draining the outbox...")
        set_mode(orch.network, "full")
        reconciled = orch.reconcile()
        e3b = orch.store.get_event(e3.id)
        rdiag = e3b.cloud_diagnosis or {}
        _say(f"    reconciled {reconciled} deferred item(s); diagnosis back-filled: "
             f"defect={rdiag.get('defect_present')} type={rdiag.get('defect_type')}")

        # Beat 5: summary from the log.
        _say("\n[5] Audit summary (from the local log):")
        events = orch.store.all_events()
        total_pii = sum(e.pii_bytes for e in events)
        total_bytes = sum(e.bytes_to_cloud for e in events)
        _say(f"    items inspected : {len(events)}")
        _say(f"    bytes to cloud  : {total_bytes}  |  PII bytes out: {total_pii}  (target 0)")
        _say(f"    deferred->synced: {sum(1 for e in events if e.outbox_state == 'reconciled')}")
        _say("=" * 64)

        return {
            "events": [e.id for e in events],
            "pii_bytes_out": total_pii,
            "reconciled": reconciled,
        }


# The demo drives frames one at a time; MockSource yields them in order.
def next_frame(orch):
    if not hasattr(orch, "_frame_iter"):
        orch._frame_iter = orch.source.frames()
    return next(orch._frame_iter)


if __name__ == "__main__":
    run_demo()
