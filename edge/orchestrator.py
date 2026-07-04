"""The main loop: frame -> perceive -> route -> (cloud | local) -> act -> log.

Wires every component behind its interface. The full happy path lands in M4
(full mode); M5 inserts the privacy filter on the escalation path; M6 adds the
degraded/offline branches and the outbox. This module stays thin — all policy
lives in router.py.
"""

import hashlib
import time
import uuid
from typing import Optional

from edge.actuator import Actuator
from edge.cloud_client import CloudClient, CloudUnreachable
from edge.config import Config
from edge.drift import DriftDetector, DriftState, ModelDriftMonitor
from edge.frame_source import FrameSource
from edge.network import NetworkController
from edge.outbox import Outbox
from edge.perception import OnnxClassifier
from edge.privacy import PrivacyFilter, PrivacyViolation
from edge.router import (
    Action,
    Decision,
    NetworkMode,
    decide,
    escalation_band,
    local_action,
)
from edge.store import InspectionEvent, Store


class Orchestrator:
    def __init__(
        self,
        config: Config,
        source: FrameSource,
        perception: OnnxClassifier,
        actuator: Actuator,
        store: Store,
        network: NetworkController,
        cloud: Optional[CloudClient] = None,
        privacy: Optional[PrivacyFilter] = None,
        outbox: Optional[Outbox] = None,
        category: str = "",
    ):
        self.config = config
        self.source = source
        self.perception = perception
        self.actuator = actuator
        self.store = store
        self.network = network
        self.cloud = cloud
        self.privacy = privacy
        self.outbox = outbox
        self.category = category  # non-PII context label sent with escalations
        self._drift: DriftDetector | None = None      # initialised on first frame with cal data
        self._model_drift = ModelDriftMonitor(config.model_drift)

    def run(self) -> list:
        """Process every frame from the source to completion. Returns the events logged."""
        events = []
        for frame in self.source.frames():
            self.network.probe()  # update FULL/DEGRADED/OFFLINE before each frame
            events.append(self.process_frame(frame))
        return events

    def process_frame(self, frame) -> InspectionEvent:
        """Run one item through perceive -> route -> act -> log and return its event.

        Actuation never depends on the cloud: if an escalation can't reach the cloud,
        we fall back to the cost-minimizing local action so the line never stalls (§3.6).
        """
        started = time.time()
        costs = self.config.costs
        mode = self.network.mode
        event_id = str(uuid.uuid4())

        perception = self.perception.predict(frame)
        p = perception.p

        # Log confidence and check for calibration drift. On ALERT every frame is
        # treated as in-band (conservative escalation) until the operator recalibrates.
        self.store.append_confidence(started, p)
        drift_alert = self._check_drift(p)

        decision = decide(p, mode, costs)
        if drift_alert and decision != Decision.ESCALATE and mode != NetworkMode.OFFLINE:
            decision = Decision.ESCALATE  # conservative: escalate everything under drift
        escalated = False
        bytes_to_cloud = 0
        pii_bytes = 0
        diagnosis: Optional[dict] = None
        outbox_state = "none"

        if decision == Decision.ESCALATE:
            action, escalated, bytes_to_cloud, pii_bytes, diagnosis = self._escalate(
                event_id, frame, p, started
            )
        elif decision == Decision.DEFER_AND_ACT:
            # Degraded link: queue the escalation, act on the local decision now so the
            # line never stalls; the cloud diagnosis is reconciled on reconnect.
            outbox_state = self._defer(event_id, frame, started)
            action = local_action(p, costs)
        else:
            # LOCAL_ACT: offline or outside the band. Act locally now — under the
            # asymmetry this conservatively rejects when in doubt (graceful degradation).
            action = local_action(p, costs)
            # Offline but in-band: this was a would-be escalation. Queue it so the cloud
            # can confirm the diagnosis once the link returns (build plan §3.5).
            if mode == NetworkMode.OFFLINE and self._in_band(p):
                outbox_state = self._defer(event_id, frame, started)

        fired = self.actuator.fire(action)

        event = InspectionEvent(
            id=event_id,
            ts=started,
            frame_hash=self._frame_hash(frame),
            p=p,
            uncertainty=perception.uncertainty,
            decision=action.value,
            escalated=escalated,
            network_mode=mode.value,
            action_fired=fired,
            latency_ms=(time.time() - started) * 1000.0,
            bytes_to_cloud=bytes_to_cloud,
            pii_bytes=pii_bytes,
            cloud_diagnosis=diagnosis,
            outbox_state=outbox_state,
        )
        self.store.insert_event(event)
        return event

    def _escalate(self, event_id: str, frame, p: float, ts: float):
        """Send a privacy-filtered payload to the cloud; fall back locally if unreachable.

        Returns (action, escalated, bytes_to_cloud, pii_bytes, diagnosis). Every byte/field
        that crosses the device boundary is recorded to the boundary log against event_id,
        making the 'zero PII egress' claim measurable.
        """
        costs = self.config.costs
        if self.cloud is None:
            return local_action(p, costs), False, 0, 0, None
        payload = self._filtered_payload(event_id, frame, ts)
        if payload is None:
            # No safe egress path / filter refused -> never transmit; act locally.
            return local_action(p, costs), False, 0, 0, None

        pii_bytes = payload.pii_bytes
        bytes_to_cloud = payload.total_bytes
        try:
            diagnosis = self.cloud.diagnose(
                roi_png_b64=_b64(payload.roi_png),
                embedding=payload.embedding,
                context=payload.context,
            )
        except CloudUnreachable:
            # Cloud gone mid-call: fall back to the local action, log no diagnosis.
            return local_action(p, costs), True, bytes_to_cloud, pii_bytes, None

        cloud_defect = bool(diagnosis.get("defect_present"))
        action = Action.REJECT if cloud_defect else Action.PASS
        edge_defect = local_action(p, costs) == Action.REJECT
        self._model_drift.record(edge_defect, cloud_defect)
        if self._model_drift.is_drifted:
            import logging
            logging.getLogger(__name__).warning(
                "edge-vs-cloud disagreement rate %.0f%% over last %d frames — "
                "consider recalibrating the edge model",
                self._model_drift.disagreement_rate * 100,
                self.config.model_drift.window,
            )
        return action, True, bytes_to_cloud, pii_bytes, diagnosis

    def _defer(self, event_id: str, frame, ts: float) -> str:
        """Queue a deferred escalation for later reconnect. Returns the outbox_state
        ('queued' if queued, 'none' if no safe egress path or no outbox is wired)."""
        if self.outbox is None:
            return "none"
        payload = self._filtered_payload(event_id, frame, ts)
        if payload is None:
            return "none"
        self.outbox.enqueue(event_id, ts, {
            "roi_png_b64": _b64(payload.roi_png),
            "embedding": payload.embedding,
            "context": payload.context,
        })
        return "queued"

    def reconcile(self) -> int:
        """Drain the outbox on reconnect: send queued payloads to the cloud and write the
        late diagnoses back onto their events. Returns the number reconciled."""
        if self.outbox is None or self.cloud is None:
            return 0
        return self.outbox.drain(
            lambda payload: self.cloud.diagnose(
                roi_png_b64=payload.get("roi_png_b64", ""),
                embedding=payload.get("embedding"),
                context=payload.get("context", {}),
            )
        )

    def _filtered_payload(self, event_id: str, frame, ts: float):
        """Build the privacy-filtered payload and log its boundary crossings. Returns the
        payload, or None if there's no safe egress path (no filter, or it refused)."""
        if self.privacy is None:
            return None
        try:
            payload = self.privacy.filter(
                frame, self._roi_bbox(frame), context={"category": self.category}
            )
        except PrivacyViolation:
            return None
        self.store.log_boundary(event_id, ts, payload.crossings)
        return payload

    @property
    def drift_state(self) -> DriftState:
        return self._drift.state if self._drift else DriftState.OK

    @property
    def model_drift_rate(self) -> float:
        """Rolling edge-vs-cloud disagreement rate over recent escalated frames."""
        return self._model_drift.disagreement_rate

    def _check_drift(self, p: float) -> bool:
        """Lazy-init the drift detector from the reference distribution stored in the
        calibration file, then record p and maybe run the KS test. Returns True on alert."""
        if self._drift is None:
            ref = self._load_reference_distribution()
            if ref is None:
                return False
            self._drift = DriftDetector(ref, self.config.drift)
        recent = self.store.recent_confidences(self.config.drift.window)
        state = self._drift.record_and_maybe_check(p, recent)
        return state == DriftState.ALERT

    def _load_reference_distribution(self):
        """Load the reference confidence distribution saved alongside the calibration.
        Returns None if the file doesn't exist or lacks the key."""
        import logging
        import json
        from pathlib import Path
        _log = logging.getLogger(__name__)
        cal_path = Path(self.config.paths.calibration)
        if not cal_path.exists():
            return None
        try:
            data = json.loads(cal_path.read_text())
            ref = data.get("reference_confidences")
            if ref is None:
                _log.warning(
                    "calibration file %s has no reference_confidences — "
                    "drift detection disabled; re-run eval.fit_calibration", cal_path
                )
            return ref
        except Exception as exc:
            _log.warning(
                "failed to load reference distribution from %s: %s — "
                "drift detection disabled", cal_path, exc
            )
            return None

    def _in_band(self, p: float) -> bool:
        band = escalation_band(self.config.costs)
        return band is not None and band[0] <= p <= band[1]

    def _roi_bbox(self, frame):
        """Default ROI: a centered crop inset from the edges, strictly smaller than the
        frame so the privacy filter never sees a full-frame request. A detector can
        replace this with a tight part localization later."""
        h, w = frame.shape[:2]
        inset_w, inset_h = max(1, w // 8), max(1, h // 8)
        return (inset_w, inset_h, w - 2 * inset_w, h - 2 * inset_h)

    @staticmethod
    def _frame_hash(frame) -> str:
        return hashlib.sha256(frame.tobytes()).hexdigest()


def _b64(data) -> str:
    import base64

    return base64.b64encode(data).decode("ascii") if data else ""
