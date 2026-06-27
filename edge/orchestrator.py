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
from edge.frame_source import FrameSource
from edge.network import NetworkController
from edge.outbox import Outbox
from edge.perception import OnnxClassifier
from edge.privacy import PrivacyFilter, PrivacyViolation
from edge.router import Action, Decision, decide, local_action
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

    def run(self) -> list:
        """Process every frame from the source to completion. Returns the events logged."""
        events = []
        for frame in self.source.frames():
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

        decision = decide(p, mode, costs)
        escalated = False
        bytes_to_cloud = 0
        pii_bytes = 0
        diagnosis: Optional[dict] = None

        if decision == Decision.ESCALATE:
            action, escalated, bytes_to_cloud, pii_bytes, diagnosis = self._escalate(
                event_id, frame, p, started
            )
        else:
            # DEFER_AND_ACT (M6) and LOCAL_ACT both act locally in full-mode M4.
            action = local_action(p, costs)

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
            outbox_state="none",
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
        if self.cloud is None or self.privacy is None:
            # No safe egress path -> never transmit a raw frame; act locally.
            return local_action(p, costs), False, 0, 0, None

        try:
            payload = self.privacy.filter(
                frame, self._roi_bbox(frame), context={"category": self.category}
            )
        except PrivacyViolation:
            # Filter refused (e.g. ROI == full frame): do not leak, act locally.
            return local_action(p, costs), False, 0, 0, None

        self.store.log_boundary(event_id, ts, payload.crossings)
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

        action = Action.REJECT if diagnosis.get("defect_present") else Action.PASS
        return action, True, bytes_to_cloud, pii_bytes, diagnosis

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
