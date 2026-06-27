"""The main loop: frame -> perceive -> route -> (cloud | local) -> act -> log.

Wires every component behind its interface. The full happy path lands in M4
(full mode); M5 inserts the privacy filter on the escalation path; M6 adds the
degraded/offline branches and the outbox. This module stays thin — all policy
lives in router.py.
"""

import hashlib
import time
from typing import Optional

from edge.actuator import Actuator
from edge.cloud_client import CloudClient
from edge.config import Config
from edge.frame_source import FrameSource
from edge.network import NetworkController
from edge.outbox import Outbox
from edge.perception import OnnxClassifier
from edge.privacy import PrivacyFilter
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

    def run(self) -> None:  # M4 (then extended in M5/M6)
        """Process every frame from the source to completion.

        For each frame:
          1. perceive -> calibrated p, uncertainty
          2. decide(p, network.mode, costs) -> ESCALATE | DEFER_AND_ACT | LOCAL_ACT
          3. on ESCALATE: privacy.filter -> cloud.diagnose -> final action
             on DEFER_AND_ACT: outbox.enqueue + local_action now
             on LOCAL_ACT: local_action
          4. actuator.fire(action); persist a complete InspectionEvent
        """
        raise NotImplementedError

    @staticmethod
    def _frame_hash(frame) -> str:
        return hashlib.sha256(frame.tobytes()).hexdigest()
