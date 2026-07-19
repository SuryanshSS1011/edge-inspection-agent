"""Eval harness internals that replay a labeled item stream through the real orchestrator
under each operating condition and read metrics back from its store.

Kept separate from run_eval.py so the per-condition runner is unit-testable. The stream
is a list of EvalItem(p, label): `p` is the local calibrated defect probability (from the
real model or a fixture) and `label` is ground truth (1=defect). The cloud is modeled by
a callable verdict so the table generates without credentials; swap in the real
CloudClient + OnnxClassifier to produce the live numbers.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np  # type: ignore

from edge.actuator import MockActuator
from edge.config import Config, Paths
from edge.network import NetworkController
from edge.orchestrator import Orchestrator
from edge.outbox import Outbox
from edge.perception import Perception
from edge.privacy import PrivacyFilter
from edge.router import Costs, NetworkMode
from edge.store import Store
from eval.metrics import (
    bytes_to_cloud_per_item,
    cloud_cost_per_1k,
    cost_weighted_recall,
    latency_percentiles,
    pii_bytes_out,
)


@dataclass
class EvalItem:
    p: float       # local calibrated defect probability
    label: int     # ground truth, 1 = defect


@dataclass
class ConditionResult:
    condition: str
    cost_weighted_recall: float
    p50_latency_ms: float
    p99_latency_ms: float
    bytes_to_cloud_per_item: float
    cloud_cost_per_1k: float
    pii_bytes_out: int
    n_escalations: int
    n_items: int
    n_deferred: int = 0


class _Cursor:
    """Shared per-item pointer. predict() advances it; diagnose() reads the same item, so
    the modeled cloud verdict is always matched to the item being inspected."""

    def __init__(self, items: List[EvalItem]):
        self.items = items
        self.i = -1

    @property
    def current_label(self) -> int:
        return self.items[self.i].label


class _StreamPerception:
    """Yields the next item's p in order, instead of running a model."""

    def __init__(self, cursor: "_Cursor"):
        self._cursor = cursor

    def predict(self, frame) -> Perception:
        self._cursor.i += 1
        p = self._cursor.items[self._cursor.i].p
        return Perception(p=p, uncertainty=1.0 - abs(2 * p - 1))


class _Cloud:
    """Models the cloud verdict. accuracy = P(correct defect/no-defect call).

    The true label is threaded through the privacy-filtered context (key 'label') so the
    modeled verdict is matched to the *actual* item being diagnosed rather than to call
    order, which differs across conditions because each escalates a different subset.
    """

    def __init__(self, cursor: "_Cursor", accuracy: float = 0.98, seed: int = 0):
        self._cursor = cursor
        self._rng = np.random.default_rng(seed)
        self.accuracy = accuracy
        self.calls = 0

    def diagnose(self, roi_png_b64="", embedding=None, context=None):
        self.calls += 1
        label = self._cursor.current_label
        correct = self._rng.uniform() < self.accuracy
        present = bool(label) if correct else not bool(label)
        return {
            "defect_present": present,
            "defect_type": "defect" if present else "none",
            "confidence": self.accuracy,
            "root_cause": "modeled",
            "recommended_action": "reject" if present else "pass",
        }

    def healthz(self):
        return True


def _costs_for_condition(base: Costs, condition: str) -> Costs:
    """Force the escalation band per condition by tuning C_cloud.

    cloud_everything -> C_cloud≈0 so the band covers essentially all p (escalate all).
    local_only       -> C_cloud huge so the band is empty (never escalate).
    hybrid_*         -> the real configured costs.
    """
    if condition == "cloud_everything":
        return Costs(base.C_FN, base.C_FP, C_cloud=0.0, residual_cloud_error=0.0)
    if condition == "local_only":
        return Costs(base.C_FN, base.C_FP, C_cloud=1e9, residual_cloud_error=0.0)
    return base


_MODE = {
    "cloud_everything": NetworkMode.FULL,
    "local_only": NetworkMode.OFFLINE,
    "hybrid_full": NetworkMode.FULL,
    "hybrid_degraded": NetworkMode.DEGRADED,
    "hybrid_offline": NetworkMode.OFFLINE,
}


def run_condition(
    condition: str,
    items: List[EvalItem],
    base_costs: Costs,
    db_path: str,
    cloud_accuracy: float = 0.98,
    seed: int = 0,
) -> ConditionResult:
    """Replay the stream under one condition and compute its metrics row."""
    from edge.frame_source import MockSource

    labels = [it.label for it in items]
    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in items]
    costs = _costs_for_condition(base_costs, condition)
    mode = _MODE[condition]

    store = Store(db_path)
    cursor = _Cursor(items)
    cloud = _Cloud(cursor, accuracy=cloud_accuracy, seed=seed)
    # local_only has no safe egress path at all; others get the privacy filter + cloud.
    # Embedding mode keeps the eval dependency-free (no cv2) and sends no pixels at all.
    privacy = None if condition == "local_only" else PrivacyFilter(mode="embedding")

    orch = Orchestrator(
        config=Config(costs=costs, default_mode=mode,
                      paths=Paths(db=db_path, model="", calibration="")),
        source=MockSource(frames),
        perception=_StreamPerception(cursor),
        actuator=MockActuator(),
        store=store,
        network=NetworkController(mode),
        cloud=None if condition == "local_only" else cloud,
        privacy=privacy,
        outbox=Outbox(store),
        category="bottle",
    )
    orch.run()
    if condition == "hybrid_degraded":
        orch.reconcile()   # link returns; deferred escalations drain

    # Score on the real configured costs; the per-call dollar cost is the real C_cloud,
    # not the band-tuning value used to force each condition's escalation behavior.
    return _summarize(condition, store, labels, base_costs)


def _summarize(condition, store, labels, costs) -> ConditionResult:
    events = store.all_events()
    y_pred = [1 if e.decision == "REJECT" else 0 for e in events]
    latencies = [e.latency_ms for e in events]
    total_bytes = sum(e.bytes_to_cloud for e in events)
    n = len(events)
    lat = latency_percentiles(latencies)

    # Live escalations = the cloud was actually called now (bytes left the device this run).
    # These are the only ones that incur current cloud cost / egress in this mode.
    n_live = sum(1 for e in events if e.bytes_to_cloud > 0)
    # Deferred = would-be escalations queued for sync on reconnect. Their cost is realized
    # later (in the reconnect/sync row), NOT here. Booking it twice would double-count.
    n_deferred = sum(1 for e in events if e.outbox_state in ("queued", "reconciled"))

    return ConditionResult(
        condition=condition,
        cost_weighted_recall=cost_weighted_recall(labels, y_pred, costs),
        p50_latency_ms=lat["p50"],
        p99_latency_ms=lat["p99"],
        bytes_to_cloud_per_item=bytes_to_cloud_per_item(total_bytes, n),
        cloud_cost_per_1k=cloud_cost_per_1k(n_live, n, costs.C_cloud),
        pii_bytes_out=pii_bytes_out(store.boundary_rows()),
        n_escalations=n_live,
        n_items=n,
        n_deferred=n_deferred,
    )
