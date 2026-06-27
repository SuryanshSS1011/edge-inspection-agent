"""M6 tests for the outbox: enqueue, drain/reconcile, and failure-retry semantics."""

from edge.outbox import Outbox
from edge.store import InspectionEvent, Store


def _seed_event(store, eid):
    store.insert_event(InspectionEvent(
        id=eid, ts=1.0, frame_hash="h", p=0.3, uncertainty=0.6, decision="REJECT",
        escalated=False, network_mode="degraded", action_fired="mock:REJECT",
        latency_ms=1.0, outbox_state="none",
    ))


def test_enqueue_marks_event_queued(tmp_path):
    store = Store(str(tmp_path / "edge.db"))
    _seed_event(store, "e1")
    Outbox(store).enqueue("e1", 2.0, {"roi_png_b64": "x", "embedding": None, "context": {}})
    assert store.get_event("e1").outbox_state == "queued"
    assert len(store.pending_outbox()) == 1


def test_drain_reconciles_and_writes_diagnosis(tmp_path):
    store = Store(str(tmp_path / "edge.db"))
    _seed_event(store, "e1")
    ob = Outbox(store)
    ob.enqueue("e1", 2.0, {"roi_png_b64": "x", "embedding": None, "context": {}})

    diagnosis = {"defect_present": True, "defect_type": "crack"}
    count = ob.drain(lambda payload: diagnosis)

    assert count == 1
    event = store.get_event("e1")
    assert event.cloud_diagnosis == diagnosis
    assert event.outbox_state == "reconciled"
    assert ob.pending_count() == 0


def test_failed_drain_leaves_item_queued(tmp_path):
    store = Store(str(tmp_path / "edge.db"))
    _seed_event(store, "e1")
    ob = Outbox(store)
    ob.enqueue("e1", 2.0, {"roi_png_b64": "x", "embedding": None, "context": {}})

    def still_down(payload):
        raise RuntimeError("cloud unreachable")

    assert ob.drain(still_down) == 0
    assert ob.pending_count() == 1  # retained for the next reconnect

    # Recovered: a later drain reconciles it.
    assert ob.drain(lambda p: {"defect_present": False}) == 1
    assert ob.pending_count() == 0


def test_drain_order_is_oldest_first(tmp_path):
    store = Store(str(tmp_path / "edge.db"))
    for eid, ts in [("a", 3.0), ("b", 1.0), ("c", 2.0)]:
        _seed_event(store, eid)
        Outbox(store).enqueue(eid, ts, {"roi_png_b64": "x", "embedding": None, "context": {}})
    seen = []
    Outbox(store).drain(lambda p: seen.append(1) or {"defect_present": False})
    assert len(seen) == 3
    assert [eid for eid, _ in store.pending_outbox()] == []  # all drained
