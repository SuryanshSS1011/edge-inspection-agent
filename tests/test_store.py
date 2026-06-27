"""Tests for the SQLite store: round-trip an event, JSON diagnosis handling, and the
M6 reconcile update."""

from edge.store import InspectionEvent, Store


def _event(eid="e1", diagnosis=None):
    return InspectionEvent(
        id=eid,
        ts=123.0,
        frame_hash="abc",
        p=0.3,
        uncertainty=0.6,
        decision="REJECT",
        escalated=True,
        network_mode="full",
        action_fired="mock:REJECT",
        latency_ms=12.5,
        bytes_to_cloud=7,
        pii_bytes=0,
        cloud_diagnosis=diagnosis,
    )


def test_insert_and_get_roundtrip(tmp_path):
    store = Store(str(tmp_path / "edge.db"))
    diag = {"defect_present": True, "defect_type": "crack"}
    store.insert_event(_event(diagnosis=diag))
    got = store.get_event("e1")
    assert got is not None
    assert got.escalated is True
    assert got.cloud_diagnosis == diag
    assert got.bytes_to_cloud == 7


def test_null_diagnosis_roundtrips_as_none(tmp_path):
    store = Store(str(tmp_path / "edge.db"))
    store.insert_event(_event(diagnosis=None))
    assert store.get_event("e1").cloud_diagnosis is None


def test_update_diagnosis_reconciles(tmp_path):
    store = Store(str(tmp_path / "edge.db"))
    store.insert_event(_event(diagnosis=None))
    store.update_diagnosis("e1", {"defect_present": False})
    got = store.get_event("e1")
    assert got.cloud_diagnosis == {"defect_present": False}
    assert got.outbox_state == "reconciled"


def test_all_events_ordered(tmp_path):
    store = Store(str(tmp_path / "edge.db"))
    store.insert_event(_event("a"))
    store.insert_event(_event("b"))
    ids = [e.id for e in store.all_events()]
    assert ids == ["a", "b"]
