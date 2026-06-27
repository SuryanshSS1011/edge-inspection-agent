"""Outbox: SQLite queue of deferred escalations that drains and reconciles on
reconnect (§3.5).

When the link is degraded or offline we can't escalate live, so the privacy-filtered
payload is queued and the line acts on the local decision immediately. On reconnect the
outbox drains: each payload goes to the cloud and the late diagnosis is written back onto
the already-logged event (outbox_state -> reconciled). Only the non-PII filtered payload
is ever persisted, so the queue carries no raw frames.
"""

from typing import Callable

from edge.store import Store


class Outbox:
    def __init__(self, store: Store):
        self.store = store

    def enqueue(self, event_id: str, enqueued_ts: float, payload: dict) -> None:
        """Queue a deferred escalation (degraded/offline) and mark the event queued.

        payload carries the privacy-filtered representation:
            {"roi_png_b64": str, "embedding": list|None, "context": dict}
        """
        self.store.enqueue_outbox(event_id, enqueued_ts, payload)

    def pending_count(self) -> int:
        return len(self.store.pending_outbox())

    def drain(self, call_cloud: Callable[[dict], dict]) -> int:
        """On reconnect: send each queued payload to the cloud, write the diagnosis back
        onto its event (reconciled), mark it drained. Returns the number reconciled.

        A payload that fails (cloud still unreachable) is left queued for the next drain.
        """
        reconciled = 0
        for event_id, payload in self.store.pending_outbox():
            try:
                diagnosis = call_cloud(payload)
            except Exception:  # noqa: BLE001 - leave it queued and try again next reconnect
                continue
            self.store.update_diagnosis(event_id, diagnosis)
            self.store.mark_drained(event_id)
            reconciled += 1
        return reconciled
