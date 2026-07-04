"""Outbox: SQLite queue of deferred escalations that drains and reconciles on
reconnect (§3.5).

When the link is degraded or offline we can't escalate live, so the privacy-filtered
payload is queued and the line acts on the local decision immediately. On reconnect the
outbox drains: each payload goes to the cloud and the late diagnosis is written back onto
the already-logged event (outbox_state -> reconciled). Only the non-PII filtered payload
is ever persisted, so the queue carries no raw frames.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from edge.store import Store

_DRAIN_WORKERS = 4


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

    def drain(self, call_cloud: Callable[[dict], dict], workers: int = _DRAIN_WORKERS) -> int:
        """On reconnect: send queued payloads to the cloud concurrently, write each
        diagnosis back onto its event, mark it drained. Returns the number reconciled.

        Payloads that fail (cloud still unreachable) are left queued for the next drain.
        Up to `workers` requests run in parallel to avoid sequential backlog after a long
        offline period.
        """
        pending = self.store.pending_outbox()
        if not pending:
            return 0

        def _call(item):
            event_id, payload = item
            diagnosis = call_cloud(payload)
            return event_id, diagnosis

        reconciled = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_call, item): item[0] for item in pending}
            for future in as_completed(futures):
                event_id = futures[future]
                try:
                    event_id, diagnosis = future.result()
                except Exception:  # noqa: BLE001 - leave queued, retry next reconnect
                    continue
                self.store.update_diagnosis(event_id, diagnosis)
                self.store.mark_drained(event_id)
                reconciled += 1
        return reconciled
