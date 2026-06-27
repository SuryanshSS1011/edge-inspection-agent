"""Outbox: SQLite queue of deferred escalations that drains and reconciles on
reconnect (§3.5). Late cloud diagnosis updates the already-logged event row. M6.
"""

from typing import Callable, Optional

from edge.store import Store


class Outbox:
    def __init__(self, store: Store):
        self.store = store

    def enqueue(self, event_id: str, payload: dict) -> None:  # M6
        """Queue a deferred escalation (degraded/offline). Marks event outbox_state=queued."""
        raise NotImplementedError

    def drain(self, call_cloud: Callable[[dict], dict]) -> int:  # M6
        """On reconnect: send each queued payload to the cloud, write the diagnosis
        back onto its event (outbox_state=reconciled), mark drained. Returns count drained.
        """
        raise NotImplementedError
