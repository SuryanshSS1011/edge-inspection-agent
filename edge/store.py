"""SQLite state, log, and outbox. The InspectionEvent record is both the audit
log row and the eval dataset row (build plan §3.7).

This module owns the schema; outbox.py operates on the `outbox`/`events` tables
through this store. Implementations land in M4 (events), M5 (boundary_log), M6 (outbox).
"""

import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    ts              REAL NOT NULL,
    frame_hash      TEXT NOT NULL,
    p               REAL NOT NULL,
    uncertainty     REAL NOT NULL,
    decision        TEXT NOT NULL,        -- PASS | REJECT
    escalated       INTEGER NOT NULL,
    network_mode    TEXT NOT NULL,        -- full | degraded | offline
    action_fired    TEXT NOT NULL,
    latency_ms      REAL NOT NULL,
    bytes_to_cloud  INTEGER NOT NULL DEFAULT 0,
    pii_bytes       INTEGER NOT NULL DEFAULT 0,
    cloud_diagnosis TEXT,                 -- JSON string or NULL
    outbox_state    TEXT NOT NULL DEFAULT 'none'  -- none | queued | reconciled
);

CREATE TABLE IF NOT EXISTS outbox (
    id          TEXT PRIMARY KEY,         -- same id as the event it defers
    enqueued_ts REAL NOT NULL,
    payload     TEXT NOT NULL,            -- privacy-filtered ROI/embedding ref + context
    drained     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS boundary_log (
    id          TEXT PRIMARY KEY,
    event_id    TEXT NOT NULL,
    ts          REAL NOT NULL,
    field       TEXT NOT NULL,            -- what crossed the device boundary
    nbytes      INTEGER NOT NULL,
    is_pii      INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass
class InspectionEvent:
    id: str
    ts: float
    frame_hash: str
    p: float
    uncertainty: float
    decision: str
    escalated: bool
    network_mode: str
    action_fired: str
    latency_ms: float
    bytes_to_cloud: int = 0
    pii_bytes: int = 0
    cloud_diagnosis: Optional[dict] = None
    outbox_state: str = "none"


class Store:
    """Thin SQLite wrapper. Full CRUD lands in M4."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def insert_event(self, event: InspectionEvent) -> None:  # M4
        raise NotImplementedError

    def update_diagnosis(self, event_id: str, diagnosis: dict) -> None:  # M6 reconcile
        raise NotImplementedError

    def close(self) -> None:
        self._conn.close()
