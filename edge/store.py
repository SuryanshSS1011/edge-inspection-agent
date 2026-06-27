"""SQLite state, log, and outbox. The InspectionEvent record is both the audit
log row and the eval dataset row (build plan §3.7).

This module owns the schema; outbox.py operates on the `outbox`/`events` tables
through this store. Implementations land in M4 (events), M5 (boundary_log), M6 (outbox).
"""

import json
import sqlite3
from dataclasses import dataclass
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
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT NOT NULL,
    ts          REAL NOT NULL,
    field       TEXT NOT NULL,            -- what crossed the device boundary
    nbytes      INTEGER NOT NULL,
    is_pii      INTEGER NOT NULL DEFAULT 0,
    blocked     INTEGER NOT NULL DEFAULT 0  -- 1 if the filter refused to let it cross
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


_EVENT_COLUMNS = (
    "id", "ts", "frame_hash", "p", "uncertainty", "decision", "escalated",
    "network_mode", "action_fired", "latency_ms", "bytes_to_cloud", "pii_bytes",
    "cloud_diagnosis", "outbox_state",
)


def _row_to_event(row: sqlite3.Row) -> InspectionEvent:
    diagnosis = json.loads(row["cloud_diagnosis"]) if row["cloud_diagnosis"] else None
    return InspectionEvent(
        id=row["id"],
        ts=row["ts"],
        frame_hash=row["frame_hash"],
        p=row["p"],
        uncertainty=row["uncertainty"],
        decision=row["decision"],
        escalated=bool(row["escalated"]),
        network_mode=row["network_mode"],
        action_fired=row["action_fired"],
        latency_ms=row["latency_ms"],
        bytes_to_cloud=row["bytes_to_cloud"],
        pii_bytes=row["pii_bytes"],
        cloud_diagnosis=diagnosis,
        outbox_state=row["outbox_state"],
    )


class Store:
    """SQLite-backed log, state, and outbox. The events table is also the eval dataset."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def insert_event(self, event: InspectionEvent) -> None:
        diagnosis = json.dumps(event.cloud_diagnosis) if event.cloud_diagnosis is not None else None
        self._conn.execute(
            f"INSERT INTO events ({', '.join(_EVENT_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in _EVENT_COLUMNS)})",
            (
                event.id, event.ts, event.frame_hash, event.p, event.uncertainty,
                event.decision, int(event.escalated), event.network_mode,
                event.action_fired, event.latency_ms, event.bytes_to_cloud,
                event.pii_bytes, diagnosis, event.outbox_state,
            ),
        )
        self._conn.commit()

    def update_diagnosis(self, event_id: str, diagnosis: dict) -> None:  # M6 reconcile
        self._conn.execute(
            "UPDATE events SET cloud_diagnosis = ?, outbox_state = 'reconciled' WHERE id = ?",
            (json.dumps(diagnosis), event_id),
        )
        self._conn.commit()

    def log_boundary(self, event_id: str, ts: float, crossings) -> None:
        """Persist every CrossingRecord for an escalation to the boundary log. This log
        is the audit trail behind the measured 'zero PII egress' claim."""
        self._conn.executemany(
            "INSERT INTO boundary_log (event_id, ts, field, nbytes, is_pii, blocked) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (event_id, ts, c.field, c.nbytes, int(c.is_pii), int(getattr(c, "blocked", False)))
                for c in crossings
            ],
        )
        self._conn.commit()

    def boundary_rows(self) -> list:
        """Return boundary-log rows as dicts (for eval.metrics.pii_bytes_out)."""
        rows = self._conn.execute(
            "SELECT event_id, ts, field, nbytes, is_pii, blocked FROM boundary_log ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_event(self, event_id: str) -> Optional[InspectionEvent]:
        row = self._conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return _row_to_event(row) if row else None

    def all_events(self) -> list:
        rows = self._conn.execute("SELECT * FROM events ORDER BY ts").fetchall()
        return [_row_to_event(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
