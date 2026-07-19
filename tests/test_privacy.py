"""M5 tests that the privacy filter never lets the full frame or PII fields cross, and
that the boundary log measures exactly what does.

Cropping and measurement are pure numpy; we use embedding mode (no cv2 needed) for the
egress-measurement tests and only touch ROI/PNG behavior where the logic is cv2-free.
"""

import numpy as np
import pytest

from edge.privacy import (
    PII_CONTEXT_KEYS,
    PrivacyFilter,
    PrivacyViolation,
)


def _frame(h=64, w=64):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_embedding_mode_sends_no_pixels():
    pf = PrivacyFilter(mode="embedding")
    payload = pf.filter(_frame(), bbox=(8, 8, 32, 32))
    assert payload.roi_png is None
    assert payload.embedding is not None
    # Default embedding is per-channel mean+std => 6 numbers for a 3-channel ROI.
    assert len(payload.embedding) == 6


def test_full_frame_roi_is_refused():
    pf = PrivacyFilter(mode="roi")
    frame = _frame()
    # bbox covering the whole frame must raise rather than leak raw pixels.
    with pytest.raises(PrivacyViolation):
        pf.filter(frame, bbox=(0, 0, frame.shape[1], frame.shape[0]))


def test_pii_context_is_blocked_and_logged():
    pf = PrivacyFilter(mode="embedding")
    payload = pf.filter(
        _frame(),
        bbox=(8, 8, 32, 32),
        context={"category": "bottle", "operator": "alice", "serial": "SN-123"},
    )
    # Allowed key survives.
    assert payload.context == {"category": "bottle"}
    # PII keys are recorded as blocked crossings and contribute zero egress.
    blocked = {c.field for c in payload.crossings if c.blocked}
    assert "context.operator" in blocked
    assert "context.serial" in blocked
    assert payload.pii_bytes == 0


def test_unlisted_context_keys_are_dropped_and_logged():
    # Under default-deny an unanticipated key is NOT sent, and IS logged as a blocked crossing
    # so the audit trail can prove the filter caught an unknown identifier.
    pf = PrivacyFilter(mode="embedding")
    payload = pf.filter(_frame(), bbox=(8, 8, 32, 32), context={"random_note": "hi"})
    assert "random_note" not in payload.context          # not sent
    blocked = {c.field for c in payload.crossings if c.blocked}
    assert "context.random_note" in blocked              # but logged as caught
    assert payload.pii_bytes == 0                         # blocked => zero egress


def test_total_and_pii_bytes_accounting():
    pf = PrivacyFilter(mode="embedding")
    payload = pf.filter(_frame(), bbox=(8, 8, 32, 32), context={"category": "bottle"})
    # Embedding (6 floats * 8 bytes) + the category context crossing.
    embedding_bytes = 6 * 8
    assert payload.total_bytes >= embedding_bytes
    assert payload.pii_bytes == 0


def test_empty_roi_raises():
    pf = PrivacyFilter(mode="embedding")
    with pytest.raises(ValueError):
        pf.filter(_frame(), bbox=(100, 100, 10, 10))  # outside the frame


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        PrivacyFilter(mode="raw")


def test_red_team_leak_attempts_are_all_blocked():
    """Adversarially try to smuggle out PII, an unknown identifier, and the full frame.
    All must be blocked-and-logged, and measured PII egress stays 0. This is what makes
    'zero PII egress' a caught-leak result, not a tautology."""
    pf = PrivacyFilter(mode="embedding")
    frame = _frame()

    # 1) Known PII field + 2) an unanticipated identifier, alongside an allowed key.
    payload = pf.filter(frame, bbox=(8, 8, 32, 32), context={
        "category": "bottle",          # allowed
        "operator": "alice",           # known PII
        "device_serial": "DS-42",      # unanticipated identifier (unlisted)
    })
    blocked = {c.field for c in payload.crossings if c.blocked}
    assert "context.operator" in blocked
    assert "context.device_serial" in blocked
    assert payload.context == {"category": "bottle"}     # only the allowed key survived
    assert payload.pii_bytes == 0                         # nothing sensitive egressed

    # 3) Attempt to escalate the FULL frame (a raw-image leak) that must be refused.
    with pytest.raises(PrivacyViolation):
        pf_roi = PrivacyFilter(mode="roi")
        pf_roi.filter(frame, bbox=(0, 0, frame.shape[1], frame.shape[0]))


def test_pii_key_set_is_nonempty():
    # Guard against accidentally emptying the PII allowlist in a refactor.
    assert "operator" in PII_CONTEXT_KEYS
