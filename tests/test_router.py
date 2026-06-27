"""Unit tests for the cost-aware router — the priority test target.

These pin the band math against hand-worked cost values so the core IP cannot
silently regress.
"""

import math

import pytest

from edge.router import (
    Action,
    Costs,
    Decision,
    NetworkMode,
    decide,
    escalation_band,
    local_action,
    p_star,
)

# Default config costs (mirror config.yaml).
DEFAULT = Costs(C_FN=100.0, C_FP=5.0, C_cloud=2.0, residual_cloud_error=0.3)


def test_p_star_hand_worked():
    # p* = C_FP / (C_FP + C_FN) = 5 / 105
    assert p_star(DEFAULT) == pytest.approx(5.0 / 105.0)
    # Sits low because misses are expensive.
    assert p_star(DEFAULT) < 0.05


def test_band_hand_worked():
    # threshold T = C_cloud + eps = 2.3
    # p_lo = T / C_FN = 2.3 / 100 = 0.023
    # p_hi = 1 - T / C_FP = 1 - 2.3/5 = 0.54
    band = escalation_band(DEFAULT)
    assert band is not None
    p_lo, p_hi = band
    assert p_lo == pytest.approx(0.023)
    assert p_hi == pytest.approx(0.54)
    # Band straddles p*.
    assert p_lo <= p_star(DEFAULT) <= p_hi


def test_band_empty_when_cloud_too_expensive():
    # Peak local cost = C_FP*C_FN/(C_FP+C_FN) = 500/105 ≈ 4.76.
    # Set C_cloud above that -> band must be empty (never escalate).
    costs = Costs(C_FN=100.0, C_FP=5.0, C_cloud=10.0, residual_cloud_error=0.0)
    assert escalation_band(costs) is None


def test_band_widens_as_cloud_gets_cheaper():
    cheap = Costs(C_FN=100.0, C_FP=5.0, C_cloud=0.5, residual_cloud_error=0.0)
    pricey = Costs(C_FN=100.0, C_FP=5.0, C_cloud=4.0, residual_cloud_error=0.0)
    lo_c, hi_c = escalation_band(cheap)
    lo_p, hi_p = escalation_band(pricey)
    # Cheaper cloud -> wider band on both sides.
    assert (hi_c - lo_c) > (hi_p - lo_p)
    assert lo_c <= lo_p
    assert hi_c >= hi_p


def test_band_clamps_to_unit_interval():
    # Tiny C_cloud could push p_lo toward 0 and p_hi toward 1; must stay in [0,1].
    costs = Costs(C_FN=100.0, C_FP=5.0, C_cloud=0.0, residual_cloud_error=0.0)
    lo, hi = escalation_band(costs)
    assert 0.0 <= lo <= hi <= 1.0


def test_full_mode_escalates_in_band():
    # p = 0.3 is inside [0.023, 0.54].
    assert decide(0.3, NetworkMode.FULL, DEFAULT) == Decision.ESCALATE


def test_full_mode_local_outside_band():
    # p = 0.9 is above p_hi=0.54 -> confident defect, no need to ask.
    assert decide(0.9, NetworkMode.FULL, DEFAULT) == Decision.LOCAL_ACT
    # p = 0.001 is below p_lo=0.023 -> confident good.
    assert decide(0.001, NetworkMode.FULL, DEFAULT) == Decision.LOCAL_ACT


def test_degraded_never_escalates_but_still_acts():
    # In band under degraded: defer + act, never a live escalation.
    d = decide(0.3, NetworkMode.DEGRADED, DEFAULT)
    assert d == Decision.DEFER_AND_ACT
    assert d != Decision.ESCALATE


def test_offline_in_band_rejects_conservatively():
    # The graceful-degradation safety property: offline + in band -> LOCAL_ACT,
    # and because p (0.3) >= p* (0.048), the local action is REJECT.
    d = decide(0.3, NetworkMode.OFFLINE, DEFAULT)
    assert d == Decision.LOCAL_ACT
    assert local_action(0.3, DEFAULT) == Action.REJECT


def test_local_action_boundary():
    ps = p_star(DEFAULT)
    # Exactly at p*: ties break toward REJECT.
    assert local_action(ps, DEFAULT) == Action.REJECT
    # Just below: PASS.
    assert local_action(ps - 1e-6, DEFAULT) == Action.PASS


def test_local_action_extremes():
    assert local_action(0.0, DEFAULT) == Action.PASS
    assert local_action(1.0, DEFAULT) == Action.REJECT


def test_decide_rejects_invalid_probability():
    with pytest.raises(ValueError):
        decide(1.5, NetworkMode.FULL, DEFAULT)
    with pytest.raises(ValueError):
        decide(-0.1, NetworkMode.FULL, DEFAULT)


def test_costs_validation():
    with pytest.raises(ValueError):
        Costs(C_FN=-1.0, C_FP=5.0, C_cloud=2.0)
    with pytest.raises(ValueError):
        Costs(C_FN=100.0, C_FP=5.0, C_cloud=-1.0)


def test_empty_band_means_never_escalate_any_mode():
    costs = Costs(C_FN=100.0, C_FP=5.0, C_cloud=10.0, residual_cloud_error=0.0)
    for p in (0.01, 0.3, 0.5, 0.9):
        assert decide(p, NetworkMode.FULL, costs) == Decision.LOCAL_ACT
        assert decide(p, NetworkMode.DEGRADED, costs) == Decision.LOCAL_ACT
