"""Cost-aware escalation router, the core IP of the system.

Decides, per inspected item, whether to act locally now or escalate to the cloud.
The escalation threshold is derived from *asymmetric operator cost*, not raw model
confidence. See docs/router_derivation.md for the full derivation.

Cost model (per item, binary defect/no-defect):
    cost_local(p) = min( C_FP * (1 - p),    # cost of REJECT when part was good
                         C_FN * p )          # cost of ACCEPT when part was defective

The two branches cross at the cost-optimal decision boundary:
    p* = C_FP / (C_FP + C_FN)

Escalate iff the local decision is risky enough that the cloud's better accuracy
saves more than the call costs:
    escalate  <=>  cost_local(p) - residual_cloud_error > C_cloud

cost_local peaks at p* and falls off on either side, so the escalation region is an
uncertainty band [p_lo, p_hi] straddling p*. Solving each linear branch for the
threshold T = C_cloud + residual_cloud_error:
    p_lo = T / C_FN          (rising branch, cost_local = C_FN * p)
    p_hi = 1 - T / C_FP      (falling branch, cost_local = C_FP * (1 - p))

If T exceeds the band's peak (cloud too expensive to ever be worth it), the band is
empty and nothing escalates.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class Decision(str, Enum):
    ESCALATE = "ESCALATE"          # full mode, in band: ask the cloud
    DEFER_AND_ACT = "DEFER_AND_ACT"  # degraded mode, in band: queue + act locally now
    LOCAL_ACT = "LOCAL_ACT"        # offline or outside band: act locally


class Action(str, Enum):
    PASS = "PASS"
    REJECT = "REJECT"


class NetworkMode(str, Enum):
    FULL = "full"
    DEGRADED = "degraded"
    OFFLINE = "offline"


@dataclass(frozen=True)
class Costs:
    C_FN: float                  # false negative (defect shipped), large cost
    C_FP: float                  # false positive (good part rejected), small cost
    C_cloud: float               # cost of one escalation
    residual_cloud_error: float = 0.0  # eps: residual error even after a cloud call

    def __post_init__(self) -> None:
        if self.C_FN <= 0 or self.C_FP <= 0:
            raise ValueError("C_FN and C_FP must be positive")
        if self.C_cloud < 0 or self.residual_cloud_error < 0:
            raise ValueError("C_cloud and residual_cloud_error must be non-negative")


def p_star(costs: Costs) -> float:
    """Cost-optimal local decision boundary. Low because C_FN >> C_FP."""
    return costs.C_FP / (costs.C_FP + costs.C_FN)


def _peak_cost(costs: Costs) -> float:
    """Height of cost_local at p*, the max value the band's threshold can clear."""
    return (costs.C_FP * costs.C_FN) / (costs.C_FP + costs.C_FN)


def escalation_band(costs: Costs) -> Optional[Tuple[float, float]]:
    """Return (p_lo, p_hi) for the escalation band, or None if the band is empty.

    Empty band means escalation is never cost-justified (cloud too expensive
    relative to the worst-case local cost). Callers must treat None as 'never
    escalate', NOT as a clamped-but-inverted range.
    """
    threshold = costs.C_cloud + costs.residual_cloud_error

    # If the threshold clears the peak local cost, no p makes escalation worth it.
    if threshold >= _peak_cost(costs):
        return None

    p_lo = threshold / costs.C_FN          # rising branch root
    p_hi = 1.0 - threshold / costs.C_FP    # falling branch root

    p_lo = max(0.0, p_lo)
    p_hi = min(1.0, p_hi)

    if p_lo > p_hi:                        # degenerate -> empty band
        return None
    return p_lo, p_hi


def local_action(p: float, costs: Costs) -> Action:
    """Cost-minimizing local choice. Because C_FN >> C_FP, ties break toward REJECT."""
    return Action.REJECT if p >= p_star(costs) else Action.PASS


def decide(p: float, network_mode: NetworkMode, costs: Costs) -> Decision:
    """Route an item given its calibrated defect probability and the network mode.

    - full + in band     -> ESCALATE (ask the cloud)
    - degraded + in band -> DEFER_AND_ACT (queue to outbox, act locally now, reconcile later)
    - offline, or out of band, or empty band -> LOCAL_ACT
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be a probability in [0, 1]")

    band = escalation_band(costs)
    in_band = band is not None and band[0] <= p <= band[1]

    if in_band and network_mode == NetworkMode.FULL:
        return Decision.ESCALATE
    if in_band and network_mode == NetworkMode.DEGRADED:
        return Decision.DEFER_AND_ACT
    return Decision.LOCAL_ACT
