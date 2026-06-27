"""Evaluation metrics (build plan §7). Definitions live here so the repo states each
metric precisely. Lands in M7.
"""

from typing import List

from edge.router import Costs


def cost_weighted_recall(y_true: List[int], y_pred: List[int], costs: Costs) -> float:  # M7
    """Recall weighted by the asymmetric miss/false-alarm costs — the metric that
    actually matters operationally (penalize false negatives by C_FN, false
    positives by C_FP)."""
    raise NotImplementedError


def latency_percentiles(latencies_ms: List[float], pcts=(50, 99)) -> dict:  # M7
    raise NotImplementedError


def bytes_to_cloud_per_item(total_bytes: int, n_items: int) -> float:
    return total_bytes / n_items if n_items else 0.0


def cloud_cost_per_1k(n_escalations: int, n_items: int, c_cloud: float) -> float:
    return (n_escalations / n_items * 1000 * c_cloud) if n_items else 0.0


def pii_bytes_out(boundary_log_rows: List[dict]) -> int:
    """PII bytes that actually left the device. Target: 0.

    A crossing flagged is_pii but blocked means the filter *caught* it — it did not
    egress, so it must not count toward leaked PII.
    """
    return sum(
        r["nbytes"]
        for r in boundary_log_rows
        if r.get("is_pii") and not r.get("blocked")
    )
