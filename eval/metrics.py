"""Evaluation metrics (build plan §7). Definitions live here so the repo states each
metric precisely.

Cost-weighted recall is the headline operational metric. Raw model accuracy treats a
missed defect and a needless rejection as equally bad; operationally they are not. We
score on the asymmetric operator cost:

    incurred_cost = C_FN * false_negatives + C_FP * false_positives

and normalize to a [0, 1] "cost-weighted recall" where higher is better:

    cost_weighted_recall = 1 - incurred_cost / worst_case_cost

worst_case_cost is the cost of getting every item wrong in the most expensive way
(every actual defect missed at C_FN, every actual good rejected at C_FP). A perfect
inspector scores 1.0; one that misses every defect and rejects every good scores 0.0.
"""

import math
from typing import List

from edge.router import Costs


def confusion(y_true: List[int], y_pred: List[int]):
    """Return (tp, fp, tn, fn) treating 1 as 'defect' (the positive/reject class)."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 0:
            tn += 1
        else:  # p == 0 and t == 1 -> a shipped defect
            fn += 1
    return tp, fp, tn, fn


def incurred_cost(y_true: List[int], y_pred: List[int], costs: Costs) -> float:
    _tp, fp, _tn, fn = confusion(y_true, y_pred)
    return costs.C_FN * fn + costs.C_FP * fp


def worst_case_cost(y_true: List[int], costs: Costs) -> float:
    n_pos = sum(1 for t in y_true if t == 1)   # actual defects -> worst is missing all
    n_neg = sum(1 for t in y_true if t == 0)   # actual goods   -> worst is rejecting all
    return costs.C_FN * n_pos + costs.C_FP * n_neg


def cost_weighted_recall(y_true: List[int], y_pred: List[int], costs: Costs) -> float:
    """1 - incurred/worst-case operator cost. Higher is better; perfect = 1.0."""
    worst = worst_case_cost(y_true, costs)
    if worst == 0:
        return 1.0
    return 1.0 - incurred_cost(y_true, y_pred, costs) / worst


def latency_percentiles(latencies_ms: List[float], pcts=(50, 99)) -> dict:
    """Nearest-rank percentiles of per-decision latency."""
    if not latencies_ms:
        return {f"p{int(p)}": 0.0 for p in pcts}
    ordered = sorted(latencies_ms)
    out = {}
    for p in pcts:
        # nearest-rank: ceil(p/100 * n), 1-indexed
        rank = max(1, math.ceil((p / 100.0) * len(ordered)))
        out[f"p{int(p)}"] = ordered[rank - 1]
    return out


def bytes_to_cloud_per_item(total_bytes: int, n_items: int) -> float:
    return total_bytes / n_items if n_items else 0.0


def cloud_cost_per_1k(n_escalations: int, n_items: int, c_cloud: float) -> float:
    return (n_escalations / n_items * 1000 * c_cloud) if n_items else 0.0


def pii_bytes_out(boundary_log_rows: List[dict]) -> int:
    """PII bytes that actually left the device. Target: 0.

    A crossing flagged is_pii but blocked means the filter *caught* it. It did not
    egress, so it must not count toward leaked PII.
    """
    return sum(
        r["nbytes"]
        for r in boundary_log_rows
        if r.get("is_pii") and not r.get("blocked")
    )


def bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for the mean of `values`. Returns (lo, mean, hi).

    Used to report spread across seeds/runs (build plan §7 methodology).
    """
    import numpy as np  # type: ignore

    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)
    ])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, float(arr.mean()), hi)
