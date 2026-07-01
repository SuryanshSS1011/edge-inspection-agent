"""M7 tests: the eval harness produces sensible, internally-consistent rows for each
condition, and the table renders. Uses a synthetic (p, label) stream — no model/cloud.
"""

import numpy as np
import pytest

from edge.router import Costs
from eval.harness import EvalItem, run_condition
from eval.make_table import to_markdown
from eval.run_eval import run_all

COSTS = Costs(C_FN=100.0, C_FP=5.0, C_cloud=2.0, residual_cloud_error=0.3)


def _stream(n=200, seed=0):
    """Defects get high p, goods get low p, with overlap in the ambiguous band."""
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(n):
        label = int(rng.uniform() < 0.5)
        if label:
            p = float(np.clip(rng.normal(0.7, 0.25), 0, 1))
        else:
            p = float(np.clip(rng.normal(0.15, 0.2), 0, 1))
        items.append(EvalItem(p=p, label=label))
    return items


def test_local_only_never_touches_cloud(tmp_path):
    res = run_condition("local_only", _stream(), COSTS, str(tmp_path / "a.db"))
    assert res.n_escalations == 0
    assert res.bytes_to_cloud_per_item == 0
    assert res.pii_bytes_out == 0
    assert res.cloud_cost_per_1k == 0


def test_cloud_everything_escalates_most(tmp_path):
    res = run_condition("cloud_everything", _stream(), COSTS, str(tmp_path / "b.db"))
    # C_cloud≈0 widens the band to cover essentially all probabilities.
    assert res.n_escalations >= 0.9 * res.n_items
    assert res.bytes_to_cloud_per_item > 0


def test_hybrid_escalates_only_the_band(tmp_path):
    cloud_all = run_condition("cloud_everything", _stream(), COSTS, str(tmp_path / "c.db"))
    hybrid = run_condition("hybrid_full", _stream(), COSTS, str(tmp_path / "d.db"))
    # Hybrid escalates strictly fewer items than cloud-everything -> less bandwidth.
    assert hybrid.n_escalations < cloud_all.n_escalations
    assert hybrid.bytes_to_cloud_per_item < cloud_all.bytes_to_cloud_per_item


def test_all_conditions_zero_pii(tmp_path):
    for i, cond in enumerate(
        ["cloud_everything", "local_only", "hybrid_full", "hybrid_degraded", "hybrid_offline"]
    ):
        res = run_condition(cond, _stream(), COSTS, str(tmp_path / f"{i}.db"))
        assert res.pii_bytes_out == 0, f"{cond} leaked PII"


def test_degraded_defers_then_reconciles(tmp_path):
    res = run_condition("hybrid_degraded", _stream(), COSTS, str(tmp_path / "e.db"))
    # Degraded makes a decision for every item, calls the cloud live for none of them
    # (deferred instead), and queues the would-be escalations for the reconnect row.
    assert res.n_items > 0
    assert res.n_escalations == 0     # no live cloud cost/egress while degraded
    assert res.n_deferred > 0         # but the queue is non-empty


def test_offline_has_no_live_cloud_cost(tmp_path):
    res = run_condition("hybrid_offline", _stream(), COSTS, str(tmp_path / "f.db"))
    assert res.bytes_to_cloud_per_item == 0
    assert res.cloud_cost_per_1k == 0   # offline egress is genuinely zero
    assert res.n_deferred > 0           # would-be escalations are queued, not lost


def test_table_renders_all_rows():
    results = run_all(_stream(120), COSTS, seeds=(0, 1))
    table = to_markdown(results)
    assert "Cost-weighted recall" in table
    assert "**Hybrid (ours)**" in table
    assert table.count("\n") >= 6  # header + sep + 5 rows


def _row_cost(table, label):
    import re
    for line in table.splitlines():
        if line.startswith(f"| {label}"):
            m = re.findall(r"\$([0-9]+\.[0-9]+)", line)
            if m:
                return float(m[-1])
    return None


def test_reconnect_cost_never_exceeds_cloud_everything():
    # The anomaly guard: the degradation path (deferred, then drained on reconnect) must
    # not cost MORE than always calling the cloud — that would invert the thesis.
    results = run_all(_stream(400), COSTS, seeds=(0, 1))
    table = to_markdown(results)
    cloud_every = _row_cost(table, "Cloud-everything")
    hybrid = _row_cost(table, "**Hybrid (ours)**")
    reconnect = _row_cost(table, "Reconnect / sync")
    assert reconnect is not None
    assert reconnect <= cloud_every          # never worse than cloud-everything
    assert reconnect <= hybrid + 1e-6        # batched drain <= live hybrid cost


def test_reconnect_counts_one_drain_set_not_two():
    # degraded and offline defer the SAME items; the reconnect row must reflect one drain
    # set, so its cost tracks a single condition's deferred count (with batch discount) —
    # not the sum of degraded + offline (which would ~double it).
    from eval.make_table import BATCH_DISCOUNT, _c_cloud
    results = run_all(_stream(400), COSTS, seeds=(0,))
    n_def = max(results[c]["result"].n_deferred for c in results)
    n_items = results["hybrid_full"]["result"].n_items
    expected = (n_def / n_items) * 1000 * _c_cloud(results) * BATCH_DISCOUNT
    reconnect = _row_cost(to_markdown(results), "Reconnect / sync")
    assert abs(reconnect - expected) < 1.0


def test_reconnect_undiscounted_equals_hybrid_live():
    # The conclusion must not depend on the batching discount: undiscounted, the deferred
    # drain is the SAME band items at the SAME per-call price as hybrid's live escalations,
    # so reconnect(undiscounted) == hybrid live cost. This is what lets us drop the 10%
    # and still keep reconnect <= hybrid <= cloud-everything.
    from eval.make_table import _c_cloud
    results = run_all(_stream(400), COSTS, seeds=(0,))
    n_def = max(results[c]["result"].n_deferred for c in results)
    n_items = results["hybrid_full"]["result"].n_items
    undiscounted = (n_def / n_items) * 1000 * _c_cloud(results)
    hybrid_live = results["hybrid_full"]["result"].cloud_cost_per_1k
    assert abs(undiscounted - hybrid_live) < 1.0
