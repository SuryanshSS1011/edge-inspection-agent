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
    # Degraded queues escalations (counted) but still produces decisions for every item.
    assert res.n_items > 0
    assert res.n_escalations > 0


def test_table_renders_all_rows():
    results = run_all(_stream(120), COSTS, seeds=(0, 1))
    table = to_markdown(results)
    assert "Cost-weighted recall" in table
    assert "**Hybrid (ours)**" in table
    assert table.count("\n") >= 6  # header + sep + 5 rows
