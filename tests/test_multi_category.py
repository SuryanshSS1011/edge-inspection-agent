"""Tests for the multi-category robustness table aggregation (no dataset/model needed)."""

from eval.run_multi_category import to_markdown


def _row(cat, local, hybrid, cloud=0.99, n=60):
    return {
        "category": cat,
        "n_eval": n,
        "temperature": 1.0,
        "ece_before": 0.10,
        "ece_after": 0.06,
        "local_only": local,
        "cloud_everything": cloud,
        "hybrid": hybrid,
        "hybrid_ci": (hybrid - 0.005, hybrid, hybrid + 0.005),
    }


def test_table_has_a_row_per_category():
    rows = [_row("bottle", 0.90, 0.98), _row("grid", 0.80, 0.94)]
    table = to_markdown(rows)
    assert "| bottle |" in table
    assert "| grid |" in table


def test_aggregate_reports_mean_and_lift():
    rows = [_row("a", 0.80, 0.96), _row("b", 0.90, 0.96)]
    table = to_markdown(rows)
    # hybrid mean 0.96, local mean 0.85 -> lift +0.110
    assert "Aggregate across 2 categories" in table
    assert "0.960" in table
    assert "+0.110" in table


def test_robustness_framing_present():
    table = to_markdown([_row("bottle", 0.90, 0.98)])
    assert "robustness" in table.lower()
    assert "generalizes" in table.lower()
