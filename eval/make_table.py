"""Render the results table (build plan §7) as markdown from run_eval output."""

import argparse
import json

ROW_LABELS = {
    "cloud_everything": "Cloud-everything (baseline)",
    "local_only": "Local-only",
    "hybrid_full": "**Hybrid (ours)**",
    "hybrid_degraded": "Hybrid — degraded",
    "hybrid_offline": "Hybrid — offline",
}

ORDER = ["cloud_everything", "local_only", "hybrid_full", "hybrid_degraded", "hybrid_offline"]

HEADER = (
    "| Condition | Cost-weighted recall | p50 / p99 latency (ms) | "
    "Bytes → cloud / item | Cloud cost / 1k | PII bytes out |"
)
SEP = "|---|---|---|---|---|---|"


def _as_dict(result):
    """Accept either a ConditionResult dataclass (in-process) or a plain dict (from JSON)."""
    return result if isinstance(result, dict) else vars(result)


def to_markdown(results: dict) -> str:
    lines = [HEADER, SEP]
    deferred_total = 0
    deferred_cost = 0.0
    for cond in ORDER:
        if cond not in results:
            continue
        r = _as_dict(results[cond]["result"])
        lo, _mean, hi = results[cond]["recall_ci"]
        recall = f"{r['cost_weighted_recall']:.3f} [{lo:.3f}–{hi:.3f}]"
        lat = f"{r['p50_latency_ms']:.1f} / {r['p99_latency_ms']:.1f}"
        bytes_item = f"{r['bytes_to_cloud_per_item']:.0f}"
        cost = f"${r['cloud_cost_per_1k']:.2f}"
        pii = str(r["pii_bytes_out"])
        lines.append(
            f"| {ROW_LABELS[cond]} | {recall} | {lat} | {bytes_item} | {cost} | {pii} |"
        )
        # Accumulate deferred (would-be) escalations whose cost is realized on reconnect.
        n_def = r.get("n_deferred", 0)
        n_items = r.get("n_items", 0) or 1
        deferred_total += n_def
        deferred_cost += (n_def / n_items) * 1000 * _c_cloud(results)

    if deferred_total:
        lines.append(
            f"| Reconnect / sync (drains queue) | — | — | (batched) | "
            f"${deferred_cost:.2f} | 0 |"
        )

    lines.append("")
    lines.append(
        "> Per-mode columns measure egress **under that network condition**. Offline shows "
        "$0 / 0 bytes because the device decides locally with no cloud call; the deferred "
        "diagnoses are drained — and their cost charged once — in the reconnect/sync row, "
        "so no call is counted twice. Deferred diagnoses reconcile the **log**, not the "
        "action: the offline decision was already made by the local conservative policy."
    )
    return "\n".join(lines)


def _c_cloud(results: dict) -> float:
    """Recover the per-call cost from a row that escalated live (cost = n/items*1000*C)."""
    for cond in ("cloud_everything", "hybrid_full"):
        if cond not in results:
            continue
        r = _as_dict(results[cond]["result"])
        n = r.get("n_escalations", 0)
        items = r.get("n_items", 0)
        cost = r.get("cloud_cost_per_1k", 0.0)
        if n and items and cost:
            return cost / (n / items * 1000)
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="eval/results.json")
    parser.add_argument("--out", default="eval/results_table.md")
    args = parser.parse_args()

    results = json.loads(open(args.results).read())
    table = to_markdown(results)
    with open(args.out, "w") as fh:
        fh.write(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
