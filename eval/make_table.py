"""Render the results table (build plan §7) as markdown from run_eval output. M7."""

COLUMNS = [
    "Condition",
    "Cost-weighted recall",
    "p50 / p99 latency",
    "Bytes -> cloud / item",
    "Cloud cost / 1k",
    "PII bytes out",
]


def to_markdown(results: dict) -> str:  # M7
    raise NotImplementedError
