"""Run every operating condition end-to-end against the DB and collect metrics (§7).

Conditions (table rows): cloud-everything, local-only, hybrid (full), hybrid-degraded,
hybrid-offline. Replays MVTec via FileSource through the orchestrator; multiple seeds,
report spread (bootstrap CIs). Lands in M7.
"""

CONDITIONS = [
    "cloud_everything",
    "local_only",
    "hybrid_full",
    "hybrid_degraded",
    "hybrid_offline",
]


def run_all(config_path: str = "config.yaml", seeds=(0, 1, 2)) -> dict:  # M7
    raise NotImplementedError
