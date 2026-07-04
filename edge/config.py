"""Typed configuration loaded from config.yaml (build plan §6).

Costs feed directly into the router; thresholds are derived, never hand-set.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from edge.dotenv import load_dotenv
from edge.drift import DriftConfig
from edge.router import Costs, NetworkMode


@dataclass(frozen=True)
class Paths:
    db: str
    model: str
    calibration: str


@dataclass(frozen=True)
class Config:
    costs: Costs
    default_mode: NetworkMode
    paths: Paths
    drift: DriftConfig = DriftConfig()


def load_config(path: str = "config.yaml") -> Config:
    load_dotenv()  # populate os.environ from .env if present (no-op when absent)
    raw = yaml.safe_load(Path(path).read_text())
    c = raw["costs"]
    costs = Costs(
        C_FN=float(c["C_FN"]),
        C_FP=float(c["C_FP"]),
        C_cloud=float(c["C_cloud"]),
        residual_cloud_error=float(c.get("residual_cloud_error", 0.0)),
    )
    paths = Paths(**raw["paths"])
    mode = NetworkMode(raw["network"]["default_mode"])
    d = raw.get("drift", {})
    drift = DriftConfig(
        window=int(d.get("window", 200)),
        check_every=int(d.get("check_every", 50)),
        ks_threshold=float(d.get("ks_threshold", 0.15)),
    )
    return Config(costs=costs, default_mode=mode, paths=paths, drift=drift)
