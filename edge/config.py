"""Typed configuration loaded from config.yaml (build plan §6).

Costs feed directly into the router; thresholds are derived, never hand-set.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from edge.dotenv import load_dotenv
from edge.drift import DriftConfig, ModelDriftConfig
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
    model_drift: ModelDriftConfig = ModelDriftConfig()


class LiveConfig:
    """Thread-safe wrapper that hot-reloads config.yaml when the file changes on disk.

    The orchestrator can call `.get()` each frame-loop iteration; the frozen `Config`
    object is replaced atomically only when the file's mtime changes, so the cost
    parameters (and escalation band) update without a process restart. This closes the
    static-cost reviewer gap: operators can tune C_FN/C_FP for a shift change or a
    new part family just by editing config.yaml.
    """

    def __init__(self, path: str = "config.yaml") -> None:
        self._path = path
        self._mtime: float = 0.0
        self._config: Config = load_config(path)
        self._mtime = Path(path).stat().st_mtime

    def get(self) -> Config:
        try:
            mtime = Path(self._path).stat().st_mtime
        except OSError:
            return self._config
        if mtime != self._mtime:
            try:
                self._config = load_config(self._path)
                self._mtime = mtime
            except Exception:
                pass  # keep previous config on parse error
        return self._config


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
    md = raw.get("model_drift", {})
    model_drift = ModelDriftConfig(
        window=int(md.get("window", 100)),
        disagreement_threshold=float(md.get("disagreement_threshold", 0.20)),
    )
    return Config(costs=costs, default_mode=mode, paths=paths, drift=drift, model_drift=model_drift)
