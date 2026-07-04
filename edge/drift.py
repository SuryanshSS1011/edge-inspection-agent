"""Drift detectors for calibration drift and edge-vs-cloud model disagreement.

CalibrationDriftDetector (DriftDetector):
  Watches the rolling window of edge confidence values with a KS test. Fires
  DriftState.ALERT when the distribution shifts from the calibration-time reference,
  signalling covariate shift (lighting, lens, vibration). Orchestrator switches to
  conservative mode on alert.

ModelDriftMonitor:
  Watches systematic disagreement between local edge decisions and cloud diagnoses on
  escalated frames. When the disagreement rate in a rolling window exceeds a threshold
  it logs a warning: the edge model and VLM have drifted apart and recalibration or
  fine-tuning is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
from scipy.stats import ks_2samp


class DriftState(Enum):
    OK = "ok"
    ALERT = "alert"


@dataclass(frozen=True)
class DriftConfig:
    window: int = 200       # number of recent predictions to test against reference
    check_every: int = 50   # run KS test every N new predictions
    ks_threshold: float = 0.15  # KS statistic above this triggers alert


class DriftDetector:
    """Stateful detector: call record() after every prediction, check() periodically."""

    def __init__(self, reference: Sequence[float], cfg: DriftConfig | None = None):
        self._ref = np.array(reference, dtype=float)
        self._cfg = cfg or DriftConfig()
        self._state = DriftState.OK
        self._since_last_check = 0

    @property
    def state(self) -> DriftState:
        return self._state

    def check(self, recent: Sequence[float]) -> DriftState:
        """Run KS test between reference and recent window. Updates and returns state."""
        if len(recent) < self._cfg.window // 2:
            return self._state   # not enough data yet
        stat, _ = ks_2samp(self._ref, np.array(recent, dtype=float))
        self._state = DriftState.ALERT if stat >= self._cfg.ks_threshold else DriftState.OK
        return self._state

    def record_and_maybe_check(self, p: float, recent: Sequence[float]) -> DriftState:
        """Call after every prediction. Runs KS test every check_every calls."""
        self._since_last_check += 1
        if self._since_last_check >= self._cfg.check_every:
            self._since_last_check = 0
            return self.check(recent)
        return self._state


@dataclass(frozen=True)
class ModelDriftConfig:
    window: int = 100          # escalated frames to include in the disagreement window
    disagreement_threshold: float = 0.20  # fraction above which drift is flagged


class ModelDriftMonitor:
    """Detects systematic edge-vs-cloud disagreement on escalated frames.

    Each time a cloud diagnosis comes back, record() is called with the edge decision
    and the cloud verdict. When the rolling disagreement rate (over the last `window`
    paired decisions) exceeds `disagreement_threshold`, is_drifted returns True.

    This surfaces VLM weight updates or edge model staleness without requiring any
    ground-truth labels: a sustained disagreement pattern is the signal.
    """

    def __init__(self, cfg: ModelDriftConfig | None = None):
        self._cfg = cfg or ModelDriftConfig()
        self._window: list = []  # bool: True = disagreed

    def record(self, edge_defect: bool, cloud_defect: bool) -> None:
        self._window.append(edge_defect != cloud_defect)
        if len(self._window) > self._cfg.window:
            self._window.pop(0)

    @property
    def disagreement_rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def is_drifted(self) -> bool:
        return (
            len(self._window) >= self._cfg.window // 2
            and self.disagreement_rate >= self._cfg.disagreement_threshold
        )

    def load_from_pairs(self, pairs: Sequence[tuple]) -> None:
        """Seed the window from stored (edge_defect, cloud_defect) pairs (e.g. on startup)."""
        self._window = [e != c for e, c in pairs[-self._cfg.window:]]
