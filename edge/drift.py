"""Calibration drift detector (KS test on rolling confidence window).

After calibration the edge model's confidence distribution should be stationary.
Covariate shift (lighting changes, lens dust, camera vibration) causes it to drift,
making the temperature-scaled probabilities stale and breaking the cost inequality.

This module watches the rolling window of predicted probabilities. Every `check_every`
frames it runs a two-sample KS test against the reference distribution captured at
calibration time. If the statistic exceeds `ks_threshold` the detector fires an alert
and returns DriftState.ALERT — the orchestrator switches to conservative mode (treat
every frame as in-band) until the operator recalibrates.
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
