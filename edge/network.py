"""Network-tier controller: full / degraded / offline (§3.5).

The mode is switchable at runtime and injected into the router, never read ad hoc.
For the demo and eval, force()/release() pin a mode deterministically. In live use,
probe() measures link health via an injected probe callable and sets the mode.
"""

from typing import Callable, Optional

from edge.router import NetworkMode

# A probe returns (reachable, latency_ms). latency_ms is None when unreachable.
ProbeFn = Callable[[], "tuple"]


class NetworkController:
    def __init__(
        self,
        mode: NetworkMode = NetworkMode.FULL,
        probe_fn: Optional[ProbeFn] = None,
        degraded_latency_ms: float = 500.0,
    ):
        self._mode = mode
        self._forced = False
        self._probe_fn = probe_fn
        self.degraded_latency_ms = degraded_latency_ms

    @property
    def mode(self) -> NetworkMode:
        return self._mode

    @property
    def forced(self) -> bool:
        return self._forced

    def force(self, mode: NetworkMode) -> None:
        """Demo/eval override: pin a mode regardless of the live health probe."""
        self._mode = mode
        self._forced = True

    def release(self) -> None:
        """Stop forcing; the next probe() decides the mode again."""
        self._forced = False

    def probe(self) -> NetworkMode:
        """Measure link health and update mode unless forced.

        full: reachable & latency under the degraded threshold.
        degraded: reachable but slow.
        offline: unreachable.
        """
        if self._forced:
            return self._mode
        if self._probe_fn is None:
            return self._mode  # no probe wired -> keep the current mode
        reachable, latency_ms = self._probe_fn()
        if not reachable:
            self._mode = NetworkMode.OFFLINE
        elif latency_ms is not None and latency_ms > self.degraded_latency_ms:
            self._mode = NetworkMode.DEGRADED
        else:
            self._mode = NetworkMode.FULL
        return self._mode
