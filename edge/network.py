"""Network-tier controller: full / degraded / offline (§3.5).

The mode is switchable at runtime (health probe or forced toggle for the demo)
and injected into the router — never read ad hoc. Lands in M6.
"""

from edge.router import NetworkMode


class NetworkController:
    def __init__(self, mode: NetworkMode = NetworkMode.FULL):
        self._mode = mode
        self._forced = False

    @property
    def mode(self) -> NetworkMode:
        return self._mode

    def force(self, mode: NetworkMode) -> None:
        """Demo/eval override: pin a mode regardless of the live health probe."""
        self._mode = mode
        self._forced = True

    def probe(self) -> NetworkMode:  # M6
        """Measure link health (reachability + latency) and update mode unless forced.

        full: cloud reachable & latency ok. degraded: weak/intermittent.
        offline: unreachable.
        """
        if self._forced:
            return self._mode
        raise NotImplementedError
