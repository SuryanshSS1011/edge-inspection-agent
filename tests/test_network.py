"""M6 tests for the network-tier controller: probe transitions and forced override."""

from edge.network import NetworkController
from edge.router import NetworkMode


def test_probe_full_when_reachable_and_fast():
    nc = NetworkController(probe_fn=lambda: (True, 50.0), degraded_latency_ms=500.0)
    assert nc.probe() == NetworkMode.FULL


def test_probe_degraded_when_slow():
    nc = NetworkController(probe_fn=lambda: (True, 900.0), degraded_latency_ms=500.0)
    assert nc.probe() == NetworkMode.DEGRADED


def test_probe_offline_when_unreachable():
    nc = NetworkController(probe_fn=lambda: (False, None))
    assert nc.probe() == NetworkMode.OFFLINE


def test_force_overrides_probe():
    nc = NetworkController(probe_fn=lambda: (True, 10.0))
    nc.force(NetworkMode.OFFLINE)
    assert nc.probe() == NetworkMode.OFFLINE  # probe ignored while forced
    assert nc.forced is True


def test_release_restores_probe():
    nc = NetworkController(probe_fn=lambda: (True, 10.0))
    nc.force(NetworkMode.OFFLINE)
    nc.release()
    assert nc.probe() == NetworkMode.FULL
    assert nc.forced is False


def test_no_probe_keeps_mode():
    nc = NetworkController(mode=NetworkMode.DEGRADED)
    assert nc.probe() == NetworkMode.DEGRADED
