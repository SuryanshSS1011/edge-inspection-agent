"""Deterministic software switch to force full / degraded / offline for the demo and
eval (build plan §6, §8). A software switch (not a physical pull) keeps the
network-cut beat reliable on video. Lands in M8.
"""

from edge.network import NetworkController
from edge.router import NetworkMode


def set_mode(controller: NetworkController, mode: str) -> None:
    controller.force(NetworkMode(mode))
