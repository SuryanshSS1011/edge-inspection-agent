"""Actuation: map the final decision to a physical action (§3.6).

Actuation must fire correctly in all three network modes — it never depends on the
cloud. UsbRelayActuator for the real loop (M4), MockActuator for dev/tests.
"""

from abc import ABC, abstractmethod

from edge.router import Action


class Actuator(ABC):
    @abstractmethod
    def fire(self, action: Action) -> str:
        """Execute the physical action; return a label of what fired (for the log)."""
        raise NotImplementedError


class UsbRelayActuator(Actuator):  # M4 — pyserial to a USB relay board
    def __init__(self, port: str, baud: int = 9600):
        self.port = port
        self.baud = baud
        self._serial = None  # open lazily

    def fire(self, action: Action) -> str:
        raise NotImplementedError


class MockActuator(Actuator):
    """Records actions instead of touching hardware — lets all software land before
    the relay arrives (top-risk mitigation, build plan §12)."""

    def __init__(self):
        self.history = []

    def fire(self, action: Action) -> str:
        self.history.append(action)
        return f"mock:{action.value}"
