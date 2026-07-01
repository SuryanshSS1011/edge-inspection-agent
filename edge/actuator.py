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


class UsbRelayActuator(Actuator):
    """Drives a USB relay over serial (pyserial). REJECT energizes the relay (reject
    light / diverter / lock); PASS releases it. Defaults match the common LC-relay byte
    protocol; override on_cmd/off_cmd for a different board. Serial port opens lazily so
    constructing the object never touches hardware."""

    # Common cheap USB relay board (e.g. LCUS-1): ON = A0 01 01 A2, OFF = A0 01 00 A1.
    DEFAULT_ON = bytes([0xA0, 0x01, 0x01, 0xA2])
    DEFAULT_OFF = bytes([0xA0, 0x01, 0x00, 0xA1])

    def __init__(self, port: str, baud: int = 9600, on_cmd: bytes = None, off_cmd: bytes = None):
        self.port = port
        self.baud = baud
        self.on_cmd = on_cmd or self.DEFAULT_ON
        self.off_cmd = off_cmd or self.DEFAULT_OFF
        self._serial = None

    def _ensure_open(self):
        if self._serial is None:
            import serial  # lazy: only needed with a real relay

            self._serial = serial.Serial(self.port, self.baud, timeout=1)
        return self._serial

    def fire(self, action: Action) -> str:
        ser = self._ensure_open()
        cmd = self.on_cmd if action == Action.REJECT else self.off_cmd
        ser.write(cmd)
        ser.flush()
        return f"relay:{action.value}"

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None


class MockActuator(Actuator):
    """Records actions instead of touching hardware — lets all software land before
    the relay arrives (top-risk mitigation, build plan §12)."""

    def __init__(self):
        self.history = []

    def fire(self, action: Action) -> str:
        self.history.append(action)
        return f"mock:{action.value}"
