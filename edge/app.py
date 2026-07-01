"""Entrypoint / CLI: live | replay | demo.

  python -m edge.app live   --config config.yaml [--relay-port /dev/ttyUSB0] [--cloud-url URL]
  python -m edge.app replay --config config.yaml --category bottle --data /path/to/mvtec  # M7
  python -m edge.app demo   --config config.yaml                                          # M8

`live` wires the real components (webcam -> ONNX -> router -> relay -> SQLite log). If no
relay port is given it uses a mock actuator so the loop still runs and logs; if no cloud
URL is given, in-band items act locally (no escalation) — actuation never blocks on cloud.
"""

import argparse
import os

from edge.actuator import MockActuator, UsbRelayActuator
from edge.calibration import load as load_temperature
from edge.cloud_client import CloudClient
from edge.config import load_config
from edge.frame_source import WebcamSource
from edge.network import NetworkController
from edge.orchestrator import Orchestrator
from edge.outbox import Outbox
from edge.perception import OnnxClassifier
from edge.privacy import PrivacyFilter
from edge.store import Store


def build_live(args, config) -> Orchestrator:
    try:
        temperature = load_temperature(config.paths.calibration)
    except (FileNotFoundError, ValueError):
        temperature = 1.0  # uncalibrated until the calibration fit has been run

    # CLI flags win; otherwise fall back to .env (EDGE_RELAY_PORT / EDGE_CLOUD_URL).
    relay_port = args.relay_port or os.environ.get("EDGE_RELAY_PORT") or None
    cloud_url = args.cloud_url or os.environ.get("EDGE_CLOUD_URL") or None

    actuator = UsbRelayActuator(relay_port) if relay_port else MockActuator()
    cloud = CloudClient(cloud_url) if cloud_url else None
    privacy = PrivacyFilter() if cloud_url else None
    store = Store(config.paths.db)

    probe_fn = (lambda: (cloud.healthz(), 0.0)) if cloud else None
    return Orchestrator(
        config=config,
        source=WebcamSource(args.camera),
        perception=OnnxClassifier(config.paths.model, temperature=temperature),
        actuator=actuator,
        store=store,
        network=NetworkController(config.default_mode, probe_fn=probe_fn),
        cloud=cloud,
        privacy=privacy,
        outbox=Outbox(store),
        category=args.category,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="edge-inspection-agent")
    parser.add_argument("mode", choices=["live", "replay", "demo"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--data", help="MVTec root (replay mode)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--relay-port", help="serial port for the USB relay; mock if omitted")
    parser.add_argument("--cloud-url", help="deployed cloud reasoning endpoint")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[edge] mode={args.mode} costs={config.costs} network={config.default_mode.value}")

    if args.mode == "live":
        orch = build_live(args, config)
        events = orch.run()
        print(f"[edge] processed {len(events)} items")
        return
    raise SystemExit(f"{args.mode} mode lands in a later milestone")


if __name__ == "__main__":
    main()
