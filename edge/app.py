"""Entrypoint / CLI: live | replay | demo.

  python -m edge.app live    --config config.yaml
  python -m edge.app replay  --config config.yaml --category bottle   # eval (M7)
  python -m edge.app demo    --config config.yaml                     # scripted demo (M8)
"""

import argparse

from edge.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="edge-inspection-agent")
    parser.add_argument("mode", choices=["live", "replay", "demo"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--category", default="bottle")
    args = parser.parse_args()

    config = load_config(args.config)
    # Component wiring lands in M4; this validates config loads cleanly today.
    print(f"[edge] mode={args.mode} costs={config.costs} network={config.default_mode.value}")
    raise SystemExit("orchestrator wiring lands in M4")


if __name__ == "__main__":
    main()
