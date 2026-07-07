"""Entrypoint / CLI: camera | data | demo.

  python -m edge.app camera [--camera auto|0|URL] [--relay-port PORT] [--cloud-url URL]
  python -m edge.app data   [--category bottle] [--data DIR] [--limit N]
  python -m edge.app demo

Two of these run the *same real pipeline* (camera/ONNX/router/privacy/cloud/relay/log) and
differ only in where frames come from:

  * camera runs a real capture device (laptop, USB, or a phone streaming over wifi). If no
    device opens it falls back to replaying data/<category>/ so a live run never dies on a
    missing camera, and the source it landed on is logged.
  * data replays images from a folder through the real pipeline (no camera). Useful for
    development and reproducible runs. --limit N bounds the number of frames.

`demo` is different. It is a self-contained, deterministic scripted walkthrough (a modeled
cloud and staged probabilities) that narrates the 6-beat story for a video/presentation. It
does not touch a camera, the real model, or the deployed cloud.

--camera accepts:
  * an int index   (0 = built-in laptop cam, 1+ = USB / Continuity / OBS virtual cam)
  * "auto"         (scan indices 0..3, use the first that opens)
  * a stream URL   (a phone over wifi, e.g. an Android "IP Webcam" feed:
                    http://<phone-ip>:8080/video, or any rtsp:// / mjpeg URL)

If no relay port is given it uses a mock actuator so the loop still runs and logs. If no
cloud URL is given, in-band items act locally (no escalation) and actuation never blocks on
the cloud.
"""

import argparse
import logging
import os
from pathlib import Path

from edge.actuator import MockActuator, UsbRelayActuator
from edge.calibration import load as load_temperature
from edge.cloud_client import CloudClient
from edge.config import load_config
from edge.frame_source import FallbackSource, FileSource, WebcamSource
from edge.network import NetworkController
from edge.orchestrator import Orchestrator
from edge.outbox import Outbox
from edge.perception import OnnxClassifier
from edge.privacy import PrivacyFilter
from edge.store import Store

_log = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# --- frame sources -----------------------------------------------------------

def _parse_camera(value: str):
    """--camera is a string on the CLI; turn "0"/"1" into ints, keep "auto" and URLs as-is."""
    if value is None:
        return 0
    return int(value) if value.isdigit() else value


def _image_paths(data_root: str, category: str, limit=None) -> list:
    """Images under data/<category>/ to replay. Prefers a proper MVTec test split; otherwise
    globs whatever images exist (incl. ground_truth masks). Truncated to `limit` if set."""
    cat_dir = Path(data_root) / category
    test_dir = cat_dir / "test"
    search = test_dir if test_dir.is_dir() else cat_dir
    paths = sorted(str(p) for p in search.rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    return paths[:limit] if limit else paths


def _camera_source(args):
    """A camera source that falls back to file replay if the device can't be opened."""
    webcam = WebcamSource(_parse_camera(args.camera))
    fallback_paths = _image_paths(args.data or "data", args.category)
    if not fallback_paths:
        _log.warning(
            "no fallback images under data/%s, so a missing camera will end the run",
            args.category,
        )
        return webcam
    _log.info("camera mode: device=%r (falls back to %d data/%s images)",
              args.camera, len(fallback_paths), args.category)
    return FallbackSource(webcam, FileSource(fallback_paths))


def _data_source(args):
    """A files-only source that runs the real pipeline over a folder of images, no camera."""
    paths = _image_paths(args.data or "data", args.category, limit=args.limit)
    if not paths:
        raise SystemExit(
            f"data mode: no images under {args.data or 'data'}/{args.category} "
            f"(expected .png/.jpg files)"
        )
    _log.info("data mode: replaying %d images from data/%s%s",
              len(paths), args.category, f" (limit {args.limit})" if args.limit else "")
    return FileSource(paths)


# --- pipeline ----------------------------------------------------------------

def build_orchestrator(args, config, source) -> Orchestrator:
    """Wire the real pipeline around a given frame source. camera and data modes share
    this and differ only in `source`."""
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

    def _probe():
        import time
        t0 = time.time()
        reachable = cloud.healthz()
        latency_ms = (time.time() - t0) * 1000.0
        return reachable, latency_ms

    probe_fn = _probe if cloud else None
    return Orchestrator(
        config=config,
        source=source,
        perception=OnnxClassifier(config.paths.model, temperature=temperature),
        actuator=actuator,
        store=store,
        network=NetworkController(config.default_mode, probe_fn=probe_fn),
        cloud=cloud,
        privacy=privacy,
        outbox=Outbox(store),
        category=args.category,
    )


# --- narration ---------------------------------------------------------------

def _narrate_event(i: int, e) -> None:
    """One concise line per inspected frame, driven off the logged InspectionEvent."""
    tag = "ESCALATE" if e.escalated else e.decision
    cloud = ""
    if e.cloud_diagnosis is not None:
        cloud = f" cloud={'defect' if e.cloud_diagnosis.get('defect_present') else 'clean'}"
    defer = f" outbox={e.outbox_state}" if e.outbox_state != "none" else ""
    print(
        f"[{i:3d}] p={e.p:.3f} {tag:<8}{cloud} "
        f"bytes={e.bytes_to_cloud} pii={e.pii_bytes} net={e.network_mode}{defer}",
        flush=True,
    )


def _run_pipeline(orch: Orchestrator, mode: str) -> None:
    """Drive the source frame-by-frame, narrating each event, then print an audit summary."""
    print(f"[edge] mode={mode} inspecting...", flush=True)
    events = []
    for frame in orch.source.frames():
        orch.network.probe()
        e = orch.process_frame(frame)
        events.append(e)
        _narrate_event(len(events), e)

    reconciled = 0
    if orch.outbox is not None and orch.cloud is not None and orch.outbox.pending_count():
        reconciled = orch.reconcile()

    total_bytes = sum(e.bytes_to_cloud for e in events)
    total_pii = sum(e.pii_bytes for e in events)
    escalated = sum(1 for e in events if e.escalated)
    print(
        f"summary: {len(events)} items | {escalated} escalated | {total_bytes} bytes to cloud "
        f"| PII {total_pii} (target 0) | reconciled {reconciled}",
        flush=True,
    )


# --- CLI ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(prog="edge-inspection-agent")
    parser.add_argument("mode", choices=["camera", "data", "demo"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--data", help="image root (default: data/) used by data mode and "
                                       "as the camera-mode fallback")
    parser.add_argument("--limit", type=int, help="data mode: max frames to replay")
    parser.add_argument(
        "--camera", default="0",
        help="camera source: int index (0=laptop), 'auto', or a stream URL "
             "(e.g. a phone: http://<phone-ip>:8080/video)",
    )
    parser.add_argument("--relay-port", help="serial port for the USB relay; mock if omitted")
    parser.add_argument("--cloud-url", help="deployed cloud reasoning endpoint")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(args.config)
    print(f"[edge] mode={args.mode} costs={config.costs} network={config.default_mode.value}")

    if args.mode == "demo":
        # The self-contained scripted walkthrough (deterministic, no hardware/cloud).
        from demo.demo_runner import run_demo
        run_demo(args.config)
        return

    source = _camera_source(args) if args.mode == "camera" else _data_source(args)
    orch = build_orchestrator(args, config, source)
    _run_pipeline(orch, args.mode)


if __name__ == "__main__":
    main()
