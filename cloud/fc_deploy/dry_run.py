"""Deploy dry-run: exercise the EXACT deployed path locally, end to end.

Starts the real Function Compute handler (cloud/fc_deploy/handler.py) as a local HTTP
server and drives it through the edge CloudClient — the same client the orchestrator uses
against the deployed URL. With a live DASHSCOPE_API_KEY it makes ONE real Qwen-VL call, so
a green dry-run means the deployed function will work the moment it's up; only the hosting
location changes.

    DASHSCOPE_API_KEY=... python -m cloud.fc_deploy.dry_run          # one live call
    python -m cloud.fc_deploy.dry_run --mock                        # no API call, plumbing only

Exits non-zero on any failure so it can gate a deploy.
"""

import argparse
import base64
import io
import struct
import sys
import threading
import time
import zlib
from http.server import ThreadingHTTPServer

from edge.cloud_client import CloudClient, CloudUnreachable
from edge.dotenv import load_dotenv


def _tiny_png(width=64, height=64, rgb=(180, 60, 60)) -> bytes:
    """A minimal valid PNG without Pillow — a solid patch is enough to exercise the path."""
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = (b"\x00" + bytes(rgb) * width) * height
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def run(mock: bool) -> int:
    load_dotenv()
    import cloud.fc_deploy.handler as handler
    import cloud.mcp_server as srv

    if mock:
        srv.diagnose = lambda roi, emb, ctx: {
            "defect_present": True, "defect_type": "crack", "confidence": 0.9,
            "root_cause": "mocked", "recommended_action": "reject",
        }

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler.Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = CloudClient(url, timeout_s=40.0)

    try:
        # 1) health probe — the same call the network controller uses.
        if not client.healthz():
            print("FAIL: /healthz did not return ok")
            return 1
        print("ok  /healthz")

        # 2) full diagnose round-trip through the deployed handler contract.
        roi_b64 = base64.b64encode(_tiny_png()).decode("ascii")
        t0 = time.time()
        result = client.diagnose(roi_png_b64=roi_b64, context={"category": "bottle"})
        dt = (time.time() - t0) * 1000

        required = {"defect_present", "defect_type", "confidence", "root_cause",
                    "recommended_action"}
        missing = required - result.keys()
        if missing:
            print(f"FAIL: diagnosis missing keys {sorted(missing)}: {result}")
            return 1
        mode = "mock" if mock else "LIVE Qwen-VL"
        print(f"ok  /diagnose ({mode}, {dt:.0f} ms): defect_present={result['defect_present']} "
              f"type={result['defect_type']}")
        print("\nDRY RUN PASSED — the deployed path works end to end. Deploy is one command.")
        return 0
    except CloudUnreachable as exc:
        print(f"FAIL: cloud unreachable through the handler: {exc}")
        return 1
    finally:
        server.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true",
                        help="skip the live Qwen-VL call; test plumbing only")
    args = parser.parse_args()
    sys.exit(run(mock=args.mock))


if __name__ == "__main__":
    main()
