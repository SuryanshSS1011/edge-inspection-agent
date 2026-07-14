"""Alibaba Function Compute HTTP handler wrapping the diagnose_defect tool.

FC custom runtime invokes a long-running HTTP server (see bootstrap). This exposes:
    GET  /healthz            -> {"ok": true}
    POST /diagnose           -> {defect_present, defect_type, confidence, root_cause,
                                 recommended_action}
        body: {"roi_png_b64": "...", "embedding": [...], "context": {...}}

The edge `cloud_client` calls POST /diagnose. Keeping HTTP here (rather than MCP stdio)
matches FC's request model; mcp_server.py remains the MCP-native interface.

CORS is enabled so the project site can call /diagnose from a browser. Restrict the allowed
origin in production by setting CORS_ALLOW_ORIGIN (defaults to "*").

Reads DASHSCOPE_API_KEY / QWEN_MODEL / QWEN_BASE_URL from the FC environment.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# FC custom runtime sets PYTHONPATH to the code root; import the package tool.
from cloud.mcp_server import diagnose_defect

# Which browser origins may call this server. "*" allows any (fine for a public demo);
# set to your Pages origin (e.g. https://user.github.io) to lock it down.
CORS_ALLOW_ORIGIN = os.environ.get("CORS_ALLOW_ORIGIN", "*")


class Handler(BaseHTTPRequestHandler):
    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", CORS_ALLOW_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        # CORS preflight: the browser sends this before a cross-origin POST.
        self.send_response(204)
        self._cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path.rstrip("/") == "/healthz":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/diagnose":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            result = diagnose_defect(
                roi_png_b64=req.get("roi_png_b64", ""),
                embedding=req.get("embedding"),
                context=req.get("context"),
            )
            self._send(200, result)
        except ValueError as exc:
            self._send(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - surface upstream/model failures as 502
            self._send(502, {"error": f"diagnosis failed: {exc}"})

    def log_message(self, *args):  # quiet the default stderr access log
        pass


def main() -> None:
    # FC sets FC_SERVER_PORT; Docker/SAS conventionally use PORT. Honor both, default 9000.
    port = int(os.environ.get("FC_SERVER_PORT") or os.environ.get("PORT") or "9000")
    print(f"[cloud] diagnose_defect HTTP server listening on 0.0.0.0:{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
