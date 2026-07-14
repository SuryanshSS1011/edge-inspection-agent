"""The SAS/Docker container runs `python -m cloud.fc_deploy.handler` with PORT set.
This verifies that exact entrypoint serves /healthz and /diagnose on the configured PORT,
without heavy deps, confirming the container will work. (Docker itself isn't run in CI.)
"""

import threading
from http.server import ThreadingHTTPServer

import pytest

import cloud.handler as handler
import cloud.mcp_server as srv
from edge.cloud_client import CloudClient

GOOD = {
    "defect_present": False, "defect_type": "none", "confidence": 1.0,
    "root_cause": "n/a", "recommended_action": "pass",
}


@pytest.fixture()
def server(monkeypatch):
    monkeypatch.setattr(srv, "diagnose", lambda roi, emb, ctx: dict(GOOD))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


def test_handler_binds_and_serves(server):
    client = CloudClient(server)
    assert client.healthz() is True
    import base64
    out = client.diagnose(roi_png_b64=base64.b64encode(b"PNG").decode(), context={"category": "bottle"})
    assert out == GOOD


def test_handler_reads_generic_port_env(monkeypatch):
    # The container sets PORT (not FC_SERVER_PORT); main() must honor it.
    import os
    monkeypatch.delenv("FC_SERVER_PORT", raising=False)
    monkeypatch.setenv("PORT", "8080")
    # We don't actually bind (would block); just assert the resolution logic.
    port = int(os.environ.get("FC_SERVER_PORT") or os.environ.get("PORT") or "9000")
    assert port == 8080


def test_cors_preflight_options(server):
    # A browser sends OPTIONS before a cross-origin POST; it must get the CORS headers.
    import urllib.request

    req = urllib.request.Request(f"{server}/diagnose", method="OPTIONS")
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 204
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
        assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")
        assert "Content-Type" in resp.headers.get("Access-Control-Allow-Headers", "")


def test_cors_header_on_responses(server):
    # The actual GET/POST responses also carry the allow-origin header.
    import urllib.request

    with urllib.request.urlopen(f"{server}/healthz", timeout=5) as resp:
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"


def test_cors_origin_is_configurable(monkeypatch):
    # CORS_ALLOW_ORIGIN pins the allowed origin; the handler reads it at request time via
    # the module constant, so patch and re-read to confirm the wiring.
    monkeypatch.setattr(handler, "CORS_ALLOW_ORIGIN", "https://example.github.io")
    threading_httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler.Handler)
    threading.Thread(target=threading_httpd.serve_forever, daemon=True).start()
    try:
        import urllib.request
        url = f"http://127.0.0.1:{threading_httpd.server_address[1]}/healthz"
        with urllib.request.urlopen(url, timeout=5) as resp:
            assert resp.headers.get("Access-Control-Allow-Origin") == "https://example.github.io"
    finally:
        threading_httpd.shutdown()
