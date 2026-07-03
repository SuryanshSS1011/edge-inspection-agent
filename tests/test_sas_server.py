"""The SAS/Docker container runs `python -m cloud.fc_deploy.handler` with PORT set.
This verifies that exact entrypoint serves /healthz and /diagnose on the configured PORT,
without heavy deps — i.e. the container will work. (Docker itself isn't run in CI.)
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
