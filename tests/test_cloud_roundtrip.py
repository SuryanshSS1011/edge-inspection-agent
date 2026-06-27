"""Integration test: edge CloudClient -> FC HTTP handler -> (mocked) Qwen call.

Proves the full edge->cloud round-trip and the unreachable fallback, without any API
key or real network egress.
"""

import base64
import threading
from http.server import ThreadingHTTPServer

import pytest

import cloud.fc_deploy.handler as handler
import cloud.mcp_server as srv
from edge.cloud_client import CloudClient, CloudUnreachable

GOOD = {
    "defect_present": True,
    "defect_type": "crack",
    "confidence": 0.9,
    "root_cause": "thermal stress",
    "recommended_action": "reject",
}


@pytest.fixture()
def live_server(monkeypatch):
    # Patch the Qwen call so the handler runs without an API key.
    monkeypatch.setattr(srv, "diagnose", lambda roi, emb, ctx: dict(GOOD))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler.Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()


def test_healthz(live_server):
    assert CloudClient(live_server).healthz() is True


def test_diagnose_roundtrip(live_server):
    client = CloudClient(live_server)
    out = client.diagnose(
        roi_png_b64=base64.b64encode(b"PNG").decode(),
        context={"category": "bottle"},
    )
    assert out == GOOD


def test_unreachable_raises():
    # Nothing is listening on this port.
    client = CloudClient("http://127.0.0.1:1", timeout_s=0.5)
    assert client.healthz() is False
    with pytest.raises(CloudUnreachable):
        client.diagnose(roi_png_b64=base64.b64encode(b"PNG").decode())
