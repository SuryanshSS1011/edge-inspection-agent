"""Integration test: edge CloudClient -> FC HTTP handler -> (mocked) Qwen call.

Proves the full edge->cloud round-trip and the unreachable fallback, without any API
key or real network egress.
"""

import base64
import threading
from http.server import ThreadingHTTPServer

import pytest

import cloud.handler as handler
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


def test_429_retried_then_succeeds(live_server, monkeypatch):
    """First call returns 429, second succeeds — client retries transparently."""
    import urllib.error
    call_count = {"n": 0}
    original_diagnose = srv.diagnose

    def flaky_diagnose(roi, emb, ctx):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise urllib.error.HTTPError(
                url="", code=429, msg="Too Many Requests", hdrs=None, fp=None
            )
        return original_diagnose(roi, emb, ctx)

    monkeypatch.setattr(srv, "diagnose", flaky_diagnose)
    client = CloudClient(live_server, max_attempts=3)
    out = client.diagnose(roi_png_b64=base64.b64encode(b"PNG").decode())
    assert out["defect_present"] is True
    assert call_count["n"] == 2


def test_429_exhausted_raises_unreachable(monkeypatch):
    """All attempts return 429 — CloudUnreachable is raised after max_attempts."""
    import urllib.error
    from unittest.mock import patch

    http_err = urllib.error.HTTPError(
        url="", code=429, msg="Too Many Requests", hdrs=None, fp=None
    )
    client = CloudClient("http://127.0.0.1:1", timeout_s=0.5, max_attempts=2)
    with patch("urllib.request.urlopen", side_effect=http_err):
        with pytest.raises(CloudUnreachable):
            client.diagnose(roi_png_b64=base64.b64encode(b"PNG").decode())


def test_4xx_not_retried(monkeypatch):
    """400 Bad Request is not retryable — should raise CloudUnreachable immediately."""
    import urllib.error
    from unittest.mock import patch

    call_count = {"n": 0}

    def counting_urlopen(*a, **kw):
        call_count["n"] += 1
        raise urllib.error.HTTPError(
            url="", code=400, msg="Bad Request", hdrs=None, fp=None
        )

    client = CloudClient("http://127.0.0.1:1", timeout_s=0.5, max_attempts=3)
    with patch("urllib.request.urlopen", side_effect=counting_urlopen):
        with pytest.raises((CloudUnreachable, urllib.error.HTTPError)):
            client.diagnose(roi_png_b64=base64.b64encode(b"PNG").decode())
    assert call_count["n"] == 1  # no retries
