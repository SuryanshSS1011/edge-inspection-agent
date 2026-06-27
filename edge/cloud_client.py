"""Client for the deployed cloud reasoning tool. Calls the Alibaba Function Compute
HTTP endpoint (`POST /diagnose`) and parses the structured JSON diagnosis (§3.4).

Dependency-free (urllib) to keep the edge runtime light. Raises CloudUnreachable on
network/timeout failures so the orchestrator can fall back to a local action.
"""

import json
import urllib.error
import urllib.request
from typing import List, Optional


class CloudUnreachable(RuntimeError):
    """The cloud endpoint could not be reached or failed — caller should fall back."""


class CloudClient:
    def __init__(self, server_url: str, timeout_s: float = 10.0):
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s

    def diagnose(
        self,
        roi_png_b64: str = "",
        embedding: Optional[List[float]] = None,
        context: Optional[dict] = None,
    ) -> dict:
        """Invoke the cloud tool with a privacy-filtered payload.

        Returns: {defect_present, defect_type, confidence, root_cause, recommended_action}
        Raises CloudUnreachable on any transport/timeout/server error.
        """
        body = json.dumps({
            "roi_png_b64": roi_png_b64,
            "embedding": embedding,
            "context": context or {},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server_url}/diagnose",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise CloudUnreachable(str(exc)) from exc

    def healthz(self) -> bool:
        """Lightweight reachability probe for the network controller (M6)."""
        try:
            with urllib.request.urlopen(f"{self.server_url}/healthz", timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8")).get("ok") is True
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return False
