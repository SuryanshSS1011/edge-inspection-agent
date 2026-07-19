"""Client for the deployed cloud reasoning tool. Calls the Alibaba Function Compute
HTTP endpoint (`POST /diagnose`) and parses the structured JSON diagnosis (§3.4).

Raises CloudUnreachable on any transport/timeout/server error so the orchestrator
can fall back to a local action. Retryable errors (429 rate-limit and 5xx transient
failures) are retried with exponential backoff + full jitter via tenacity before
CloudUnreachable is raised, protecting recovery queues during extended outage drains.
"""

import json
import urllib.error
import urllib.request
from typing import List, Optional

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)


class CloudUnreachable(RuntimeError):
    """The cloud endpoint could not be reached or failed, so the caller should fall back."""


def _is_retryable(exc: BaseException) -> bool:
    """Retry on 429 rate-limit or 5xx transient server errors; pass 4xx through immediately."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, (urllib.error.URLError, TimeoutError, OSError))


class CloudClient:
    def __init__(self, server_url: str, timeout_s: float = 10.0,
                 max_attempts: int = 3):
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s
        self._max_attempts = max_attempts

    def diagnose(
        self,
        roi_png_b64: str = "",
        embedding: Optional[List[float]] = None,
        context: Optional[dict] = None,
    ) -> dict:
        """Invoke the cloud tool with a privacy-filtered payload.

        Retries up to max_attempts times on 429/5xx with exponential backoff + full
        jitter (2-30 s). All other errors raise CloudUnreachable immediately.

        Returns: {defect_present, defect_type, confidence, root_cause, recommended_action}
        """
        body = json.dumps({
            "roi_png_b64": roi_png_b64,
            "embedding": embedding,
            "context": context or {},
        }).encode("utf-8")

        @retry(
            retry=retry_if_exception(_is_retryable),
            wait=wait_random_exponential(multiplier=1, min=2, max=30),
            stop=stop_after_attempt(self._max_attempts),
            reraise=True,
        )
        def _call() -> dict:
            req = urllib.request.Request(
                f"{self.server_url}/diagnose",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))

        try:
            return _call()
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise CloudUnreachable(str(exc)) from exc

    def healthz(self) -> bool:
        """Lightweight reachability probe for the network controller (M6)."""
        try:
            with urllib.request.urlopen(f"{self.server_url}/healthz", timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8")).get("ok") is True
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return False
