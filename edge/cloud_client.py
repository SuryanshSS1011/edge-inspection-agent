"""MCP client wrapper: calls the deployed cloud reasoning tool and parses the
structured JSON diagnosis (§3.4). Lands in M4 (wired) once M1's tool is deployed.
"""

from typing import Optional


class CloudClient:
    def __init__(self, server_url: str, timeout_s: float = 10.0):
        self.server_url = server_url
        self.timeout_s = timeout_s

    def diagnose(self, payload: dict) -> dict:  # M4
        """Invoke the `diagnose_defect` MCP tool with a privacy-filtered payload.

        Returns structured JSON:
          {defect_present, defect_type, confidence, root_cause, recommended_action}
        Raises on unreachable cloud so the orchestrator can fall back to local action.
        """
        raise NotImplementedError
