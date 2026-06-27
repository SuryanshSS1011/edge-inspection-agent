"""MCP tool server exposing `diagnose_defect`, deployed on Alibaba Cloud Function
Compute (build plan §6 — must NOT run from a laptop).

The tool takes a privacy-filtered ROI (base64 PNG) or an abstracted embedding plus
minimal context, and returns the structured diagnosis from qwen_reason.diagnose().

`diagnose_defect` is plain Python (importable/testable without the MCP SDK). The
`build_server()` / `main()` wiring registers it as an MCP tool when the SDK is present.
"""

import base64
from typing import List, Optional

from cloud.qwen_reason import diagnose, validate


def diagnose_defect(
    roi_png_b64: str = "",
    embedding: Optional[List[float]] = None,
    context: Optional[dict] = None,
) -> dict:
    """MCP tool entrypoint. Decodes the ROI, calls Qwen-VL, returns validated JSON.

    Args:
        roi_png_b64: base64-encoded PNG of the cropped region of interest.
        embedding:   abstracted feature vector, used when no raw ROI is sent.
        context:     minimal non-PII context (e.g. {"category": "bottle"}).

    Returns:
        {defect_present, defect_type, confidence, root_cause, recommended_action}
    """
    if not roi_png_b64 and embedding is None:
        raise ValueError("diagnose_defect requires either roi_png_b64 or embedding")
    roi = base64.b64decode(roi_png_b64) if roi_png_b64 else None
    result = diagnose(roi, embedding, context or {})
    return validate(result)


def build_server():
    """Construct the MCP server with diagnose_defect registered. Requires the `mcp` SDK."""
    from mcp.server.fastmcp import FastMCP  # imported lazily so tests don't need the SDK

    server = FastMCP("edge-inspection-cloud")

    @server.tool()
    def diagnose_defect_tool(
        roi_png_b64: str = "",
        embedding: Optional[List[float]] = None,
        context: Optional[dict] = None,
    ) -> dict:
        """Diagnose a manufacturing defect from a region of interest. Returns structured JSON."""
        return diagnose_defect(roi_png_b64, embedding, context)

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
