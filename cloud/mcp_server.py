"""MCP tool server exposing `diagnose_defect`, deployed on Alibaba Cloud Function
Compute (build plan §6 — must NOT run from a laptop). Lands in M1.

The tool takes a privacy-filtered ROI (or embedding) plus minimal context and
returns the structured diagnosis from qwen_reason.diagnose().
"""

# from mcp.server import Server   # wired in M1
from cloud.qwen_reason import diagnose, validate


def diagnose_defect(roi_png_b64: str = "", embedding=None, context=None) -> dict:  # M1
    """MCP tool entrypoint. Decodes the ROI, calls Qwen-VL, returns validated JSON."""
    import base64

    roi = base64.b64decode(roi_png_b64) if roi_png_b64 else None
    result = diagnose(roi, embedding, context or {})
    return validate(result)


# def build_server() -> Server: ...   # register diagnose_defect as an MCP tool, M1
