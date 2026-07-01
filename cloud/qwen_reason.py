"""Qwen-VL reasoning over an escalated ROI/embedding -> structured JSON (§3.4).

Prompts Qwen-VL to return a strict schema and validates it server-side so the edge
always receives well-formed output.

Transport: the Qwen Cloud API (DashScope) exposes an OpenAI-compatible endpoint, so
this calls it over plain HTTP (no SDK dependency — keeps the cloud function small).
Set these env vars in the deployment:
    DASHSCOPE_API_KEY   - the Qwen Cloud API key (from the hackathon voucher)
    QWEN_MODEL          - vision model id, defaults to "qwen-vl-plus"
    QWEN_BASE_URL       - defaults to the international DashScope compatible endpoint
"""

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Optional

DIAGNOSIS_SCHEMA_KEYS = {
    "defect_present",     # bool
    "defect_type",        # str
    "confidence",         # float 0..1
    "root_cause",         # str
    "recommended_action", # str
}

SYSTEM_PROMPT = (
    "You are an industrial visual-inspection reasoner. Given a cropped region of "
    "interest from a manufacturing part, return ONLY a JSON object with exactly these "
    "keys: defect_present (boolean), defect_type (string, \"none\" if no defect), "
    "confidence (number 0-1), root_cause (string), recommended_action (string). "
    "No markdown, no prose, no code fences — JSON object only."
)

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-vl-plus"


class CloudConfigError(RuntimeError):
    """Raised when required cloud credentials/config are missing."""


def _build_messages(roi_png: Optional[bytes], embedding: Optional[list], context: dict):
    user_content = []
    ctx_note = json.dumps(context) if context else "{}"
    user_content.append({
        "type": "text",
        "text": f"Inspect this region of a manufacturing part. Context: {ctx_note}",
    })
    if roi_png is not None:
        b64 = base64.b64encode(roi_png).decode("ascii")
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    elif embedding is not None:
        # No raw image available (privacy mode = embedding): pass the abstracted vector.
        user_content.append({
            "type": "text",
            "text": f"Abstracted feature embedding of the ROI: {embedding}",
        })
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _post_chat(messages, *, base_url: str, model: str, api_key: str, timeout: float) -> str:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


def _parse_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from the model's reply."""
    text = text.strip()
    if text.startswith("```"):
        # Strip a ```json ... ``` fence if the model added one despite instructions.
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return json.loads(text)


def diagnose(
    roi_png: Optional[bytes],
    embedding: Optional[list],
    context: dict,
    *,
    timeout: float = 30.0,
    max_retries: int = 1,
) -> dict:
    """Call Qwen-VL via the Qwen Cloud API, parse + validate the JSON, return it.

    Retries once on malformed/invalid output. Raises CloudConfigError if the API key
    is missing, ValueError if the model output cannot be made to match the schema.
    """
    try:
        from edge.dotenv import load_dotenv
        load_dotenv()  # pick up DASHSCOPE_API_KEY from .env for local runs
    except Exception:
        pass  # in the deployed function the vars come from the FC environment
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise CloudConfigError("DASHSCOPE_API_KEY is not set")
    base_url = os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("QWEN_MODEL", DEFAULT_MODEL)
    messages = _build_messages(roi_png, embedding, context)

    last_err: Optional[Exception] = None
    for _ in range(max_retries + 1):
        try:
            raw = _post_chat(
                messages, base_url=base_url, model=model, api_key=api_key, timeout=timeout
            )
            return validate(_coerce(_parse_json(raw)))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            last_err = exc
            messages.append({
                "role": "user",
                "content": "That was not valid. Return ONLY the JSON object with the "
                           "required keys.",
            })
    raise ValueError(f"Qwen-VL did not return valid diagnosis JSON: {last_err}")


def _coerce(d: dict) -> dict:
    """Light normalization so minor type drift doesn't fail an otherwise-good answer."""
    if "defect_present" in d and isinstance(d["defect_present"], str):
        d["defect_present"] = d["defect_present"].strip().lower() in ("true", "yes", "1")
    if "confidence" in d:
        try:
            d["confidence"] = float(d["confidence"])
        except (TypeError, ValueError):
            pass
    return d


def validate(diagnosis: dict) -> dict:
    missing = DIAGNOSIS_SCHEMA_KEYS - diagnosis.keys()
    if missing:
        raise ValueError(f"diagnosis missing keys: {sorted(missing)}")
    if not isinstance(diagnosis["defect_present"], bool):
        raise ValueError("defect_present must be a boolean")
    conf = diagnosis["confidence"]
    if not isinstance(conf, (int, float)) or not 0.0 <= float(conf) <= 1.0:
        raise ValueError("confidence must be a number in [0, 1]")
    # Drop any extra keys the model invented so the edge gets exactly the contract.
    return {k: diagnosis[k] for k in DIAGNOSIS_SCHEMA_KEYS}
