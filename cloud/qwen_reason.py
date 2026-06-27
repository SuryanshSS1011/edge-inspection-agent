"""Qwen-VL reasoning over an escalated ROI/embedding -> structured JSON (§3.4).

Prompts Qwen-VL to return a strict schema and validates it server-side so the edge
always receives well-formed output. Lands in M1.
"""

import json
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
    "interest from a manufacturing part, return ONLY a JSON object with keys: "
    "defect_present (bool), defect_type (string), confidence (0-1 float), "
    "root_cause (string), recommended_action (string). No prose."
)


def diagnose(roi_png: Optional[bytes], embedding: Optional[list], context: dict) -> dict:  # M1
    """Call Qwen-VL via the Qwen Cloud API, parse + validate the JSON, return it.

    Raises if the model output does not match DIAGNOSIS_SCHEMA_KEYS after a retry.
    """
    raise NotImplementedError


def validate(diagnosis: dict) -> dict:
    missing = DIAGNOSIS_SCHEMA_KEYS - diagnosis.keys()
    if missing:
        raise ValueError(f"diagnosis missing keys: {sorted(missing)}")
    return diagnosis
