"""Tests for the cloud reasoning logic that doesn't require network/API access:
schema validation, JSON parsing, coercion, and message assembly.
"""

import json

import pytest

from cloud.qwen_reason import (
    DIAGNOSIS_SCHEMA_KEYS,
    _build_messages,
    _coerce,
    _parse_json,
    validate,
)

GOOD = {
    "defect_present": True,
    "defect_type": "crack",
    "confidence": 0.91,
    "root_cause": "thermal stress during cure",
    "recommended_action": "reject_and_flag_batch",
}


def test_validate_accepts_good_diagnosis():
    assert validate(dict(GOOD)) == GOOD


def test_validate_strips_extra_keys():
    extra = dict(GOOD, hallucinated="ignore me")
    out = validate(extra)
    assert out.keys() == DIAGNOSIS_SCHEMA_KEYS


def test_validate_rejects_missing_keys():
    bad = dict(GOOD)
    del bad["root_cause"]
    with pytest.raises(ValueError):
        validate(bad)


def test_validate_rejects_non_bool_present():
    with pytest.raises(ValueError):
        validate(dict(GOOD, defect_present="true"))


def test_validate_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        validate(dict(GOOD, confidence=1.5))


def test_parse_json_plain():
    assert _parse_json(json.dumps(GOOD)) == GOOD


def test_parse_json_strips_code_fence():
    fenced = "```json\n" + json.dumps(GOOD) + "\n```"
    assert _parse_json(fenced) == GOOD


def test_parse_json_extracts_object_amid_prose():
    noisy = "Here is the result: " + json.dumps(GOOD) + " hope that helps"
    assert _parse_json(noisy) == GOOD


def test_coerce_string_bool_and_confidence():
    out = _coerce({"defect_present": "Yes", "confidence": "0.5"})
    assert out["defect_present"] is True
    assert out["confidence"] == 0.5


def test_build_messages_includes_image_data_url():
    msgs = _build_messages(b"\x89PNG\r\n", None, {"line": "A"})
    user = msgs[-1]["content"]
    image_parts = [p for p in user if p.get("type") == "image_url"]
    assert image_parts and image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_build_messages_embedding_fallback_has_no_image():
    msgs = _build_messages(None, [0.1, 0.2, 0.3], {})
    user = msgs[-1]["content"]
    assert all(p.get("type") != "image_url" for p in user)
