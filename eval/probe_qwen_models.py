"""Probe which Qwen-VL model ids actually resolve on your DashScope voucher, and how fast.

Model availability varies by region/voucher, and "best of the best" means using the
strongest model that is actually reachable, not the strongest one on paper. This sends a
tiny real vision request to each candidate and reports which succeed, their latency, and
whether the structured-output contract holds. Run it before a demo to pick QWEN_MODEL.

    python -m eval.probe_qwen_models

Needs DASHSCOPE_API_KEY (from .env). Costs a handful of cheap calls.
"""

import base64
import io
import os
import time
from typing import List

# Candidate ids in rough strongest-first order. Add/remove as DashScope changes.
CANDIDATES: List[str] = [
    "qwen3-vl-max",
    "qwen3-vl-plus",
    "qwen3.7-plus",
    "qwen-vl-max",
    "qwen-vl-plus",
    "qwen3-vl-flash",
]


def _tiny_png_b64() -> str:
    from PIL import Image  # lazy

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (170, 170, 170)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _probe_one(model: str, roi_b64: str, timeout: float = 40.0) -> dict:
    """Send one real diagnose-style call. Returns {model, ok, ms, note}."""
    from cloud import qwen_reason

    prev = os.environ.get("QWEN_MODEL")
    os.environ["QWEN_MODEL"] = model
    t0 = time.time()
    try:
        out = qwen_reason.diagnose(
            roi_png=base64.b64decode(roi_b64),
            embedding=None,
            context={"category": "probe"},
            timeout=timeout,
            max_retries=0,
        )
        ms = int((time.time() - t0) * 1000)
        keys_ok = qwen_reason.DIAGNOSIS_SCHEMA_KEYS <= out.keys()
        return {"model": model, "ok": True, "ms": ms, "note": "schema ok" if keys_ok else "schema drift"}
    except Exception as exc:  # noqa: BLE001 - report every failure mode compactly
        ms = int((time.time() - t0) * 1000)
        msg = str(exc)
        # Compact the common "model not found" upstream error.
        if "model" in msg.lower() and ("not" in msg.lower() or "invalid" in msg.lower()):
            note = "unavailable on this endpoint"
        else:
            note = msg[:80]
        return {"model": model, "ok": False, "ms": ms, "note": note}
    finally:
        if prev is None:
            os.environ.pop("QWEN_MODEL", None)
        else:
            os.environ["QWEN_MODEL"] = prev


def main() -> None:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        try:
            from edge.dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("DASHSCOPE_API_KEY not set (put it in .env)")

    roi = _tiny_png_b64()
    print(f"{'model':<20} {'ok':<4} {'ms':>7}  note")
    print("-" * 60)
    working = []
    for model in CANDIDATES:
        r = _probe_one(model, roi)
        flag = "yes" if r["ok"] else "no"
        print(f"{r['model']:<20} {flag:<4} {r['ms']:>7}  {r['note']}")
        if r["ok"]:
            working.append(r)

    print("-" * 60)
    if working:
        best = working[0]  # CANDIDATES is strongest-first, so first working is best
        print(f"\nRecommended QWEN_MODEL={best['model']}  (strongest that resolved, {best['ms']} ms)")
    else:
        print("\nNo candidate resolved. Check the key/region and DashScope model list.")


if __name__ == "__main__":
    main()
