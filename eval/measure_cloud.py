"""Measure REAL Qwen-VL cloud behavior on a small, capped batch of bottle ROIs.

Turns the modeled cloud numbers into measured ones for the results table's cloud row:
per-call latency (p50/p99) and verdict accuracy on real images. Free-tier-conscious:
--max-calls caps the number of live API calls (default 12), sampled to include both
classes and prioritize in-band items (the ones that actually escalate live).

    DASHSCOPE_API_KEY=... python -m eval.measure_cloud --max-calls 12

Writes eval/cloud_measured.json. Uses the privacy filter (embedding-free ROI PNG) so the
payload is exactly what the deployed path would send.
"""

import argparse
import base64
import json
import time

import numpy as np  # type: ignore

from edge.calibration import load as load_temperature
from edge.dotenv import load_dotenv
from edge.perception import OnnxClassifier, _pick_score_output, logit_from_output
from edge.router import Costs, escalation_band
from eval.features import extract
from eval.metrics import latency_percentiles


def _load_eval_with_p(model, splits, temperature):
    import onnxruntime as ort  # lazy

    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    clf = OnnxClassifier(model, temperature=temperature)
    out = []
    for path, label in splits["eval"]:
        feat = extract(path).reshape(1, -1).astype(np.float32)
        p = clf.predict_from_logit(logit_from_output(_pick_score_output(sess.run(None, {name: feat})))).p
        out.append((path, int(label), p))
    return out


def _roi_png_b64(path):
    """Encode a center-cropped ROI as PNG base64 — the same shape the edge would send.

    Uses Pillow (cv2 not required). Inset by 1/8 on each edge = the orchestrator's default
    ROI, so the payload matches the deployed path.
    """
    import io

    from PIL import Image  # lazy

    img = Image.open(path).convert("RGB")
    w, h = img.size
    # Light 1/16 inset: strips the frame border (still not the full frame, so the privacy
    # boundary holds) while keeping the part — including edge/mouth defects — in view.
    iw, ih = w // 16, h // 16
    roi = img.crop((iw, ih, w - iw, h - ih))
    buf = io.BytesIO()
    roi.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _sample(items, band, max_calls, seed=0):
    """Prioritize in-band items, keep both classes represented, cap at max_calls."""
    lo, hi = band
    rng = np.random.default_rng(seed)
    inband = [it for it in items if lo <= it[2] <= hi]
    outband = [it for it in items if not (lo <= it[2] <= hi)]
    defects_out = [it for it in outband if it[1] == 1]  # clear defects for class balance

    chosen = []
    # ensure both classes present: all in-band defects, some clear out-of-band defects
    chosen += [it for it in inband if it[1] == 1]
    chosen += list(rng.permutation(np.array(defects_out, dtype=object).reshape(-1, 3))[:2]) if defects_out else []
    goods_inband = [it for it in inband if it[1] == 0]
    rng.shuffle(goods_inband)
    chosen += goods_inband
    # dedupe by path, cap
    seen, capped = set(), []
    for it in chosen:
        path = it[0]
        if path not in seen:
            seen.add(path)
            capped.append(it)
        if len(capped) >= max_calls:
            break
    return capped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/classifier.onnx")
    parser.add_argument("--splits", default="models/splits.json")
    parser.add_argument("--temperature", default="models/temperature.json")
    parser.add_argument("--max-calls", type=int, default=12)
    parser.add_argument("--out", default="eval/cloud_measured.json")
    args = parser.parse_args()

    load_dotenv()
    from cloud.mcp_server import diagnose_defect  # imports the real Qwen path

    splits = json.loads(open(args.splits).read())
    temperature = load_temperature(args.temperature)
    costs = Costs(C_FN=100.0, C_FP=5.0, C_cloud=2.0, residual_cloud_error=0.3)
    band = escalation_band(costs)

    items = _load_eval_with_p(args.model, splits, temperature)
    batch = _sample(items, band, args.max_calls)
    print(f"live calls: {len(batch)} (cap {args.max_calls})")

    latencies, correct, records = [], 0, []
    for path, label, p in batch:
        roi_b64 = _roi_png_b64(path)
        t0 = time.time()
        diag = diagnose_defect(roi_png_b64=roi_b64, context={"category": "bottle"})
        dt = (time.time() - t0) * 1000
        latencies.append(dt)
        pred = 1 if diag.get("defect_present") else 0
        correct += int(pred == label)
        records.append({"path": path.split("/")[-2] + "/" + path.split("/")[-1],
                        "label": label, "local_p": round(p, 3),
                        "cloud_pred": pred, "latency_ms": round(dt)})
        print(f"  {records[-1]['path']:24} label={label} local_p={p:.3f} cloud={pred} {dt:.0f}ms")

    lat = latency_percentiles(latencies)
    result = {
        "n_calls": len(batch),
        "cloud_accuracy": round(correct / len(batch), 3) if batch else None,
        "latency_ms_p50": round(lat["p50"]),
        "latency_ms_p99": round(lat["p99"]),
        "latency_ms_mean": round(float(np.mean(latencies))) if latencies else None,
        "records": records,
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"\ncloud accuracy: {result['cloud_accuracy']}  "
          f"latency p50/p99: {result['latency_ms_p50']}/{result['latency_ms_p99']} ms")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
