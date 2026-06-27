"""Run every operating condition end-to-end and collect metrics (build plan §7).

Conditions (table rows): cloud-everything, local-only, hybrid (full/degraded/offline).
Each replays the same labeled item stream through the real orchestrator; we run multiple
seeds and report the spread (bootstrap CIs) on the headline metric.

The item stream comes from either:
  - a real run: the ONNX model over an MVTec split (use build_stream_from_mvtec), or
  - a fixture: a precomputed list of (p, label) for credential-free, deterministic runs.

    python -m eval.run_eval --data /path/to/mvtec --model models/classifier.onnx
    python -m eval.run_eval --fixture eval/fixtures/stream.json   # no model/cloud needed
"""

import argparse
import json
import os
import tempfile
from typing import List

from edge.config import load_config
from eval.harness import EvalItem, run_condition
from eval.metrics import bootstrap_ci

CONDITIONS = [
    "cloud_everything",
    "local_only",
    "hybrid_full",
    "hybrid_degraded",
    "hybrid_offline",
]


def run_all(items: List[EvalItem], base_costs, seeds=(0, 1, 2), cloud_accuracy=0.98) -> dict:
    """Return {condition: {"result": ConditionResult(seed0), "recall_ci": (lo,mean,hi)}}.

    The representative row's deterministic columns (latency, bytes, PII) come from seed 0;
    its cost-weighted recall is replaced by the across-seed mean so the point estimate is
    consistent with the bootstrap CI (cloud-verdict noise is the only seed-dependent term).
    """
    out = {}
    for condition in CONDITIONS:
        recalls = []
        first = None
        for seed in seeds:
            with tempfile.TemporaryDirectory() as d:
                db = os.path.join(d, "eval.db")
                res = run_condition(condition, items, base_costs, db,
                                    cloud_accuracy=cloud_accuracy, seed=seed)
            recalls.append(res.cost_weighted_recall)
            if first is None:
                first = res
        ci = bootstrap_ci(recalls, seed=0)
        first.cost_weighted_recall = ci[1]   # across-seed mean, matches the CI
        out[condition] = {"result": first, "recall_ci": ci}
    return out


def build_stream_from_fixture(path: str) -> List[EvalItem]:
    raw = json.loads(open(path).read())
    return [EvalItem(p=float(r["p"]), label=int(r["label"])) for r in raw]


def build_stream_from_mvtec(model_path: str, data_root: str, category: str, split: str,
                            temperature: float = 1.0) -> List[EvalItem]:
    """Run the real ONNX model over an MVTec split to build the (p, label) stream.

    Needs onnxruntime + cv2. This is the path that produces the live results table.
    """
    import cv2  # lazy
    import onnxruntime as ort  # lazy

    from edge.perception import OnnxClassifier, logit_from_output, preprocess
    from eval.datasets import load_mvtec

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    clf = OnnxClassifier(model_path, temperature=temperature)

    items = []
    for image_path, label, _cat in load_mvtec(data_root, category, split):
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        output = session.run(None, {input_name: preprocess(frame)})[0]
        p = clf.predict_from_logit(logit_from_output(output)).p
        items.append(EvalItem(p=p, label=label))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fixture", help="JSON list of {p, label} for a model-free run")
    parser.add_argument("--data", help="MVTec root (real run)")
    parser.add_argument("--model", default="models/classifier.onnx")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="eval/results.json")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.fixture:
        items = build_stream_from_fixture(args.fixture)
    elif args.data:
        items = build_stream_from_mvtec(args.model, args.data, args.category, args.split)
    else:
        raise SystemExit("provide --fixture or --data")

    results = run_all(items, config.costs)
    serializable = {
        cond: {
            "result": vars(v["result"]),
            "recall_ci": v["recall_ci"],
        }
        for cond, v in results.items()
    }
    with open(args.out, "w") as fh:
        json.dump(serializable, fh, indent=2)
    print(f"wrote {args.out} ({len(items)} items, {len(CONDITIONS)} conditions)")


if __name__ == "__main__":
    main()
