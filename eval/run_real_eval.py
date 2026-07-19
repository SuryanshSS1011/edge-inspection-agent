"""Run the results table on the REAL bottle data using the trained ONNX model and fitted
temperature over the disjoint eval split.

This is the credential-free-but-real path. The local `p` comes from the actual model on
actual MVTec images; the cloud is still modeled (the live cloud row needs the deployed
endpoint). Produces the same five-condition table as eval.run_eval, on real perception.

    python -m eval.run_real_eval --model models/classifier.onnx \
        --splits models/splits.json --temperature models/temperature.json
"""

import argparse
import json

import numpy as np  # type: ignore

from edge.calibration import load as load_temperature
from edge.perception import OnnxClassifier, _pick_score_output, logit_from_output
from edge.router import Costs
from eval.features import extract
from eval.harness import EvalItem
from eval.make_table import to_markdown
from eval.run_eval import run_all


def build_stream(model_path: str, splits_path: str, temperature: float):
    import onnxruntime as ort  # lazy

    splits = json.loads(open(splits_path).read())
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    clf = OnnxClassifier(model_path, temperature=temperature)

    items = []
    for path, label in splits["eval"]:
        feat = extract(path).reshape(1, -1).astype(np.float32)
        outputs = session.run(None, {input_name: feat})
        logit = logit_from_output(_pick_score_output(outputs))
        p = clf.predict_from_logit(logit).p
        items.append(EvalItem(p=p, label=int(label)))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/classifier.onnx")
    parser.add_argument("--splits", default="models/splits.json")
    parser.add_argument("--temperature", default="models/temperature.json")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="eval/results_table_real.md")
    args = parser.parse_args()

    from edge.config import load_config

    temperature = load_temperature(args.temperature)
    items = build_stream(args.model, args.splits, temperature)
    costs = load_config(args.config).costs

    results = run_all(items, costs, seeds=(0, 1, 2, 3, 4))
    table = to_markdown(results)
    header = (
        "# Results on real bottle data (local model + fitted calibration)\n\n"
        f"Local `p` from the trained ONNX classifier over {len(items)} held-out eval images "
        "(disjoint from train/calibration). Cloud is modeled; the live cloud row needs the "
        "deployed endpoint.\n\n"
    )
    open(args.out, "w").write(header + table + "\n")
    print(f"{len(items)} eval items")
    print(table)


if __name__ == "__main__":
    main()
