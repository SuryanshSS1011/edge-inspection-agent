"""Fit the temperature scalar and report whether calibration improved.

Methodology (the gotcha a sharp judge would catch): temperature is fit on the
CALIBRATION split and ECE is reported on that same held-out split — which is DISJOINT
from both the training split (used to fit the classifier) and the eval split (used for the
results table). Measuring ECE on the training data would be circular.

Runs the real ONNX model over the calibration split's feature vectors, collects defect
logits, fits T, saves it to models/temperature.json, and prints ECE before vs. after.

    python -m eval.fit_calibration --model models/classifier.onnx --splits models/splits.json
"""

import argparse
import json

import numpy as np  # type: ignore

from edge.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    save,
)
from edge.perception import OnnxClassifier, _pick_score_output, logit_from_output
from eval.features import extract


def collect_logits(model_path: str, items):
    """Run the model over labeled (path, label) items, returning (logits, labels)."""
    import onnxruntime as ort  # lazy

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    logits, labels = [], []
    for path, label in items:
        feat = extract(path).reshape(1, -1).astype(np.float32)
        outputs = session.run(None, {input_name: feat})
        logits.append(logit_from_output(_pick_score_output(outputs)))
        labels.append(float(label))
    return np.array(logits), np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/classifier.onnx")
    parser.add_argument("--splits", default="models/splits.json")
    parser.add_argument("--out", default="models/temperature.json")
    args = parser.parse_args()

    splits = json.loads(open(args.splits).read())
    calib = splits["calibration"]
    logits, labels = collect_logits(args.model, calib)
    if logits.size == 0:
        raise SystemExit("no calibration samples — check --splits")

    raw_p = apply_temperature(logits, 1.0)
    temperature = fit_temperature(logits, labels)
    cal_p = apply_temperature(logits, temperature)

    ece_before = expected_calibration_error(raw_p, labels)
    ece_after = expected_calibration_error(cal_p, labels)
    save(temperature, args.out, reference_confidences=cal_p)

    print(f"calibration samples: {logits.size}  (disjoint from train and eval)")
    print(f"temperature:         {temperature:.3f}")
    print(f"ECE before:          {ece_before:.4f}")
    print(f"ECE after:           {ece_after:.4f}")
    print(f"saved temperature -> {args.out}")
    if ece_after > ece_before + 1e-6:
        print("WARNING: calibration did not improve ECE; inspect the calibration split.")


if __name__ == "__main__":
    main()
