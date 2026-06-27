"""M2 exit check: fit the temperature scalar on a validation split and report whether
calibration improved.

Runs the ONNX model over a labeled MVTec split, collects defect logits, fits T, saves
it to models/temperature.json, and prints expected calibration error before vs. after.

    python -m eval.fit_calibration --model models/classifier.onnx \
        --data /path/to/mvtec --category bottle --split train

Needs onnxruntime + cv2 to run the model. The fitting math itself is covered by
tests/test_calibration.py and needs neither.
"""

import argparse

import numpy as np  # type: ignore

from edge.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    save,
)
from edge.perception import OnnxClassifier, logit_from_output, preprocess
from eval.datasets import load_mvtec


def collect_logits(model_path: str, data_root: str, category: str, split: str):
    import cv2  # lazy

    import onnxruntime as ort  # lazy

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    logits, labels = [], []
    for image_path, label, _category in load_mvtec(data_root, category, split):
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        output = session.run(None, {input_name: preprocess(frame)})[0]
        logits.append(logit_from_output(output))
        labels.append(float(label))
    return np.array(logits), np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/classifier.onnx")
    parser.add_argument("--data", required=True, help="MVTec AD root directory")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="models/temperature.json")
    args = parser.parse_args()

    logits, labels = collect_logits(args.model, args.data, args.category, args.split)
    if logits.size == 0:
        raise SystemExit("no validation samples collected — check --data/--category")

    raw_p = apply_temperature(logits, 1.0)
    temperature = fit_temperature(logits, labels)
    cal_p = apply_temperature(logits, temperature)

    ece_before = expected_calibration_error(raw_p, labels)
    ece_after = expected_calibration_error(cal_p, labels)
    save(temperature, args.out)

    print(f"samples:           {logits.size}")
    print(f"temperature:       {temperature:.3f}")
    print(f"ECE before:        {ece_before:.4f}")
    print(f"ECE after:         {ece_after:.4f}")
    print(f"saved temperature -> {args.out}")
    if ece_after > ece_before + 1e-6:
        print("WARNING: calibration did not improve ECE; inspect the validation split.")


if __name__ == "__main__":
    main()
