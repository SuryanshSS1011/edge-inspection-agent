"""Train the modest local defect classifier on MVTec and export it to ONNX.

Design choices (see the calibration rationale in docs):
  - A LogisticRegression on lightweight features (eval/features.py), NOT a CNN. The router
    needs a calibrated model that is genuinely uncertain near the boundary so it has real
    escalation work to do; local accuracy is not the target.
  - THREE disjoint splits, train / calibration / eval, each with both classes. ECE must
    be measured on data the temperature was NOT fit on, or it is circular.
  - Exports the raw decision logit (not the probability) so edge.perception's temperature
    scaling operates on logits, exactly as with any other ONNX model.

    python -m eval.train_classifier --data data --category bottle

Outputs:
    models/classifier.onnx        the model (input: feature vector, output: defect logit)
    models/splits.json            the frozen train/calibration/eval file lists (reuse in eval)
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np  # type: ignore

from eval.datasets import load_mvtec
from eval import features as handcrafted


def _backbone(name):
    """Return (extract_many, feature_dim) for the chosen feature backbone.

    handcrafted: lightweight color/edge/variance features (eval/features.py).
    mobilenet:   frozen MobileNetV2 ONNX embeddings (eval/mobilenet_features.py).
    dinov2:      frozen DINOv2 ViT embeddings, the SOTA anomaly-detection backbone
                 (eval/dinov2_features.py).
    All feed the SAME LogisticRegression + temperature head; the router, privacy filter,
    and outbox are unchanged. That's the point of the drop-in interface.
    """
    if name == "handcrafted":
        return handcrafted.extract_many, handcrafted.FEATURE_DIM
    if name == "mobilenet":
        from eval import mobilenet_features as mb
        return mb.extract_many, mb.FEATURE_DIM
    if name == "dinov2":
        from eval import dinov2_features as dv
        # FEATURE_DIM is the variant default; the real dim is confirmed from the ONNX at
        # extract time. export_onnx uses this only to name the input tensor width, and the
        # exporter writes a model whose input matches the features produced, so they agree.
        return dv.extract_many, dv.FEATURE_DIM
    raise ValueError(f"unknown backbone {name!r} (use handcrafted | mobilenet | dinov2)")


def _split_paths(data: str, category: str, seed: int = 0):
    """Return disjoint (train, calib, eval) lists of (path, label), both classes in each.

    MVTec convention: train/ is all good; test/ holds good + defects. We pool everything,
    then stratify-split per class so each split has goods and defects.
    """
    pool = list(load_mvtec(data, category, "train")) + list(load_mvtec(data, category, "test"))
    goods = [p for p, l, _ in pool if l == 0]
    defects = [p for p, l, _ in pool if l == 1]

    rng = np.random.default_rng(seed)
    rng.shuffle(goods)
    rng.shuffle(defects)

    def three_way(items):
        n = len(items)
        n_tr, n_cal = int(0.6 * n), int(0.2 * n)
        return items[:n_tr], items[n_tr:n_tr + n_cal], items[n_tr + n_cal:]

    g_tr, g_cal, g_ev = three_way(goods)
    d_tr, d_cal, d_ev = three_way(defects)

    def labeled(g, d):
        return [(p, 0) for p in g] + [(p, 1) for p in d]

    return labeled(g_tr, d_tr), labeled(g_cal, d_cal), labeled(g_ev, d_ev)


def _xy(items: List[Tuple[str, int]], extract_many):
    X = extract_many([p for p, _ in items])
    y = np.array([l for _, l in items], dtype=np.int64)
    return X, y


def export_onnx(model, scaler, out_path: str, feature_dim: int) -> None:
    """Export scaler -> logistic regression as a single ONNX graph emitting the defect logit."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.pipeline import make_pipeline

    pipe = make_pipeline(scaler, model)
    initial = [("input", FloatTensorType([None, feature_dim]))]
    # zipmap=False keeps a plain score tensor; we read the decision logit from raw scores.
    # target_opset 17 stays within onnxruntime's official support range (<=21).
    onnx_model = convert_sklearn(
        pipe, initial_types=initial,
        options={id(pipe.steps[-1][1]): {"zipmap": False}},
        target_opset=17,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(onnx_model.SerializeToString())


def train_and_export(data, category, model_out, splits_out, seed=0, backbone="handcrafted"):
    """Train the modest classifier for one category and export model + frozen splits.

    `backbone` selects the feature extractor ('handcrafted' | 'mobilenet'). The head and
    everything downstream is identical either way. Returns the train accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    extract_many, _declared_dim = _backbone(backbone)
    train, calib, ev = _split_paths(data, category, seed)
    Xtr, ytr = _xy(train, extract_many)
    # Trust the actual extracted width over the backbone's declared dim, so a DINOv2 variant
    # (384/768/1024) or any embedding size exports a correctly-shaped ONNX input.
    feature_dim = Xtr.shape[1]

    scaler = StandardScaler().fit(Xtr)
    # Deliberately modest: light regularization, no heroics. We want calibratable
    # uncertainty near the boundary, not a saturated classifier.
    clf = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)

    export_onnx(clf, scaler, model_out, feature_dim)
    Path(splits_out).parent.mkdir(parents=True, exist_ok=True)
    Path(splits_out).write_text(json.dumps({
        "train": train, "calibration": calib, "eval": ev,
        "category": category, "seed": seed, "backbone": backbone,
    }, indent=2))
    return clf.score(scaler.transform(Xtr), ytr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backbone", default="handcrafted",
                        choices=["handcrafted", "mobilenet", "dinov2"])
    parser.add_argument("--model-out", default="models/classifier.onnx")
    parser.add_argument("--splits-out", default="models/splits.json")
    args = parser.parse_args()

    acc = train_and_export(args.data, args.category, args.model_out, args.splits_out,
                           args.seed, backbone=args.backbone)
    print(f"backbone: {args.backbone}  train accuracy: {acc:.3f}")
    print(f"wrote {args.model_out} and {args.splits_out}")


if __name__ == "__main__":
    main()
