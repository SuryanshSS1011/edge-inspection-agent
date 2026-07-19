"""LOCO experiment: does the cost router escalate LOGICAL anomalies to the cloud?

Hypothesis. A local texture model is near-blind to logical anomalies (wrong count, wrong
arrangement, a missing or extra object): every surface looks normal, so it cannot tell them
from a good part. Two outcomes are possible and we report whichever is true:

  (a) The local model is genuinely UNCERTAIN on logical anomalies -> p lands in the
      escalation band -> the router sends them to Qwen-VL, which can reason about count and
      arrangement -> hybrid catches what local-only misses. This is the thesis confirmed.
  (b) The local model is CONFIDENTLY WRONG on logical anomalies -> p sits outside the band
      -> the router does NOT escalate -> hybrid misses them too. This is an honest limitation
      and motivates a logical-anomaly-aware uncertainty signal as future work.

The script trains the modest local classifier on each LOCO category (good vs. any anomaly),
computes calibrated p on the labeled test split, and reports, split by anomaly KIND:
  - escalation rate (fraction of items whose p falls in the router's band)
  - local-only vs. hybrid cost-weighted recall

Run on ROAR (has the dataset). Writes eval/results_loco.md.

    python -m eval.run_loco --data /scratch/sss6371/datasets/loco
"""

import argparse
import json
import tempfile

import numpy as np  # type: ignore

from edge.calibration import fit_temperature
from edge.perception import OnnxClassifier, _pick_score_output, logit_from_output
from edge.router import Costs, escalation_band
from eval.datasets_loco import LOCO_CATEGORIES, load_loco

COSTS = Costs(C_FN=100.0, C_FP=5.0, C_cloud=2.0, residual_cloud_error=0.3)


def _kinded_items(data, category, backbone, workdir):
    """Train on this category, return per-test-image (p, label, kind) on the eval split.

    LOCO has no separate calibration split, so we hold out a slice of train (good) plus a
    slice of the test anomalies for temperature fitting, keeping the reported eval disjoint.
    """
    import onnxruntime as ort
    from eval.train_classifier import _backbone, export_onnx
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    extract_many, _ = _backbone(backbone)

    # Train: LOCO train/ is all good; pull anomalies from a held-out half of test so the
    # classifier sees both classes. The OTHER half of test is the disjoint eval set.
    train_good = [p for p, _, _ in load_loco(data, category, "train")]
    test = list(load_loco(data, category, "test"))
    anos = [(p, k) for p, lbl, k in test if lbl == 1]
    goods = [(p, k) for p, lbl, k in test if lbl == 0]

    rng = np.random.default_rng(0)
    rng.shuffle(anos)
    rng.shuffle(goods)
    half_a, half_g = len(anos) // 2, len(goods) // 2
    fit_anos, eval_anos = anos[:half_a], anos[half_a:]
    fit_goods, eval_goods = goods[:half_g], goods[half_g:]

    # Fit classifier on train-good + fit-anomalies (+ some fit-goods for balance).
    Xg = extract_many(train_good + [p for p, _ in fit_goods])
    Xa = extract_many([p for p, _ in fit_anos])
    X = np.vstack([Xg, Xa])
    y = np.array([0] * len(Xg) + [1] * len(Xa))

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced")
    clf.fit(scaler.transform(X), y)

    model = f"{workdir}/{category}_{backbone}.onnx"
    export_onnx(clf, scaler, model, X.shape[1])
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    def logit(path):
        f = extract_many([path]).astype(np.float32)
        return logit_from_output(_pick_score_output(sess.run(None, {name: f})))

    # Temperature on the fit split (disjoint from eval).
    fit_paths = [p for p, _ in fit_goods] + [p for p, _ in fit_anos]
    fit_labels = [0] * len(fit_goods) + [1] * len(fit_anos)
    Lc = np.array([logit(p) for p in fit_paths])
    Yc = np.array([float(l) for l in fit_labels])
    temperature = fit_temperature(Lc, Yc) if len(set(fit_labels)) == 2 else 1.0
    clf_head = OnnxClassifier(model, temperature=temperature)

    # Eval items with their kind.
    items = []
    for path, kind in [(p, "good") for p, _ in eval_goods] + [(p, k) for p, k in eval_anos]:
        label = 0 if kind == "good" else 1
        p_cal = clf_head.predict_from_logit(logit(path)).p
        items.append((p_cal, label, kind))
    return items


def _analyze(items):
    """Per-kind escalation rate and local-only vs hybrid recall. cloud is assumed strong on
    escalated items (the modeled hybrid catches an escalated defect); the honest variable is
    whether logical anomalies get escalated at all."""
    band = escalation_band(COSTS)
    lo, hi = (band if band else (1.0, 0.0))

    def in_band(p):
        return lo <= p <= hi

    # local decision: reject if p >= p* (the cost-optimal local boundary)
    pstar = COSTS.C_FP / (COSTS.C_FP + COSTS.C_FN)

    by_kind = {}
    for kind in ("good", "logical", "structural"):
        sub = [(p, lbl) for p, lbl, k in items if k == kind]
        if not sub:
            continue
        n = len(sub)
        escalated = sum(1 for p, _ in sub if in_band(p))
        # local-only catch rate on defects (recall): local rejects when p >= pstar
        defects = [(p, lbl) for p, lbl in sub if lbl == 1]
        local_caught = sum(1 for p, _ in defects if p >= pstar)
        # hybrid: an escalated defect is caught by the cloud; a non-escalated one relies on local
        hybrid_caught = sum(1 for p, _ in defects if in_band(p) or p >= pstar)
        by_kind[kind] = {
            "n": n,
            "escalation_rate": escalated / n if n else 0.0,
            "local_recall": (local_caught / len(defects)) if defects else None,
            "hybrid_recall": (hybrid_caught / len(defects)) if defects else None,
        }
    return by_kind


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="LOCO root (extracted)")
    parser.add_argument("--categories", nargs="+", default=LOCO_CATEGORIES)
    parser.add_argument("--backbone", default="dinov2",
                        choices=["handcrafted", "mobilenet", "dinov2"])
    parser.add_argument("--out", default="eval/results_loco.md")
    args = parser.parse_args()

    rows = {}
    with tempfile.TemporaryDirectory() as wd:
        for cat in args.categories:
            print(f"[{cat}] training + scoring...")
            items = _kinded_items(args.data, cat, args.backbone, wd)
            rows[cat] = _analyze(items)
            for kind in ("logical", "structural"):
                r = rows[cat].get(kind)
                if r:
                    print(f"  {kind:<11} n={r['n']:<4} escalated={r['escalation_rate']:.0%}  "
                          f"local_recall={r['local_recall']:.2f}  hybrid_recall={r['hybrid_recall']:.2f}")

    _write_report(rows, args.backbone, args.out)


def _write_report(rows, backbone, out):
    lines = [
        f"# LOCO experiment: does the router escalate logical anomalies? ({backbone} backbone)",
        "",
        "Structural anomalies are local (a texture model can see them). Logical anomalies "
        "(wrong count, arrangement, missing/extra object) leave every surface looking normal, "
        "so a local model is near-blind. The question: does the router escalate the logical "
        "ones to the cloud, and does hybrid then catch what local-only misses?",
        "",
        "| Category | Kind | n | Escalation rate | Local recall | Hybrid recall |",
        "|---|---|---|---|---|---|",
    ]
    agg = {"logical": [], "structural": []}
    for cat, kinds in rows.items():
        for kind in ("logical", "structural"):
            r = kinds.get(kind)
            if not r:
                continue
            lines.append(
                f"| {cat} | {kind} | {r['n']} | {r['escalation_rate']:.0%} | "
                f"{r['local_recall']:.2f} | {r['hybrid_recall']:.2f} |"
            )
            agg[kind].append(r)

    def mean(kind, key):
        vals = [r[key] for r in agg[kind] if r[key] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    lines += [
        "",
        "**Aggregate.**",
        f"- Logical anomalies: escalation {mean('logical','escalation_rate'):.0%}, "
        f"local recall {mean('logical','local_recall'):.2f} -> hybrid "
        f"{mean('logical','hybrid_recall'):.2f}.",
        f"- Structural anomalies: escalation {mean('structural','escalation_rate'):.0%}, "
        f"local recall {mean('structural','local_recall'):.2f} -> hybrid "
        f"{mean('structural','hybrid_recall'):.2f}.",
        "",
        "Read the escalation rate on the logical row: a high rate means the router recognizes "
        "its own uncertainty on logical anomalies and defers them to the reasoning model; a "
        "low rate means the local model is confidently wrong and the router lets them through, "
        "an honest limitation that motivates a logical-anomaly-aware uncertainty signal.",
    ]
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
