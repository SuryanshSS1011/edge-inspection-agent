"""LOCO experiment: does the cost router escalate LOGICAL anomalies to the cloud?

This is the UNSUPERVISED setup that MVTec anomaly benchmarks are built for: the local model
sees only GOOD images at train time and scores a test image by how far it is from normal.
No anomaly labels leak into training, so the score is an honest uncertainty signal (this is
exactly how PatchCore / PaDiM work: features from a frozen backbone, distance from the normal
feature distribution).

Method per category:
  1. Extract frozen-backbone features on train/good only.
  2. Fit the normal distribution (mean + shrinkage covariance) and score each test image by
     Mahalanobis distance from it. That distance is the anomaly score.
  3. Map score -> calibrated p with a logistic fit on a SMALL held-out mix (some good + some
     anomalies from a disjoint slice of test), so p is a probability the router can band.
  4. On the remaining disjoint eval items, report, split by anomaly KIND (logical/structural):
       - escalation rate (fraction whose p lands in the router's band)
       - local-only vs. hybrid detection rate

The hypothesis: logical anomalies (wrong count/arrangement, missing/extra object) look
locally normal, so their distance-from-normal is modest -> p sits low/in-band -> the router
escalates them to Qwen-VL rather than the local model deciding. Whatever the data shows is
reported honestly.

    python -m eval.run_loco --data /scratch/sss6371/datasets/loco --backbone dinov2
"""

import argparse
import tempfile

import numpy as np  # type: ignore

from edge.router import Costs, escalation_band
from eval.datasets_loco import LOCO_CATEGORIES, load_loco

COSTS = Costs(C_FN=100.0, C_FP=5.0, C_cloud=2.0, residual_cloud_error=0.3)


def _features(paths, extract_many):
    return extract_many(list(paths)).astype(np.float64)


def _fit_normal(feats):
    """Mean + shrinkage covariance of the normal (good) features. Shrinkage keeps the
    covariance invertible when feature dim exceeds the number of good images."""
    mu = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False)
    d = cov.shape[0]
    shrink = 0.1
    cov = (1 - shrink) * cov + shrink * np.trace(cov) / d * np.eye(d)
    inv = np.linalg.pinv(cov)
    return mu, inv


def _mahalanobis(feats, mu, inv):
    diff = feats - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))


def _kinded_scores(data, category, backbone, workdir):
    """Return eval items [(p, label, kind)] using an unsupervised distance-from-normal score.

    Trains the normal model on train/good only. Uses a disjoint slice of test (some good +
    some anomalies) to fit the score->p logistic, and reports on the rest.
    """
    from sklearn.linear_model import LogisticRegression
    from eval.train_classifier import _backbone

    extract_many, _ = _backbone(backbone)

    train_good = [p for p, _, _ in load_loco(data, category, "train")]
    test = list(load_loco(data, category, "test"))

    # 1-2. normal model on good only.
    Fg = _features(train_good, extract_many)
    mu, inv = _fit_normal(Fg)

    # Score every test image.
    test_paths = [p for p, _, _ in test]
    test_kinds = [k for _, _, k in test]
    test_labels = [lbl for _, lbl, _ in test]
    Ft = _features(test_paths, extract_many)
    scores = _mahalanobis(Ft, mu, inv)

    # Disjoint calibration slice: half of each (good / anomaly) fits the logistic; rest is eval.
    idx = np.arange(len(test))
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    good_idx = [i for i in idx if test_labels[i] == 0]
    ano_idx = [i for i in idx if test_labels[i] == 1]
    cal_idx = good_idx[: len(good_idx) // 2] + ano_idx[: len(ano_idx) // 2]
    eval_idx = good_idx[len(good_idx) // 2:] + ano_idx[len(ano_idx) // 2:]

    # 3. score -> p via logistic on the calibration slice (standardize the 1-D score first).
    s_cal = scores[cal_idx].reshape(-1, 1)
    y_cal = np.array([test_labels[i] for i in cal_idx])
    s_mean, s_std = s_cal.mean(), s_cal.std() + 1e-9
    lr = LogisticRegression(max_iter=1000)
    lr.fit((s_cal - s_mean) / s_std, y_cal)

    def to_p(s):
        return float(lr.predict_proba(((np.array([[s]]) - s_mean) / s_std))[0, 1])

    # 4. eval items with kind.
    items = [(to_p(scores[i]), test_labels[i], test_kinds[i]) for i in eval_idx]
    return items


def _analyze(items):
    band = escalation_band(COSTS)
    lo, hi = band if band else (1.0, 0.0)
    pstar = COSTS.C_FP / (COSTS.C_FP + COSTS.C_FN)

    def in_band(p):
        return lo <= p <= hi

    by_kind = {}
    for kind in ("good", "logical", "structural"):
        sub = [(p, lbl) for p, lbl, k in items if k == kind]
        if not sub:
            continue
        defects = [(p, lbl) for p, lbl in sub if lbl == 1]
        escalated = sum(1 for p, _ in sub if in_band(p))
        # local-only detects a defect only if it decides REJECT locally (p >= pstar) AND is
        # not merely deferring. hybrid additionally catches any escalated defect (cloud sees it).
        local_caught = sum(1 for p, _ in defects if p >= pstar and not in_band(p))
        hybrid_caught = sum(1 for p, _ in defects if in_band(p) or p >= pstar)
        by_kind[kind] = {
            "n": len(sub),
            "escalation_rate": escalated / len(sub),
            "local_recall": (local_caught / len(defects)) if defects else None,
            "hybrid_recall": (hybrid_caught / len(defects)) if defects else None,
        }
    return by_kind


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--categories", nargs="+", default=LOCO_CATEGORIES)
    parser.add_argument("--backbone", default="dinov2",
                        choices=["handcrafted", "mobilenet", "dinov2"])
    parser.add_argument("--out", default="eval/results_loco.md")
    args = parser.parse_args()

    rows = {}
    with tempfile.TemporaryDirectory() as wd:
        for cat in args.categories:
            print(f"[{cat}] fitting normal model + scoring...")
            items = _kinded_scores(args.data, cat, args.backbone, wd)
            rows[cat] = _analyze(items)
            for kind in ("logical", "structural"):
                r = rows[cat].get(kind)
                if r:
                    lr = "n/a" if r["local_recall"] is None else f"{r['local_recall']:.2f}"
                    hr = "n/a" if r["hybrid_recall"] is None else f"{r['hybrid_recall']:.2f}"
                    print(f"  {kind:<11} n={r['n']:<4} escalated={r['escalation_rate']:.0%}  "
                          f"local_recall={lr}  hybrid_recall={hr}")

    _write_report(rows, args.backbone, args.out)


def _write_report(rows, backbone, out):
    lines = [
        f"# LOCO experiment: does the router escalate logical anomalies? ({backbone} backbone)",
        "",
        "Unsupervised setup: the local model sees only good images at train time and scores a "
        "test image by Mahalanobis distance from the normal feature distribution (the "
        "PatchCore/PaDiM approach), so the score is an honest uncertainty signal with no "
        "anomaly-label leakage. Structural anomalies are local and raise that distance; "
        "logical anomalies (wrong count, arrangement, missing/extra object) leave every "
        "surface normal, so the local model is near-blind. The question: does the router "
        "escalate the logical ones to the cloud?",
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
            lr = "n/a" if r["local_recall"] is None else f"{r['local_recall']:.2f}"
            hr = "n/a" if r["hybrid_recall"] is None else f"{r['hybrid_recall']:.2f}"
            lines.append(
                f"| {cat} | {kind} | {r['n']} | {r['escalation_rate']:.0%} | {lr} | {hr} |"
            )
            agg[kind].append(r)

    def mean(kind, key):
        vals = [r[key] for r in agg[kind] if r[key] is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    lines += [
        "",
        "**Aggregate.**",
        f"- Logical: escalation {mean('logical','escalation_rate'):.0%}, local recall "
        f"{mean('logical','local_recall'):.2f} -> hybrid {mean('logical','hybrid_recall'):.2f}.",
        f"- Structural: escalation {mean('structural','escalation_rate'):.0%}, local recall "
        f"{mean('structural','local_recall'):.2f} -> hybrid {mean('structural','hybrid_recall'):.2f}.",
        "",
        "A higher logical-than-structural escalation rate, plus a hybrid recall above local "
        "recall on the logical row, is the thesis confirmed: the router recognizes its own "
        "blindness to logical anomalies and defers them to the reasoning model. If logical "
        "escalation is low, that is an honest limitation, since the local model is confidently "
        "treating a wrong-count part as normal, and it motivates a logical-aware signal.",
    ]
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
