"""3D-AD evaluation: the cost router on the point-cloud modality (proof of full function).

The whole pipeline is modality-agnostic because the router only sees a calibrated p. This
runs the SAME unsupervised anomaly-score setup as the 2D/LOCO experiments, but on organized
point clouds via the frozen PointNet encoder: fit the normal (good) cloud distribution, score
each test cloud by Mahalanobis distance from it, calibrate to p (standardized by the good
distances so clean clouds pass locally), and report local-only vs. hybrid cost-weighted
recall per category. Nothing in the router, privacy filter, or outbox changes; only the
feature extractor is swapped for the 3D one.

    python -m eval.run_3d --data /scratch/sss6371/datasets/mvtec3d
"""

import argparse

import numpy as np  # type: ignore

from edge.router import Costs, escalation_band
from eval.datasets_3d import MVTEC3D_CATEGORIES, load_mvtec3d

COSTS = Costs(C_FN=100.0, C_FP=5.0, C_cloud=2.0, residual_cloud_error=0.3)


def _fit_normal(feats):
    mu = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False)
    d = cov.shape[0]
    cov = 0.9 * cov + 0.1 * np.trace(cov) / d * np.eye(d)
    return mu, np.linalg.pinv(cov)


def _maha(feats, mu, inv):
    diff = feats - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, inv, diff))


def _category(data, category):
    """Return per-test-cloud (p, label, rgb_path) using the PointNet distance-from-normal
    score. rgb_path (the paired colour image) lets the real cloud be queried on escalation,
    since Qwen-VL reads images, not raw point clouds."""
    from sklearn.linear_model import LogisticRegression
    from eval.pointcloud_features import extract_many

    train = [xyz for xyz, _rgb, _lbl, _c in load_mvtec3d(data, category, "train")]
    test = list(load_mvtec3d(data, category, "test"))
    if len(train) < 5 or not test:
        return None

    Fg = extract_many(train).astype(np.float64)
    mu, inv = _fit_normal(Fg)

    test_xyz = [xyz for xyz, _r, _l, _c in test]
    test_rgb = [rgb for _x, rgb, _l, _c in test]
    labels = np.array([lbl for _x, _r, lbl, _c in test])
    scores = _maha(extract_many(test_xyz).astype(np.float64), mu, inv)

    # Disjoint calibration slice; standardize by the GOOD distances (clean clouds -> low p).
    rng = np.random.default_rng(0)
    idx = np.arange(len(test)); rng.shuffle(idx)
    g = [i for i in idx if labels[i] == 0]; a = [i for i in idx if labels[i] == 1]
    cal = g[: len(g) // 2] + a[: len(a) // 2]
    ev = g[len(g) // 2:] + a[len(a) // 2:]
    s_cal = scores[cal].reshape(-1, 1); y_cal = labels[cal]
    gs = s_cal[y_cal == 0]
    gm = gs.mean() if gs.size else s_cal.mean()
    gstd = (gs.std() if gs.size else s_cal.std()) + 1e-9
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    if len(set(y_cal.tolist())) < 2:
        return None
    lr.fit((s_cal - gm) / gstd, y_cal)

    def to_p(s):
        return float(lr.predict_proba(((np.array([[s]]) - gm) / gstd))[0, 1])

    return [(to_p(scores[i]), int(labels[i]), test_rgb[i]) for i in ev]


def _cloud_says_defect(rgb_path, category):
    """Real Qwen verdict on the paired RGB image of an escalated cloud (Qwen reads images)."""
    if not rgb_path:
        return False
    import base64
    import os
    server = os.environ.get("EDGE_CLOUD_URL", "").strip()
    if not server:
        return False
    from edge.cloud_client import CloudClient
    try:
        with open(rgb_path, "rb") as f:
            roi = base64.b64encode(f.read()).decode("ascii")
        out = CloudClient(server, timeout_s=45.0).diagnose(
            roi_png_b64=roi, context={"category": category})
        return bool(out.get("defect_present"))
    except Exception as exc:  # noqa: BLE001
        print(f"    [cloud call failed: {exc}]", flush=True)
        return False


def _analyze(items, category="", real_cloud=False):
    band = escalation_band(COSTS)
    lo, hi = band if band else (1.0, 0.0)
    pstar = COSTS.C_FP / (COSTS.C_FP + COSTS.C_FN)

    def in_band(p):
        return lo <= p <= hi

    defects = [(p, l, r) for p, l, r in items if l == 1]
    n = len(items)
    escalated = sum(1 for p, _, _ in items if in_band(p))
    local = sum(1 for p, _, _ in defects if p >= pstar and not in_band(p))
    hybrid = 0
    for p, _, rgb in defects:
        if in_band(p):
            if real_cloud and _cloud_says_defect(rgb, category):
                hybrid += 1
        elif p >= pstar:
            hybrid += 1
    return {
        "n": n,
        "escalation_rate": escalated / n if n else 0.0,
        "local_recall": (local / len(defects)) if defects else None,
        "hybrid_recall": (hybrid / len(defects)) if (defects and real_cloud) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="MVTec 3D-AD root (extracted)")
    parser.add_argument("--categories", nargs="+", default=MVTEC3D_CATEGORIES)
    parser.add_argument("--real-cloud", action="store_true",
                        help="measure hybrid recall by sending each escalated cloud's paired "
                             "RGB image to the real qwen3-vl-plus endpoint (needs EDGE_CLOUD_URL)")
    parser.add_argument("--out", default="eval/results_3d.md")
    args = parser.parse_args()

    rows = {}
    for cat in args.categories:
        print(f"[{cat}] scoring point clouds...")
        items = _category(args.data, cat)
        if items is None:
            print(f"  [skip {cat}]")
            continue
        rows[cat] = _analyze(items, category=cat, real_cloud=args.real_cloud)
        r = rows[cat]
        lr = "n/a" if r["local_recall"] is None else f"{r['local_recall']:.2f}"
        hr = "n/a" if r["hybrid_recall"] is None else f"{r['hybrid_recall']:.2f}"
        print(f"  n={r['n']} escalated={r['escalation_rate']:.0%} local={lr} hybrid={hr}")

    _write(rows, args.out)


def _write(rows, out):
    lines = [
        "# MVTec 3D-AD: the cost router on point clouds (proof of full function)",
        "",
        "Same unsupervised anomaly-score setup as the 2D experiments, but the frozen feature "
        "extractor is a PointNet encoder over organized point clouds instead of an image "
        "backbone. The router, privacy filter, and outbox are unchanged; only the modality "
        "differs. Distance from the normal (good) cloud distribution is calibrated to p and "
        "banded exactly as in 2D, so this demonstrates the cost-routing decision is genuinely "
        "modality-agnostic.",
        "",
        "| Category | n | Escalation rate | Local recall | Hybrid recall |",
        "|---|---|---|---|---|",
    ]
    lr_all, hr_all, esc_all = [], [], []
    for cat, r in rows.items():
        lr = "n/a" if r["local_recall"] is None else f"{r['local_recall']:.2f}"
        hr = "n/a" if r["hybrid_recall"] is None else f"{r['hybrid_recall']:.2f}"
        lines.append(f"| {cat} | {r['n']} | {r['escalation_rate']:.0%} | {lr} | {hr} |")
        if r["local_recall"] is not None:
            lr_all.append(r["local_recall"])
            esc_all.append(r["escalation_rate"])
            if r["hybrid_recall"] is not None:
                hr_all.append(r["hybrid_recall"])
    if lr_all:
        measured = len(hr_all) > 0
        agg = [
            "",
            f"**Aggregate across {len(lr_all)} categories:** the PointNet local model catches "
            f"{np.mean(lr_all):.2f} of defects and escalates {np.mean(esc_all):.0%} of clouds. "
            "The router runs on 3D point clouds with ZERO change to the orchestration, privacy "
            "filter, or outbox, only the feature extractor differs. That is the modality-"
            "agnostic claim made concrete: the same cost inequality bands a calibrated p "
            "whether it comes from an image or a point cloud.",
        ]
        if measured:
            agg.append(
                f"\nWith real qwen3-vl-plus verdicts on the escalated clouds' paired RGB "
                f"images, hybrid recall is {np.mean(hr_all):.2f} (measured, not modeled)."
            )
        else:
            agg.append(
                "\nHybrid recall is left unmeasured here (run with --real-cloud to send each "
                "escalated cloud's paired RGB image to the reasoner); this run reports the real "
                "local model and escalation behavior only."
            )
        lines += agg
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
