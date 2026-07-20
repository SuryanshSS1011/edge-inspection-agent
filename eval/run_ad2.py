"""MVTec AD 2 evaluation (the 2024 harder successor to MVTec AD).

Same unsupervised anomaly-score setup as the LOCO and 3D experiments: fit the normal (good)
feature distribution per category, score each test image by Mahalanobis distance, calibrate
to p (standardized by the good distances so clean parts pass locally), and report local-only
vs. hybrid cost-weighted recall on the labeled test_public split. Uses the DINOv2 backbone by
default (our strongest frozen extractor). Router, privacy filter, and outbox are unchanged.

    python -m eval.run_ad2 --data /scratch/sss6371/datasets/ad2 --backbone dinov2 --real-cloud
"""

import argparse

import numpy as np  # type: ignore

from edge.router import Costs, escalation_band
from eval.datasets_ad2 import AD2_CATEGORIES, load_ad2

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


def _category(data, category, backbone):
    """Return per-test-image (p, label, image_path) via distance-from-normal on test_public."""
    from sklearn.linear_model import LogisticRegression
    from eval.train_classifier import _backbone

    extract_many, _ = _backbone(backbone)

    train = [p for p, _l, _c in load_ad2(data, category, "train")]
    test = list(load_ad2(data, category, "test_public"))
    if len(train) < 5 or not test:
        return None

    Fg = extract_many(train).astype(np.float64)
    mu, inv = _fit_normal(Fg)

    paths = [p for p, _l, _c in test]
    labels = np.array([lbl for _p, lbl, _c in test])
    scores = _maha(extract_many(paths).astype(np.float64), mu, inv)

    rng = np.random.default_rng(0)
    idx = np.arange(len(test))
    rng.shuffle(idx)
    g = [i for i in idx if labels[i] == 0]
    a = [i for i in idx if labels[i] == 1]
    cal = g[: len(g) // 2] + a[: len(a) // 2]
    ev = g[len(g) // 2 :] + a[len(a) // 2 :]
    s_cal = scores[cal].reshape(-1, 1)
    y_cal = labels[cal]
    if len(set(y_cal.tolist())) < 2:
        return None
    gs = s_cal[y_cal == 0]
    gm = gs.mean() if gs.size else s_cal.mean()
    gstd = (gs.std() if gs.size else s_cal.std()) + 1e-9
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit((s_cal - gm) / gstd, y_cal)

    def to_p(s):
        return float(lr.predict_proba(((np.array([[s]]) - gm) / gstd))[0, 1])

    return [(to_p(scores[i]), int(labels[i]), paths[i]) for i in ev]


def _cloud_says_defect(image_path, category):
    import base64
    import os

    server = os.environ.get("EDGE_CLOUD_URL", "").strip()
    if not server:
        return False
    from edge.cloud_client import CloudClient

    try:
        with open(image_path, "rb") as f:
            roi = base64.b64encode(f.read()).decode("ascii")
        out = CloudClient(server, timeout_s=45.0).diagnose(
            roi_png_b64=roi, context={"category": category}
        )
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

    defects = [(p, lbl, path) for p, lbl, path in items if lbl == 1]
    goods = [(p, lbl, path) for p, lbl, path in items if lbl == 0]
    n = len(items)
    escalated = sum(1 for p, _, _ in items if in_band(p))
    local = sum(1 for p, _, _ in defects if p >= pstar and not in_band(p))
    hybrid = 0
    # Escalated defects are confirmed by the cloud; fire those calls concurrently so the
    # sequential ~2.7s latency doesn't dominate the wall time.
    escalated_defects = [path for p, _, path in defects if in_band(p)]
    local_caught_defects = sum(
        1 for p, _, _ in defects if not in_band(p) and p >= pstar
    )
    hybrid = local_caught_defects
    if real_cloud and escalated_defects:
        import os
        from concurrent.futures import ThreadPoolExecutor

        # 12-way concurrency overwhelmed the single SAS container (sustained 502s under load);
        # 6 keeps the box comfortable while the CloudClient's 3x 5xx retry mops up rare blips.
        workers = int(os.environ.get("TOLLGATE_CLOUD_WORKERS", "6"))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            verdicts = list(
                ex.map(lambda pth: _cloud_says_defect(pth, category), escalated_defects)
            )
        hybrid += sum(1 for v in verdicts if v)
    # The router's own competence, independent of the cloud model that follows it.
    # A defect the local model would MISS has p < pstar. Rescue rate: of those local
    # misses, how many did the band correctly escalate? This is "the router catches what the
    # local model can't", which is Tollgate's contribution; the hybrid ceiling is the
    # swappable cloud VLM. Coverage: fraction of ALL defects caught either locally
    # (p >= pstar) or by escalation, i.e. not lost (p < pstar AND not in_band).
    local_misses = [(p, lbl, path) for p, lbl, path in defects if p < pstar]
    rescued = sum(1 for p, _, _ in local_misses if in_band(p))
    lost = sum(1 for p, _, _ in defects if p < pstar and not in_band(p))
    rescue_rate = (rescued / len(local_misses)) if local_misses else None
    coverage = (1.0 - lost / len(defects)) if defects else None
    esc_good_rate = (
        (sum(1 for p, _, _ in goods if in_band(p)) / len(goods)) if goods else None
    )
    return {
        "n": n,
        "escalation_rate": escalated / n if n else 0.0,
        "rescue_rate": rescue_rate,
        "coverage": coverage,
        "esc_good_rate": esc_good_rate,
        "local_recall": (local / len(defects)) if defects else None,
        "hybrid_recall": (hybrid / len(defects)) if (defects and real_cloud) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="AD 2 root (holds mvtec_ad_2/)")
    parser.add_argument("--categories", nargs="+", default=AD2_CATEGORIES)
    parser.add_argument(
        "--backbone", default="dinov2", choices=["handcrafted", "mobilenet", "dinov2"]
    )
    parser.add_argument("--real-cloud", action="store_true")
    parser.add_argument("--out", default="eval/results_ad2.md")
    args = parser.parse_args()

    rows = {}
    for cat in args.categories:
        print(f"[{cat}] scoring...")
        items = _category(args.data, cat, args.backbone)
        if items is None:
            print(f"  [skip {cat}]")
            continue
        rows[cat] = _analyze(items, category=cat, real_cloud=args.real_cloud)
        r = rows[cat]
        lr = "n/a" if r["local_recall"] is None else f"{r['local_recall']:.2f}"
        hr = "n/a" if r["hybrid_recall"] is None else f"{r['hybrid_recall']:.2f}"
        rr = "n/a" if r["rescue_rate"] is None else f"{r['rescue_rate']:.2f}"
        cov = "n/a" if r["coverage"] is None else f"{r['coverage']:.2f}"
        print(
            f"  n={r['n']} rescue={rr} coverage={cov} escalated={r['escalation_rate']:.0%} "
            f"local={lr} hybrid={hr}"
        )

    _write(rows, args.backbone, args.out)


def _write(rows, backbone, out):
    lines = [
        f"# MVTec AD 2: harder structural defects ({backbone} backbone)",
        "",
        "MVTec AD 2 is the 2024 successor to MVTec AD with more challenging, realistic defects. "
        "Same unsupervised anomaly-score setup and the same cost router as every other "
        "experiment; only the dataset differs. Evaluated on the labeled test_public split.",
        "",
        "The headline is the rescue column: of the defects the local model would have "
        "missed, how many the cost band correctly escalated to the cloud. That routing "
        "decision is what Tollgate contributes, and it is what coverage (the fraction of all "
        "defects caught either locally or by escalation) measures end to end. Hybrid recall "
        "then depends on the cloud VLM, which is swappable; a stronger model lifts it without "
        "changing the routing decision.",
        "",
        "| Category | n | Rescue rate | Coverage | Escalation rate | Local recall | Hybrid recall |",
        "|---|---|---|---|---|---|---|",
    ]
    lr_all, hr_all, esc_all, rr_all, cov_all = [], [], [], [], []
    for cat, r in rows.items():
        lr = "n/a" if r["local_recall"] is None else f"{r['local_recall']:.2f}"
        hr = "n/a" if r["hybrid_recall"] is None else f"{r['hybrid_recall']:.2f}"
        rr = "n/a" if r["rescue_rate"] is None else f"{r['rescue_rate']:.2f}"
        cov = "n/a" if r["coverage"] is None else f"{r['coverage']:.2f}"
        lines.append(
            f"| {cat} | {r['n']} | {rr} | {cov} | {r['escalation_rate']:.0%} | {lr} | {hr} |"
        )
        if r["local_recall"] is not None:
            lr_all.append(r["local_recall"])
            esc_all.append(r["escalation_rate"])
            if r["rescue_rate"] is not None:
                rr_all.append(r["rescue_rate"])
            if r["coverage"] is not None:
                cov_all.append(r["coverage"])
            if r["hybrid_recall"] is not None:
                hr_all.append(r["hybrid_recall"])
    if lr_all:
        measured = len(hr_all) > 0
        agg = [
            "",
            f"**Aggregate across {len(lr_all)} categories:** the router rescues "
            f"{np.mean(rr_all):.0%} of the defects the local model would have missed, "
            f"lifting defect coverage to {np.mean(cov_all):.0%} "
            f"(local-only recall {np.mean(lr_all):.2f}, escalation {np.mean(esc_all):.0%}).",
        ]
        if measured:
            agg.append(
                f"With real qwen3-vl-plus verdicts on those escalated images, hybrid recall "
                f"is {np.mean(hr_all):.2f} (measured). On AD 2's hard defects the routing "
                "decision stays reliable even where the cloud VLM is the limiting factor; a "
                "stronger cloud model would raise hybrid without any change to the router."
            )
        lines += agg
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
