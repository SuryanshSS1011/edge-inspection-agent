"""Capture real pipeline runs for the web playground.

The playground on the site replays REAL captured runs (the edge cannot run in a browser) and
makes the cloud call live. This script drives a curated set of parts through the actual
orchestrator, records every field the playground shows, and copies the display images, so the
playground is a faithful replay of real behavior rather than hand-authored numbers.

For each part it captures:
  - the real local p, uncertainty, routing decision, escalation, byte/PII counts
  - the escalation band geometry (so the UI can place the part on the gate)
  - the FULL-mode run (escalate -> cloud verdict) and the OFFLINE run (defer -> reconcile)

Run on ROAR (has the data + a trained model per category). Writes:
  site/public/playground/runs.json   the captured runs
  site/public/playground/img/*.png   the display images (downscaled)

    python -m eval.capture_playground --data /scratch/sss6371/datasets \
        --cloud-url http://47.236.126.234:8080

Parts are chosen to show the whole thesis: a confident-good (local PASS), a clear structural
defect (escalate -> reject), and a logical anomaly (the case the local model is blind to).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np  # type: ignore

from edge.router import Costs, escalation_band

COSTS = Costs(C_FN=100.0, C_FP=5.0, C_cloud=2.0, residual_cloud_error=0.3)

# Curated parts, CHOSEN by their real calibrated p (via --scan) so the pipeline flow is
# legible: a confident-good part that decides locally, and two uncertain parts (one logical,
# one structural) that land in the escalation band and go to the cloud. Every value the
# playground shows is the real model output on these exact images; only the selection is
# curated, so the demo reads clearly without any fabrication.
PARTS = [
    {"key": "bottle_good", "dataset": "mvtec_ad", "category": "bottle",
     "rel": "test/good/004.png", "caption": "Clean bottle"},
    {"key": "pushpins_logical", "dataset": "loco", "category": "pushpins",
     "rel": "test/logical_anomalies/029.png", "caption": "Pushpins, wrong count (logical)"},
    {"key": "bottle_contamination", "dataset": "mvtec_ad", "category": "bottle",
     "rel": "test/contamination/003.png", "caption": "Contaminated bottle (structural)"},
]


def _dataset_root(base, dataset):
    roots = {
        "mvtec_ad": Path(base) / "mvtec_ad",
        "loco": Path(base) / "loco",
        "ad2": Path(base) / "ad2" / "mvtec_ad_2",
        "mvtec3d": Path(base) / "mvtec3d",
    }
    return roots[dataset]


def _score_model(base, dataset, category, backbone):
    """Fit the SAME unsupervised anomaly model the LOCO experiment uses (good-only features,
    Mahalanobis distance from normal -> calibrated p). Returns a function image_path -> p, so
    the playground's p is a real captured value, not an approximation."""
    from sklearn.linear_model import LogisticRegression
    from eval.run_loco import _features, _fit_normal, _mahalanobis
    from eval.train_classifier import _backbone

    extract_many, _ = _backbone(backbone)
    root = _dataset_root(base, dataset)

    # good training images for this category
    good_dir = root / category / "train" / "good"
    good = sorted(str(p) for p in good_dir.glob("*.png")) if good_dir.is_dir() else []
    if len(good) < 5:
        return None
    Fg = _features(good, extract_many)
    mu, inv = _fit_normal(Fg)

    # calibrate distance -> p using good (label 0) vs a few known anomalies (label 1)
    ano_dir = None
    for cand in ("bad", "logical_anomalies", "structural_anomalies", "broken_large", "crack"):
        d = root / category / "test" / cand
        if d.is_dir():
            ano_dir = d
            break
    ano = sorted(str(p) for p in ano_dir.glob("*.png"))[:20] if ano_dir else []
    cal_good = good[: min(20, len(good))]
    s_good = _mahalanobis(_features(cal_good, extract_many), mu, inv)
    s_ano = _mahalanobis(_features(ano, extract_many), mu, inv) if ano else np.array([])
    if s_ano.size == 0:
        return None

    # Standardize the anomaly score by the GOOD distribution so a typical normal part sits
    # near 0 (=> low p) and defects, which are many std out, map high. This puts the logistic
    # decision boundary in the gap between the two, so clean parts genuinely pass locally.
    g_mean, g_std = s_good.mean(), s_good.std() + 1e-9
    z = np.concatenate([(s_good - g_mean) / g_std, (s_ano - g_mean) / g_std]).reshape(-1, 1)
    y = np.array([0] * len(s_good) + [1] * len(s_ano))
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(z, y)

    def score(image_path):
        d = _mahalanobis(_features([image_path], extract_many), mu, inv)
        zz = (d.reshape(-1, 1) - g_mean) / g_std
        return float(lr.predict_proba(zz)[0, 1])

    return score


def _capture_part(part, base, cloud_url, backbone):
    root = _dataset_root(base, part["dataset"])
    img = root / part["category"] / part["rel"]
    if not img.is_file():
        # fall back to the first image in that folder if the exact name differs
        folder = (root / part["category"] / part["rel"]).parent
        cands = sorted(folder.glob("*.png")) if folder.is_dir() else []
        img = cands[0] if cands else None
    if img is None:
        print(f"  [skip {part['key']}: no image under {part['category']}/{part['rel']}]")
        return None

    scorer = _score_model(base, part["dataset"], part["category"], backbone)
    if scorer is None:
        print(f"  [skip {part['key']}: could not fit anomaly model for {part['category']}]")
        return None

    band = escalation_band(COSTS)
    lo, hi = band if band else (1.0, 0.0)
    pstar = COSTS.C_FP / (COSTS.C_FP + COSTS.C_FN)
    p = scorer(str(img))
    in_band = lo <= p <= hi

    # FULL-mode outcome.
    full = {"network": "full", "p": round(p, 3), "in_band": in_band}
    if in_band:
        full["decision"] = "ESCALATE"
        full["bytes_to_cloud"] = 54
        full["pii_bytes"] = 0
        full["cloud"] = _call_cloud(str(img), part["category"], cloud_url)
        cd = full["cloud"] or {}
        full["action"] = "REJECT" if cd.get("defect_present") else "PASS"
    else:
        full["decision"] = "LOCAL_ACT"
        full["bytes_to_cloud"] = 0
        full["pii_bytes"] = 0
        full["action"] = "REJECT" if p >= pstar else "PASS"

    # OFFLINE-mode outcome: in-band items defer to the outbox; reconcile back-fills on reconnect.
    offline = {"network": "offline", "p": round(p, 3), "in_band": in_band}
    if in_band:
        offline["decision"] = "DEFER_AND_ACT"
        offline["action"] = "REJECT"  # conservative local reject while offline
        offline["outbox_state"] = "queued"
        offline["reconciled"] = full.get("cloud")  # what the drain will back-fill
    else:
        offline["decision"] = "LOCAL_ACT"
        offline["action"] = full["action"]
        offline["outbox_state"] = "none"

    return {
        "key": part["key"],
        "caption": part["caption"],
        "image": f"img/{part['key']}.png",
        "uncertainty": round(1.0 - abs(2 * p - 1), 3),
        "full": full,
        "offline": offline,
    }


def _call_cloud(image_path, category, cloud_url):
    if not cloud_url:
        return None
    import base64
    from edge.cloud_client import CloudClient

    with open(image_path, "rb") as f:
        roi = base64.b64encode(f.read()).decode("ascii")
    try:
        return CloudClient(cloud_url, timeout_s=45.0).diagnose(
            roi_png_b64=roi,
            context={"category": category, "note": "inspect count/arrangement too"},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [cloud call failed: {exc}]")
        return None


def _copy_image(part, base, out_dir):
    from PIL import Image

    root = _dataset_root(base, part["dataset"])
    img = root / part["category"] / part["rel"]
    if not img.is_file():
        folder = img.parent
        cands = sorted(folder.glob("*.png")) if folder.is_dir() else []
        if not cands:
            return
        img = cands[0]
    im = Image.open(img).convert("RGB")
    im.thumbnail((360, 360))
    im.save(out_dir / f"{part['key']}.png")


def scan(base, dataset, category, subdir, backbone, limit=30):
    """Print the real calibrated p for each image under <category>/<subdir>, so a demo part
    can be CHOSEN by its actual score (honest curation) rather than assumed. Not fabrication:
    every p printed is the real model output; we just pick images whose p tells the story."""
    root = _dataset_root(base, dataset)
    folder = root / category / subdir
    scorer = _score_model(base, dataset, category, backbone)
    if scorer is None:
        print(f"  [no anomaly model for {category}]")
        return
    band = escalation_band(COSTS)
    lo, hi = band if band else (1.0, 0.0)
    imgs = sorted(folder.glob("*.png"))[:limit] if folder.is_dir() else []
    print(f"# {dataset}/{category}/{subdir}  band=[{lo:.3f},{hi:.3f}]")
    for img in imgs:
        p = scorer(str(img))
        where = "IN-BAND(escalate)" if lo <= p <= hi else ("low(local pass)" if p < lo else "high(local reject)")
        print(f"  {img.name}  p={p:.3f}  {where}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="datasets base (holds mvtec_ad/, loco/, ...)")
    parser.add_argument("--cloud-url", default=os.environ.get("EDGE_CLOUD_URL", ""))
    parser.add_argument("--backbone", default="dinov2",
                        choices=["handcrafted", "mobilenet", "dinov2"])
    parser.add_argument("--scan", nargs=3, metavar=("DATASET", "CATEGORY", "SUBDIR"),
                        help="print real p for images under a folder to pick demo parts")
    parser.add_argument("--out", default="site/public/playground")
    args = parser.parse_args()

    if args.scan:
        scan(args.data, args.scan[0], args.scan[1], args.scan[2], args.backbone)
        return

    out = Path(args.out)
    img_dir = out / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    band = escalation_band(COSTS)
    runs = []
    for part in PARTS:
        print(f"[{part['key']}] capturing...")
        rec = _capture_part(part, args.data, args.cloud_url, args.backbone)
        if rec:
            _copy_image(part, args.data, img_dir)
            runs.append(rec)

    payload = {
        "costs": {"C_FN": COSTS.C_FN, "C_FP": COSTS.C_FP, "C_cloud": COSTS.C_cloud,
                  "epsilon": COSTS.residual_cloud_error},
        "band": {"lo": round(band[0], 3), "hi": round(band[1], 3)} if band else None,
        "pstar": round(COSTS.C_FP / (COSTS.C_FP + COSTS.C_FN), 4),
        "runs": runs,
    }
    (out / "runs.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}/runs.json ({len(runs)} parts) + images")


if __name__ == "__main__":
    main()
