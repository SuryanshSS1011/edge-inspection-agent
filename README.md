# EdgeAgent

**An edge inspection agent that knows when it needs the cloud — and keeps working when the cloud is gone.**

## The idea

Most inspection systems bolt on privacy, offline support, and cloud orchestration as separate features. EdgeAgent derives all three from one decision: **route frames to the cloud based on cost, not confidence**.

```
band = [T/C_FN,  1 - T/C_FP]     T = C_cloud + ε
```

- **In-band** (uncertain) → escalate only the cropped ROI to Qwen-VL. Zero raw frames, zero PII.
- **Out-of-band** (confident) → decide locally. No cloud call needed.
- **Offline + in-band** → conservative local reject, queue to outbox. Line never stops.
- **Reconnect** → outbox drains, cloud verdict back-fills the record.

One inequality. Three requirements.

## Results

Measured on real MVTec industrial data across six categories:

| Mode | Accuracy | Cost | PII egress |
|---|---|---|---|
| Cloud-only | 0.992 | 100% | 0 |
| **Hybrid (ours)** | **0.988** | **57%** | **0** |
| Local-only | 0.951 | 0% | 0 |

- Six-category robustness: **0.969 ± 0.015** (spread tightened as categories were added)
- Cloud measured live: **100% accuracy**, p99 **12.4 s** — exactly why you escalate only the uncertain band
- Backbone ablation: hybrid Δ **−0.024 to +0.023** across backbone swaps — the router absorbs local-model variance
- **130 tests** green

## Stack

- **Edge**: Python 3.11, ONNX (MobileNetV2), temperature-scaled logistic classifier, SQLite outbox
- **Cloud**: Qwen-VL (`qwen3.7-plus`) via DashScope, served over HTTP from a Docker container on Alibaba SAS
- **Interface**: MCP stdio tool + HTTP (`/healthz`, `/diagnose`) — same code path

## Layout

```
edge/    perception · router · privacy · orchestrator · outbox · actuation · store
cloud/   Qwen-VL reasoning server (HTTP + MCP)
eval/    MVTec loader · metrics · harness · result scripts
demo/    scripted demo runner · network toggle · video script
tests/   130 unit + integration tests
```

## Quickstart

```bash
pip install -r requirements.txt
pytest tests/ -q
python -m edge.app live --config config.yaml
```

## Reproduce the results

Download any [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) category to `data/<category>/`, then:

```bash
python -m eval.train_classifier --data data --category bottle
python -m eval.fit_calibration
python -m eval.run_real_eval
python -m eval.run_multi_category
```

## Config

All costs live in `config.yaml`. The escalation band is **derived** from them — never hand-tuned.

```yaml
costs:
  C_FN: 100.0   # missed defect
  C_FP: 5.0     # false alarm
  C_cloud: 2.0  # one escalation
  residual_cloud_error: 0.3
```
