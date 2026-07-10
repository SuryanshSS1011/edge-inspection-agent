# Tollgate

**An edge inspection agent that decides at the gate. It passes a frame locally when confident, pays the cloud toll only when it's worth it, and keeps working when the cloud is gone.**

![Architecture](assets/architecture-dark.png)

## The idea

Most inspection systems bolt on privacy, offline support, and cloud orchestration as separate features. Tollgate derives all three from one decision, which is to **route frames to the cloud based on cost, not confidence**. A confident part passes the gate for free (decided locally); an uncertain one pays the toll (escalates to the cloud).

```
band = [T/C_FN,  1 - T/C_FP]     T = C_cloud + ε
```

- **In-band** (uncertain): escalate only the cropped ROI to qwen3-vl-plus. Zero raw frames, zero PII.
- **Out-of-band** (confident): decide locally. No cloud call needed.
- **Offline + in-band**: conservative local reject, queue to outbox. Line never stops.
- **Reconnect**: outbox drains concurrently, cloud verdict back-fills the record.

One inequality. Three requirements.

## Results

Measured on real MVTec industrial data across six categories:

| Mode | Accuracy | Cost | Edge latency p50 / p99 | Cloud latency p50 / mean / p99 | PII egress |
|---|---|---|---|---|---|
| Cloud-only | 0.992 | 100% | 0.2 ms / 0.3 ms | 3.7 s / 4.8 s / 12.4 s | 0 |
| **Hybrid (ours)** | **0.988** | **57%** | **0.2 ms / 0.3 ms** | **3.7 s / 4.8 s / 12.4 s** | **0** |
| Local-only | 0.951 | 0% | 0.2 ms / 0.3 ms | n/a | 0 |

Cloud latency measured live on 12 real Qwen-VL calls (100% accuracy). p99 is a single cold-start spike; p50 and mean reflect steady-state. Edge latency covers the full frame pipeline: MobileNetV2 + classifier + routing decision. Hybrid cloud calls cover the escalated band only.

- Six-category robustness: **0.969 ± 0.015** (spread tightened as categories were added)
- Backbone ablation: hybrid delta **-0.024 to +0.023** across backbone swaps. The router absorbs local-model variance.
- **168 tests** green

## Stack

- **Edge**: Python 3.11, ONNX (MobileNetV2), temperature-scaled logistic classifier, SQLite outbox
- **Cloud**: qwen3-vl-plus via DashScope, served over HTTP from a Docker container on Alibaba SAS
- **Interface**: MCP stdio tool + HTTP (`/healthz`, `/diagnose`), same code path

```
edge/    perception · router · privacy · orchestrator · outbox · actuation · store · drift
cloud/   qwen3-vl-plus reasoning server (HTTP + MCP)
eval/    MVTec loader · metrics · harness · result scripts
demo/    scripted demo runner · network toggle · video script
tests/   168 unit + integration tests
```

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # 3.11 recommended
pip install -r requirements.txt
pytest tests/ -q                          # 168 tests
```

Three ways to run it. `camera` and `data` run the **same real pipeline** (perception →
router → privacy → cloud → relay → log) and differ only in where frames come from; `demo`
is a separate, deterministic scripted story for the video/presentation.

```bash
# camera runs a real device or phone. It falls back to replaying data/<category>/ if no camera opens,
# so a live run never dies on a missing device:
python -m edge.app camera                                    # laptop built-in cam
python -m edge.app camera --camera auto                      # first cam that opens
python -m edge.app camera --camera http://<phone-ip>:8080/video   # phone over wifi

# data replays a folder of images through the real pipeline (no camera). --limit bounds it:
python -m edge.app data --category bottle --limit 20

# demo runs a scripted 6-beat walkthrough; no camera, no cloud key needed:
python -m edge.app demo
```

Both `camera` and `data` print a per-frame line (`p`, decision, escalated?, bytes, PII,
network mode) and a final audit summary.

**Using a phone as the camera:** install an IP-webcam app (e.g. *IP Webcam* on Android)
or expose any MJPEG/RTSP stream, put the phone on the same wifi, and pass its stream URL to
`--camera`. On iPhone + Mac, Continuity Camera also shows up as a plain capture index
(`--camera auto` or `--camera 1`). Point the edge at the deployed reasoning server with
`--cloud-url http://<server-ip>:8080` (or set `EDGE_CLOUD_URL` in `.env`).

To reproduce the results, download any [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) category to `data/<category>/`, then:

```bash
python -m eval.train_classifier --data data --category bottle
python -m eval.fit_calibration
python -m eval.run_real_eval
python -m eval.run_multi_category
```

## Robustness features

**Calibration drift detection**: a KS-test detector (`edge/drift.py`) watches the rolling confidence window every 50 frames against the reference distribution saved at calibration time. When the KS statistic exceeds the threshold it switches to conservative mode (escalate everything) until the operator recalibrates, preventing stale probabilities from silently breaking the cost inequality.

**Edge-vs-cloud model drift monitoring**: a rolling disagreement tracker compares local edge decisions against cloud diagnoses on every escalated frame. When disagreement exceeds 20% over the last 100 escalated frames, the orchestrator logs a warning that the edge model and VLM have drifted apart. No ground-truth labels required.

**API backpressure handling**: the cloud client retries 429 and 5xx responses with exponential backoff and full jitter (2–30 s, up to 3 attempts via `tenacity`), so outbox drain after an extended outage doesn't flood the API and trigger further rate-limiting.

**Concurrent outbox drain**: on reconnect, queued escalations are sent in parallel with a `ThreadPoolExecutor` (4 workers) rather than sequentially, keeping drain time bounded after long offline periods.

**Privacy anonymization**: skin-tone regions within the escalated ROI are blurred via an HSV mask before encoding, so human hands or faces entering the inspection zone never leave the device as recoverable biometric data.

**Hot-reload cost configuration**: `LiveConfig` re-reads `config.yaml` on mtime change, so operators can tune `C_FN`/`C_FP` for a shift change or a new part family without restarting the process. The escalation band updates automatically.

## Config

All costs live in `config.yaml`. The escalation band is **derived** from them, never hand-tuned.

```yaml
costs:
  C_FN: 100.0   # missed defect
  C_FP: 5.0     # false alarm
  C_cloud: 2.0  # one escalation
  residual_cloud_error: 0.3
```
