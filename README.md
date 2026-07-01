# EdgeAgent — cost-aware industrial visual-inspection agent

**An edge inspection agent that knows when it needs the cloud — and keeps working when
the cloud is gone.** It perceives on-device, escalates only the hard cases to Qwen-VL
cloud reasoning, acts locally, with provable privacy boundaries and principled offline
degradation.

## What makes this different
The three track requirements — edge-cloud orchestration, privacy, graceful offline
degradation — are usually three bolted-on features. Here they fall out of **one design
decision**: a **cost-aware router** whose escalation threshold is derived from
*asymmetric operator cost* (a missed defect costs far more than a needless cloud call),
**not** from raw model confidence. See [`docs/router_derivation.md`](docs/router_derivation.md).

Measured on real MVTec data (`bottle`):

| | Cost-weighted recall | Bytes → cloud/item | Cloud $/1k | PII out |
|---|---|---|---|---|
| Cloud-everything | 0.998 | 54 | $2000 | 0 |
| Local-only | 0.908 | 0 | $0 | 0 |
| **Hybrid (ours)** | **0.988** | **31** | **$1148** | **0** |

Hybrid recovers near-cloud accuracy at **57% of the bytes/cost** with **zero PII egress**,
and holds at **0.969 ± 0.015 across six categories**. Cloud reasoning is *measured*, not
modeled: 12 live Qwen-VL calls → **100% accuracy**, p50/p99 latency **3.7s / 12.4s** (that
latency is exactly why you escalate only the uncertain band). **115 tests** green.

## Status
| Milestone | What | State |
|---|---|---|
| M3 | Cost-aware router + unit tests (core IP) | ✅ done, 14 tests green |
| M1 | Cloud reasoning (MCP + HTTP) | ✅ validated live (Qwen-VL, 100% acc / 12 calls); deploy dry-run passes, FC hosting blocked on Alibaba account provisioning |
| M2 | ONNX perception + temperature calibration | ✅ done; real fit on bottle data, ECE 0.0585→0.0503 |
| M4 | Actuation + local log + full-mode loop | ✅ done; cloud-independent, 10 tests |
| M5 | Privacy filter + boundary log | ✅ done; measured zero PII egress, 9 tests |
| M6 | Network controller + outbox | ✅ done; defer + reconnect-sync, 16 tests |
| M7 | Eval harness + results table | ✅ done; real-data table (hybrid 0.988 vs local 0.908) |
| M8 | Demo capture + deck | ✅ scripted demo + deck done; record with hardware |

Build order and exit checks: `../EdgeAgent_Implementation_Plan.md` §5.

## Layout
```
edge/    perception, router, privacy, network, outbox, actuation, store, orchestrator
cloud/   Qwen-VL reasoning: MCP-native tool + HTTP deploy for Alibaba Function Compute
eval/    MVTec loader, metrics, run scripts, results-table generator
demo/    network toggle, demo runner
docs/    architecture, router derivation, privacy model
tests/   router unit tests (priority)
```

## Quickstart
```bash
pip install -r requirements.txt        # recommended: Python 3.11 (code is 3.9-compatible)
python -m pytest tests/ -q             # run the test suite
python -m edge.app live --config config.yaml   # run the full inspection loop
```

## Reproduce on real data (MVTec bottle)
Extract the dataset to `data/bottle/...`, then:
```bash
python -m eval.train_classifier --data data --category bottle   # -> models/classifier.onnx + splits.json
python -m eval.fit_calibration                                  # -> models/temperature.json (+ ECE before/after)
python -m eval.run_real_eval                                    # -> the results table on held-out eval images
python -m eval.run_multi_category                               # -> six-category robustness table

# Measure the real cloud (needs DASHSCOPE_API_KEY; free-tier-conscious, capped):
python -m eval.measure_cloud --max-calls 12                     # -> real accuracy + latency
```
The classifier is a deliberately modest LogisticRegression on lightweight features — the
router needs calibrated *uncertainty*, not local accuracy, so the cloud has real work.
MobileNet is a drop-in upgrade behind the same ONNX interface.

## Deploy the cloud tool (Alibaba Function Compute)
```bash
python -m cloud.fc_deploy.preflight            # local deploy-readiness checks
python -m cloud.fc_deploy.dry_run              # exercise the exact deployed path (1 live call)
cd cloud/fc_deploy && set -a && source ../../.env && set +a && s deploy
```
See [`cloud/README.md`](cloud/README.md) for the full deploy + voucher steps.

## Config
All costs and thresholds live in `config.yaml`. Thresholds are **derived** from the
costs (never hand-set) — edit `C_FN`, `C_FP`, `C_cloud`, `residual_cloud_error` and the
escalation band recomputes.
```yaml
costs:
  C_FN: 100.0   # false negative (defect shipped)
  C_FP: 5.0     # false positive (good part rejected)
  C_cloud: 2.0  # one escalation
  residual_cloud_error: 0.3
```
