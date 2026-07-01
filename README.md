# EdgeAgent — cost-aware industrial visual-inspection agent

An edge inspection agent that **knows when it needs the cloud — and keeps working when
the cloud is gone.** Perceives on-device, escalates only the hard cases to Qwen Cloud
reasoning, acts locally, with provable privacy boundaries and principled offline
degradation.

The differentiator is the **router**: the escalation threshold is derived from
asymmetric operator cost (a miss costs far more than a needless cloud call), not raw
model confidence. See [`docs/router_derivation.md`](docs/router_derivation.md).

## Status
| Milestone | What | State |
|---|---|---|
| M3 | Cost-aware router + unit tests (core IP) | ✅ done, 14 tests green |
| M1 | Cloud reasoning (MCP + HTTP) | ✅ live exit-check passed (real Qwen-VL, ~2.1s); FC deploy pending |
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
```
The classifier is a deliberately modest LogisticRegression on lightweight features — the
router needs calibrated *uncertainty*, not local accuracy, so the cloud has real work.
MobileNet is a drop-in upgrade behind the same ONNX interface.

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
