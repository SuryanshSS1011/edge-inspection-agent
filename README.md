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
| M1 | Cloud MCP tool on Alibaba Cloud | ✅ code done; deploy + live exit-check pending creds |
| M2 | ONNX perception + temperature calibration | ✅ logic done + tested; live fit needs a model + dataset |
| M4 | Actuation + local log + full-mode loop | stub |
| M5 | Privacy filter + boundary log | stub |
| M6 | Network controller + outbox | stub |
| M7 | Eval harness + results table | stub |
| M8 | Demo capture + deck | stub |

Build order and exit checks: `../EdgeAgent_Implementation_Plan.md` §5.

## Layout
```
edge/    perception, router, privacy, network, outbox, actuation, store, orchestrator
cloud/   Qwen-VL MCP tool + Alibaba Cloud deploy config
eval/    MVTec loader, metrics, run scripts, results-table generator
demo/    network toggle, demo runner
docs/    architecture, router derivation, privacy model
tests/   router unit tests (priority)
```

## Quickstart
```bash
pip install -r requirements.txt        # recommended: Python 3.11 (code is 3.9-compatible)
python -m pytest tests/ -q             # run the router tests
python -m edge.app live --config config.yaml   # full loop lands in M4
```

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
