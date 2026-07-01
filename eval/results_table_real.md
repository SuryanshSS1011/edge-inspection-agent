# Results — real bottle data (local model + fitted calibration)

Local `p` from the trained ONNX classifier over 61 held-out eval images (disjoint from train/calibration). Cloud is modeled; the live cloud row needs the deployed endpoint.

| Condition | Cost-weighted recall | p50 / p99 latency (ms) | Bytes → cloud / item | Cloud cost / 1k | PII bytes out |
|---|---|---|---|---|---|
| Cloud-everything (baseline) | 0.998 [0.996–0.999] | 0.2 / 0.4 | 54 | $2000.00 | 0 |
| Local-only | 0.908 [0.908–0.908] | 0.0 / 0.0 | 0 | $0.00 | 0 |
| **Hybrid (ours)** | 0.988 [0.987–0.990] | 0.2 / 0.3 | 31 | $1147.54 | 0 |
| Hybrid — degraded | 0.908 [0.908–0.908] | 0.4 / 0.7 | 0 | $0.00 | 0 |
| Hybrid — offline | 0.908 [0.908–0.908] | 0.4 / 0.5 | 0 | $0.00 | 0 |
| Reconnect / sync (drains queue) | — | — | (batched) | $1032.79 | 0 |

> All "$/1k" and "bytes/item" columns use the same denominator: **per 1000 items inspected**. Per-mode columns measure egress **under that network condition** — offline/degraded show $0 / 0 bytes because the device decides locally with no cloud call. The deferred diagnoses are drained once in the reconnect/sync row (degraded and offline defer the *same* band items, so it's one drain set, not two). The reconnect figure applies a **9% batching assumption** (a batched drain amortizes per-call overhead vs. live one-at-a-time calls); it is modeled, not measured. The conclusion does **not** depend on it: *undiscounted*, the reconnect cost equals live hybrid cost (same band items, same per-call price), so it is ≤ hybrid and far below cloud-everything either way. Deferred diagnoses reconcile the **log**, not the action: the offline decision was already made locally.
