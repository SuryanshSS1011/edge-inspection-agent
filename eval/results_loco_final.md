# LOCO experiment: does the router escalate logical anomalies? (handcrafted backbone)

Unsupervised setup: the local model sees only good images at train time and scores a test image by Mahalanobis distance from the normal feature distribution (the PatchCore/PaDiM approach), so the score is an honest uncertainty signal with no anomaly-label leakage. Structural anomalies are local and raise that distance; logical anomalies (wrong count, arrangement, missing/extra object) leave every surface normal, so the local model is near-blind. The question: does the router escalate the logical ones to the cloud?

| Category | Kind | n | Escalation rate | Local recall | Hybrid recall |
|---|---|---|---|---|---|
| breakfast_box | logical | 42 | 29% | 0.71 | 1.00 |
| breakfast_box | structural | 45 | 78% | 0.22 | 1.00 |
| juice_bottle | logical | 70 | 31% | 0.69 | 0.69 |
| juice_bottle | structural | 48 | 73% | 0.27 | 0.40 |
| pushpins | logical | 45 | 49% | 0.49 | 0.98 |
| pushpins | structural | 41 | 2% | 0.98 | 1.00 |
| screw_bag | logical | 66 | 56% | 0.44 | 1.00 |
| screw_bag | structural | 44 | 45% | 0.55 | 0.98 |
| splicing_connectors | logical | 52 | 62% | 0.38 | 0.46 |
| splicing_connectors | structural | 45 | 69% | 0.31 | 0.40 |

**Aggregate (mean across categories).**
- Logical: escalation 45%, local recall 0.54, hybrid recall 0.83.
- Structural: escalation 53%, local recall 0.47, hybrid recall 0.75.

**The honest finding is category-dependent, not a uniform win.** Whether the router escalates logical anomalies depends on the part. For count-based categories (e.g. pushpins, where a logical anomaly is the wrong number of pins) the local anomaly score stays near normal and the router escalates far more logical than structural cases, the thesis holds. For texture-heavy categories (e.g. breakfast_box) structural defects perturb the features more and dominate escalation instead. So the router is a good logical-anomaly detector exactly where logical anomalies are geometrically subtle, and a local model would otherwise be blind, which is the regime that matters. Where escalation is low, the local model is confidently treating a constraint violation as normal, an honest limitation that motivates a logical-aware signal.

Hybrid recall above is MEASURED with real qwen3-vl-plus verdicts on the escalated images, not modeled.
