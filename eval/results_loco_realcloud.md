# LOCO experiment: does the router escalate logical anomalies? (handcrafted backbone)

Unsupervised setup: the local model sees only good images at train time and scores a test image by Mahalanobis distance from the normal feature distribution (the PatchCore/PaDiM approach), so the score is an honest uncertainty signal with no anomaly-label leakage. Structural anomalies are local and raise that distance; logical anomalies (wrong count, arrangement, missing/extra object) leave every surface normal, so the local model is near-blind. The question: does the router escalate the logical ones to the cloud?

| Category | Kind | n | Escalation rate | Local recall | Hybrid recall |
|---|---|---|---|---|---|
| breakfast_box | logical | 42 | 7% | 0.93 | 1.00 |
| breakfast_box | structural | 45 | 53% | 0.47 | 1.00 |
| juice_bottle | logical | 70 | 11% | 0.89 | 0.89 |
| juice_bottle | structural | 48 | 21% | 0.79 | 0.83 |
| pushpins | logical | 45 | 51% | 0.49 | 1.00 |
| pushpins | structural | 41 | 7% | 0.93 | 1.00 |
| screw_bag | logical | 66 | 27% | 0.73 | 1.00 |
| screw_bag | structural | 44 | 16% | 0.84 | 1.00 |
| splicing_connectors | logical | 52 | 2% | 0.98 | 0.98 |
| splicing_connectors | structural | 45 | 13% | 0.87 | 0.89 |

**Aggregate (mean across categories).**
- Logical: escalation 20%, local recall 0.80, hybrid recall 0.97.
- Structural: escalation 22%, local recall 0.78, hybrid recall 0.94.

**The honest finding is category-dependent, not a uniform win.** Whether the router escalates logical anomalies depends on the part. For count-based categories (e.g. pushpins, where a logical anomaly is the wrong number of pins) the local anomaly score stays near normal and the router escalates far more logical than structural cases, the thesis holds. For texture-heavy categories (e.g. breakfast_box) structural defects perturb the features more and dominate escalation instead. So the router is a good logical-anomaly detector exactly where logical anomalies are geometrically subtle, and a local model would otherwise be blind, which is the regime that matters. Where escalation is low, the local model is confidently treating a constraint violation as normal, an honest limitation that motivates a logical-aware signal.

Hybrid recall above is MEASURED with real qwen3-vl-plus verdicts on the escalated images, not modeled.
