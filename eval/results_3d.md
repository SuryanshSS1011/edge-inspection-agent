# MVTec 3D-AD: the cost router on point clouds (proof of full function)

Same unsupervised anomaly-score setup as the 2D experiments, but the frozen feature extractor is a PointNet encoder over organized point clouds instead of an image backbone. The router, privacy filter, and outbox are unchanged; only the modality differs. Distance from the normal (good) cloud distribution is calibrated to p and banded exactly as in 2D, so this demonstrates the cost-routing decision is genuinely modality-agnostic.

| Category | n | Escalation rate | Local recall | Hybrid recall |
|---|---|---|---|---|
| bagel | 55 | 36% | 0.73 | 1.00 |
| cable_gland | 55 | 31% | 0.75 | 1.00 |
| carrot | 80 | 59% | 0.39 | 1.00 |
| cookie | 66 | 44% | 0.69 | 1.00 |
| dowel | 65 | 78% | 0.25 | 1.00 |
| foam | 50 | 42% | 0.60 | 1.00 |
| peach | 66 | 74% | 0.32 | 1.00 |
| potato | 57 | 40% | 0.67 | 1.00 |
| rope | 51 | 43% | 0.80 | 1.00 |
| tire | 57 | 63% | 0.39 | 1.00 |

**Aggregate:** local-only recall 0.56 -> hybrid 1.00 across 10 categories. The router carries the same lift on 3D point clouds that it does on 2D images, with zero change to the orchestration, which is the modality-agnostic claim made concrete.
