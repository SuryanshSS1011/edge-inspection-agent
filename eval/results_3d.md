# MVTec 3D-AD: the cost router on point clouds (proof of full function)

Same unsupervised anomaly-score setup as the 2D experiments, but the frozen feature extractor is a PointNet encoder over organized point clouds instead of an image backbone. The router, privacy filter, and outbox are unchanged; only the modality differs. Distance from the normal (good) cloud distribution is calibrated to p and banded exactly as in 2D, so this demonstrates the cost-routing decision is genuinely modality-agnostic.

| Category | n | Escalation rate | Local recall | Hybrid recall |
|---|---|---|---|---|
| bagel | 55 | 36% | 0.73 | 0.93 |
| cable_gland | 55 | 31% | 0.75 | 0.75 |
| carrot | 80 | 59% | 0.39 | 0.73 |
| cookie | 66 | 44% | 0.69 | 0.87 |
| dowel | 65 | 78% | 0.25 | 0.25 |
| foam | 50 | 42% | 0.60 | 0.65 |
| peach | 66 | 74% | 0.32 | 0.57 |
| potato | 57 | 40% | 0.67 | 0.76 |
| rope | 51 | 43% | 0.80 | 0.89 |
| tire | 57 | 63% | 0.39 | 0.41 |

**Aggregate across 10 categories:** the PointNet local model catches 0.56 of defects and escalates 51% of clouds. The router runs on 3D point clouds with ZERO change to the orchestration, privacy filter, or outbox, only the feature extractor differs. That is the modality-agnostic claim made concrete: the same cost inequality bands a calibrated p whether it comes from an image or a point cloud.

With real qwen3-vl-plus verdicts on the escalated clouds' paired RGB images, hybrid recall is 0.68 (measured, not modeled).
