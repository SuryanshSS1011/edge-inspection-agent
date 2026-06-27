# Multi-category robustness (real MVTec data)

Each category has its own independently-trained modest classifier, its own fitted temperature, and its own disjoint eval split. Hybrid recall holding across categories is the robustness claim: the cost-aware router, not a single lucky model, carries the accuracy.

| Category | Eval n | ECE (before→after) | Local-only | Cloud-every | **Hybrid** |
|---|---|---|---|---|---|
| bottle | 61 | 0.059→0.050 | 0.908 | 0.997 | **0.988** [0.985–0.991] |
| grid | 69 | 0.222→0.221 | 0.811 | 0.997 | **0.939** [0.936–0.943] |
| metal_nut | 69 | 0.167→0.096 | 0.951 | 0.998 | **0.974** [0.944–0.989] |
| screw | 98 | 0.224→0.214 | 0.873 | 0.963 | **0.970** [0.969–0.972] |
| cable | 76 | 0.163→0.155 | 0.883 | 0.998 | **0.963** [0.961–0.966] |
| capsule | 72 | 0.147→0.138 | 0.921 | 0.998 | **0.980** [0.978–0.982] |

**Aggregate across 6 categories:** hybrid recall 0.969 ± 0.015 (std), vs local-only mean 0.891. Hybrid lifts recall by **+0.078** on average and stays tight across categories — the orchestration generalizes.
