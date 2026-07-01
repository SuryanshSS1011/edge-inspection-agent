# Backbone ablation — handcrafted vs. MobileNetV2 (real MVTec data)

Same LogisticRegression + temperature head, same disjoint splits, only the frozen feature backbone changes. The router, privacy filter, and outbox are untouched — this is a drop-in swap behind the ONNX interface.

| Category | Local (handcrafted → mobilenet) | Hybrid (handcrafted → mobilenet) |
|---|---|---|
| bottle | 0.908 → **0.994** (+0.086) | 0.988 → **1.000** (+0.012) |
| grid | 0.811 → **0.808** (-0.003) | 0.939 → **0.916** (-0.024) |
| metal_nut | 0.951 → **0.920** (-0.031) | 0.974 → **0.982** (+0.008) |
| screw | 0.873 → **0.888** (+0.016) | 0.970 → **0.993** (+0.023) |

**Local Δ ranges -0.031 to +0.086** — an off-the-shelf *ImageNet* MobileNet helps object-like categories (bottle, screw) but not out-of-distribution industrial textures (grid) or fine metal defects (metal_nut). A backbone fine-tuned on the domain would lift those; the generic one is a mixed bag.

**But hybrid Δ is -0.024 to +0.023** — nearly flat, positive in most categories. That is the real finding: **the router is robust to the local backbone.** It escalates the cases the local model is unsure about regardless of *why* it's unsure, so it absorbs local-model variance. The backbone is a genuine drop-in (zero router/privacy/outbox change), and the orchestration — not the choice of local model — carries the accuracy.
