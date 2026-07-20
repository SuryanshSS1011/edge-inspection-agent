# Backbone ablation: handcrafted vs. mobilenet vs. dinov2 (real MVTec data)

Same LogisticRegression + temperature head, same disjoint splits, only the frozen feature backbone changes across a weak (handcrafted), medium (ImageNet MobileNetV2), and SOTA self-supervised (DINOv2) extractor. The router, privacy filter, and outbox are untouched, so each is a drop-in swap behind the ONNX interface.

| Category | Local handcrafted | Local mobilenet | Local dinov2 | Hybrid handcrafted | Hybrid mobilenet | Hybrid dinov2 |
|---|---|---|---|---|---|---|
| bottle | 0.908 | 0.982 | 0.988 | **0.988** | **1.000** | **1.000** |
| grid | 0.811 | 0.808 | 0.993 | **0.939** | **0.916** | **1.000** |
| metal_nut | 0.951 | 0.920 | 0.976 | **0.974** | **0.982** | **0.998** |
| screw | 0.873 | 0.888 | 0.941 | **0.970** | **0.993** | **0.993** |
| cable | 0.883 | 0.874 | 0.913 | **0.963** | **0.945** | **0.995** |
| capsule | 0.921 | 0.955 | 0.906 | **0.980** | **0.961** | **0.994** |

**Local-only recall spans up to 0.185 across backbones** within a category. The choice of frozen extractor matters a lot when the local model decides alone: DINOv2 lifts the floor on hard textures, the handcrafted features lag, MobileNet sits between.

**But hybrid recall spans at most 0.084 across the same backbones.** That is the finding: **the router is robust to the local backbone, weak or SOTA.** It escalates whatever the local model is unsure about regardless of *why*, absorbing local-model variance. The backbone is a genuine drop-in (zero router/privacy/outbox change), and the orchestration, not the choice of local model, carries the accuracy.
