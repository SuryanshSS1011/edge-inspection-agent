"""Export a small PointNet encoder to ONNX for the 3D perception path.

A minimal PointNet (Qi et al. 2017): a shared per-point MLP followed by a symmetric max-pool
gives a permutation-invariant global descriptor of an unordered point set. We use it as a
FROZEN feature extractor (random-init but fixed), the 3D analogue of the frozen image
backbones: the same LogisticRegression + temperature head sits on top, so the router,
privacy filter, and outbox are unchanged. The point is not a trained 3D classifier but a
consistent embedding whose distance-from-normal the router can band, exactly as in 2D.

Input:  a fixed-size point set [1, N, 3] (sample/pad to N points, coordinates centered).
Output: a [1, D] global descriptor.

    python -m eval.export_pointnet --points 2048 --dim 256 --out models/pointnet.onnx

Run once on ROAR (needs torch). Inference then uses onnxruntime only.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=2048, help="points per cloud (N)")
    parser.add_argument("--dim", type=int, default=256, help="global descriptor width D")
    parser.add_argument("--out", default="models/pointnet.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    class PointNetEncoder(nn.Module):
        """Shared per-point MLP -> max-pool -> global feature. Frozen (eval mode, no grad)."""

        def __init__(self, dim):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Conv1d(3, 64, 1), nn.ReLU(),
                nn.Conv1d(64, 128, 1), nn.ReLU(),
                nn.Conv1d(128, dim, 1), nn.ReLU(),
            )

        def forward(self, x):  # x: [B, N, 3]
            x = x.transpose(1, 2)          # [B, 3, N]
            x = self.mlp(x)                # [B, D, N]
            x = torch.max(x, dim=2).values  # [B, D] symmetric pool -> permutation invariant
            return x

    torch.manual_seed(0)  # fixed init so the frozen embedding is reproducible
    model = PointNetEncoder(args.dim).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dummy = torch.randn(1, args.points, 3)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape[-1] == args.dim, out.shape

    torch.onnx.export(
        model, dummy, args.out,
        input_names=["points"], output_names=["embedding"],
        dynamic_axes={"points": {0: "batch", 1: "num_points"}, "embedding": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"wrote {args.out}  (points [B,{args.points},3] -> embedding [B,{args.dim}])")


if __name__ == "__main__":
    main()
