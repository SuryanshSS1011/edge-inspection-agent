"""Export a frozen DINOv2 backbone to ONNX so inference needs only onnxruntime.

Run once on ROAR (needs torch + internet for the torch.hub weights). Produces
models/dinov2.onnx that takes an ImageNet-normalized 224x224 NCHW tensor and returns the
CLS-token embedding, matching eval/dinov2_features.py.

    python -m eval.export_dinov2 --variant vits14        # 384-dim (default, light)
    python -m eval.export_dinov2 --variant vitb14        # 768-dim (stronger)

vits14 is the sensible default: strong features, cheap to run per image over a whole
dataset. vitb14 if you want maximum backbone strength for the ablation's strong arm.
"""

import argparse

_VARIANTS = {
    "vits14": ("dinov2_vits14", 384),
    "vitb14": ("dinov2_vitb14", 768),
    "vitl14": ("dinov2_vitl14", 1024),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="vits14", choices=list(_VARIANTS))
    parser.add_argument("--out", default="models/dinov2.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    import torch  # lazy, only needed for export

    hub_name, dim = _VARIANTS[args.variant]
    print(f"loading {hub_name} (dim={dim}) from torch.hub ...")
    model = torch.hub.load("facebookresearch/dinov2", hub_name)
    model.eval()

    # DINOv2's forward returns the CLS-token global embedding; wrap so the ONNX graph has a
    # single [N, dim] output that dinov2_features._embed reshapes.
    class _CLS(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)  # DINOv2 backbone returns the CLS embedding [N, dim]

    wrapped = _CLS(model)
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = wrapped(dummy)
    assert out.shape[-1] == dim, f"expected dim {dim}, got {tuple(out.shape)}"

    torch.onnx.export(
        wrapped,
        dummy,
        args.out,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"wrote {args.out}  (input 1x3x224x224 -> embedding Nx{dim})")


if __name__ == "__main__":
    main()
