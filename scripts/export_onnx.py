"""
Export BKTransformer to ONNX and verify numerical match with PyTorch.

Run once on your laptop after making Stage 2 model changes:
    .venv/bin/python scripts/export_onnx.py

Produces model.onnx in the repo root. Bake it into the Docker image via
services/Dockerfile.bkt_onnx before building the ONNX service.
"""

import numpy as np
import torch

from ai_tutor.bkt.config import BKTConfig
from ai_tutor.bkt.model import BKTransformer

CHECKPOINT = "src/ai_tutor/bkt/checkpoints/upgraded-best-epoch=09-val_auc=0.7993.pt"
OUTPUT     = "model.onnx"


def main() -> None:
    config = BKTConfig()
    model  = BKTransformer(config)
    ckpt   = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    S     = config.block_size
    dummy = torch.zeros(1, S, 2)

    # Wrap infer() so ONNX traces the inference path only.
    # Exporting model directly would trace forward() which includes the
    # training loss and its shape-dependent Python conditional.
    class _InferWrapper(torch.nn.Module):
        def __init__(self, m: torch.nn.Module) -> None:
            super().__init__()
            self.m = m

        def forward(self, obs: torch.Tensor, output: torch.Tensor):  # type: ignore[override]
            return self.m.infer(obs, output)

    print(f"Exporting to {OUTPUT} (opset 17, block_size={S})...")
    torch.onnx.export(
        _InferWrapper(model),
        (dummy, dummy),
        OUTPUT,
        input_names=["obs", "output"],
        output_names=["corrects", "latents", "params"],
        dynamic_axes={"obs": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print("Export done. Verifying numerical match...")

    import onnxruntime as ort
    sess    = ort.InferenceSession(OUTPUT, providers=["CPUExecutionProvider"])
    pt_out  = model.infer(dummy, dummy)
    ort_out = sess.run(None, {"obs": dummy.numpy(), "output": dummy.numpy()})

    assert np.allclose(
        pt_out[0].detach().numpy(), ort_out[0], atol=1e-5
    ), "corrects mismatch between PyTorch and ONNX!"
    print("ONNX export verified. Outputs match within atol=1e-5.")
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
