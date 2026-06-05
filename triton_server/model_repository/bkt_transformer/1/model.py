import os

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        import torch
        from ai_tutor.bkt.config import BKTConfig
        from ai_tutor.bkt.model import BKTransformer

        torch.set_num_threads(int(os.getenv("BKT_NUM_THREADS", "2")))

        ckpt_path = os.getenv(
            "BKT_CHECKPOINT",
            "/opt/bkt_module/ai_tutor/bkt/checkpoints/upgraded-best-epoch=09-val_auc=0.7993.pt",
        )
        self.model = BKTransformer(BKTConfig())
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt.get("state_dict", ckpt))
        self.model.eval()

    def execute(self, requests):
        import torch

        responses = []
        for request in requests:
            obs_np    = pb_utils.get_input_tensor_by_name(request, "obs").as_numpy()
            output_np = pb_utils.get_input_tensor_by_name(request, "output").as_numpy()
            obs_t    = torch.from_numpy(obs_np).float()
            output_t = torch.from_numpy(output_np).float()
            with torch.no_grad():
                corrects, latents, params = self.model.infer(obs_t, output_t)
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("corrects", corrects.numpy()),
                pb_utils.Tensor("latents",  latents.numpy()),
                pb_utils.Tensor("params",   params.numpy()),
            ]))
        return responses

    def finalize(self):
        self.model = None
