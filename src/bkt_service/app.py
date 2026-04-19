"""
BKTransformer inference service.

Loads a trained checkpoint on startup and exposes POST /infer for the
AI Tutor pipeline to call.  Run with:

    uvicorn bkt_service.app:app --port 8001

Environment variables:
    BKT_CHECKPOINT  — required, absolute path to the .pt checkpoint file
    N_SKILLS        — optional, defaults to 138 (must match checkpoint)
"""

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel

from ai_tutor.bkt.config import BKTConfig
from ai_tutor.bkt.model import BKTransformer

_model: BKTransformer | None = None


def _load_model() -> BKTransformer:
    checkpoint_path = os.environ["BKT_CHECKPOINT"]
    n_skills = int(os.getenv("N_SKILLS", "138"))
    config = BKTConfig(n_skills=n_skills)
    model = BKTransformer(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = _load_model()
    print(f"BKTransformer loaded from {os.environ['BKT_CHECKPOINT']}")
    yield


app = FastAPI(lifespan=lifespan)


class InferRequest(BaseModel):
    obs: list
    output: list


@app.post("/infer")
def infer(request: InferRequest) -> dict:
    obs = torch.tensor(request.obs, dtype=torch.float32)
    output = torch.tensor(request.output, dtype=torch.float32)

    with torch.no_grad():
        corrects, latents, params = _model.infer(obs, output)  # type: ignore[union-attr]

    return {
        "corrects": corrects[:, :, 0].tolist(),          # (B, T)
        "latents":  [lat[0].tolist() for lat in latents], # T × n_skills
        "params":   params[0].tolist(),                   # T × n_skills × 4
    }
