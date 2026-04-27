"""
BKT inference microservice (FastAPI).

Loads the BKTransformer checkpoint once at startup and exposes POST /infer.
The main tutor app calls this over HTTP so a single GPU instance can serve
many workers without replicating the model in every process.

Usage (local):
    uvicorn services.bkt_service:app --host 0.0.0.0 --port 8001

Usage (Docker):
    docker compose up bkt-service
"""

import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ai_tutor.bkt.config import BKTConfig
from ai_tutor.bkt.model import BKTransformer

CHECKPOINT_PATH = os.getenv(
    "BKT_CHECKPOINT",
    "src/ai_tutor/bkt/checkpoints/upgraded-best-epoch=09-val_auc=0.7993.pt",
)

_model: BKTransformer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    config = BKTConfig()
    m = BKTransformer(config)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    # Support both raw state_dict and checkpoint dicts (e.g. Lightning format)
    state_dict = checkpoint.get("state_dict", checkpoint)
    m.load_state_dict(state_dict)
    m.eval()
    # Dynamic quantization: Linear weights float32 → int8
    # ~4x smaller in memory, 2–4x faster CPU inference
    m = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
    # torch.compile: additional ~1.5–2x speedup (test separately — can conflict with quantization)
    # m = torch.compile(m)
    _model = m
    print(f"[bkt-service] BKTransformer loaded from {CHECKPOINT_PATH}")
    yield
    _model = None


app = FastAPI(title="BKT Inference Service", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class InferRequest(BaseModel):
    obs: list    # (1, T, 2) nested list — [skill_id, correct] per timestep
    output: list # (1, T, 2) nested list — [skill_id, correct] target


class InferResponse(BaseModel):
    corrects: list  # (1, T)           — predicted correctness probability
    latents: list   # list[T] of list[n_skills] — per-timestep knowledge state
    params: list    # list[T] of list[n_skills] of list[4] — BKT params


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    obs    = torch.tensor(req.obs,    dtype=torch.float32)
    output = torch.tensor(req.output, dtype=torch.float32)
    T = obs.shape[1]

    with torch.no_grad():
        corrects, latents, params = _model.infer(obs, output)

    return InferResponse(
        corrects=corrects[0, :T, 0].tolist(),   # (T,) — predicted correctness per timestep
        latents=latents[0, :T].tolist(),         # (T, n_skills) — knowledge state per timestep
        params=params[0, :T].tolist(),           # (T, n_skills, 4) — BKT params per timestep
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}
