"""
BKT ONNX inference microservice (FastAPI).

Serves the exported model.onnx file using ONNX Runtime instead of PyTorch.
No torch dependency — image is ~150 MB vs ~800 MB for the PyTorch version.

Endpoints:
    POST /infer   {"obs": [...], "output": [...]} → corrects, latents, params
    GET  /health

Environment variables:
    ONNX_MODEL_PATH — path to model.onnx (default: model.onnx)
"""

import os
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.environ.get("ONNX_MODEL_PATH", "model.onnx")
S          = 189   # must match block_size used during export

_session: ort.InferenceSession | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _session
    _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print(f"[bkt-service-onnx] Loaded {MODEL_PATH}")
    yield
    _session = None


app = FastAPI(title="BKT ONNX Inference Service", version="1.0.0", lifespan=lifespan)


class InferRequest(BaseModel):
    obs: list
    output: list


@app.post("/infer")
async def infer(req: InferRequest):
    if _session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    obs    = np.array(req.obs,    dtype=np.float32)
    output = np.array(req.output, dtype=np.float32)
    T      = obs.shape[1]  # actual sequence length before padding

    # ONNX model requires fixed sequence length S — pad shorter inputs
    if T < S:
        obs    = np.pad(obs,    ((0, 0), (0, S - T), (0, 0)))
        output = np.pad(output, ((0, 0), (0, S - T), (0, 0)), constant_values=-1000)

    corrects, latents, params = _session.run(None, {"obs": obs, "output": output})

    return {
        "corrects": corrects[0, :T, 0].tolist(),  # strip batch, trim to T
        "latents":  latents[0, :T].tolist(),       # strip batch, trim to T
        "params":   params[0, :T].tolist(),        # strip batch, trim to T
    }


@app.get("/health")
async def health():
    return {"status": "ok", "session_loaded": _session is not None}
