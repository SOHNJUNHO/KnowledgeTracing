"""
Unit tests for BKTransformer.

These tests use a tiny model config (n_skills=10, n_layer=1) to keep
execution fast without a GPU.  No checkpoint is loaded.
"""

import inspect
import pytest
import torch

from ai_tutor.bkt.config import BKTConfig
from ai_tutor.bkt.model import BKTransformer


@pytest.fixture
def tiny_model() -> BKTransformer:
    config = BKTConfig(n_skills=10, n_embd=32, n_layer=1, n_head=2, block_size=16, dropout=0.0)
    return BKTransformer(config).eval()


# ---------------------------------------------------------------------------
# infer() output shapes
# ---------------------------------------------------------------------------

def test_infer_corrects_shape(tiny_model):
    obs    = torch.zeros(1, 5, 2)
    output = torch.zeros(1, 5, 2)
    S = tiny_model.config.block_size
    with torch.no_grad():
        corrects, latents, params = tiny_model.infer(obs, output)
    assert corrects.shape == (1, S, 1)  # padded to block_size


def test_infer_latents_length(tiny_model):
    T = 7
    obs    = torch.zeros(1, T, 2)
    output = torch.zeros(1, T, 2)
    S = tiny_model.config.block_size
    with torch.no_grad():
        _, latents, _ = tiny_model.infer(obs, output)
    # latents is now a tensor (B, S, n_skills) padded to block_size
    assert latents.shape == (1, S, tiny_model.n_skills)


def test_infer_params_shape(tiny_model):
    n_skills = tiny_model.n_skills
    obs    = torch.zeros(1, 4, 2)
    output = torch.zeros(1, 4, 2)
    with torch.no_grad():
        _, _, params = tiny_model.infer(obs, output)
    assert params.shape == (1, 4, n_skills, 4)


def test_infer_priors_in_unit_interval(tiny_model):
    obs    = torch.zeros(1, 4, 2)
    output = torch.zeros(1, 4, 2)
    with torch.no_grad():
        _, latents, _ = tiny_model.infer(obs, output)
    # latents is now a tensor (B, S, n_skills)
    assert latents.min().item() >= 0.0
    assert latents.max().item() <= 1.0


# ---------------------------------------------------------------------------
# Regression: lambd must be a tuple, not a mutable list
# ---------------------------------------------------------------------------

def test_lambd_default_is_tuple():
    sig = inspect.signature(BKTransformer.forward)
    default = sig.parameters["lambd"].default
    assert isinstance(default, tuple), (
        "lambd default must be a tuple to avoid mutable default argument bug"
    )


# ---------------------------------------------------------------------------
# forward() still returns loss for training
# ---------------------------------------------------------------------------

def test_forward_loss_is_scalar(tiny_model):
    obs    = torch.zeros(1, 3, 2)
    output = torch.zeros(1, 3, 2)
    with torch.no_grad():
        _, _, _, loss = tiny_model.forward(obs, output)
    assert loss.ndim == 0
