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
    with torch.no_grad():
        corrects, latents, params = tiny_model.infer(obs, output)
    assert corrects.shape == (1, 5, 1)


def test_infer_latents_length(tiny_model):
    T = 7
    obs    = torch.zeros(1, T, 2)
    output = torch.zeros(1, T, 2)
    with torch.no_grad():
        _, latents, _ = tiny_model.infer(obs, output)
    assert len(latents) == T


def test_infer_params_shape(tiny_model):
    n_skills = tiny_model.n_skills
    obs    = torch.zeros(1, 4, 2)
    output = torch.zeros(1, 4, 2)
    with torch.no_grad():
        _, _, params = tiny_model.infer(obs, output)
    assert params.shape == (1, 4, n_skills, 4)


def test_infer_returns_three_values(tiny_model):
    """infer() must not return a loss — callers unpack exactly 3 values."""
    obs    = torch.zeros(1, 3, 2)
    output = torch.zeros(1, 3, 2)
    with torch.no_grad():
        result = tiny_model.infer(obs, output)
    assert len(result) == 3


def test_infer_priors_in_unit_interval(tiny_model):
    obs    = torch.zeros(1, 4, 2)
    output = torch.zeros(1, 4, 2)
    with torch.no_grad():
        _, latents, _ = tiny_model.infer(obs, output)
    for latent in latents:
        assert latent.min().item() >= 0.0
        assert latent.max().item() <= 1.0


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

def test_forward_returns_four_values(tiny_model):
    obs    = torch.zeros(1, 3, 2)
    output = torch.zeros(1, 3, 2)
    with torch.no_grad():
        result = tiny_model.forward(obs, output)
    assert len(result) == 4


def test_forward_loss_is_scalar(tiny_model):
    obs    = torch.zeros(1, 3, 2)
    output = torch.zeros(1, 3, 2)
    with torch.no_grad():
        _, _, _, loss = tiny_model.forward(obs, output)
    assert loss.ndim == 0
