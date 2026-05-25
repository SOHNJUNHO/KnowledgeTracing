"""
Unit tests for BKTransformer.

These tests use a tiny model config (n_skills=10, n_layer=1) to keep
execution fast without a GPU.  No checkpoint is loaded.
"""

import pytest
import torch

from ai_tutor.bkt.config import BKTConfig
from ai_tutor.bkt.model import BKTransformer


@pytest.fixture
def tiny_model() -> BKTransformer:
    config = BKTConfig(n_skills=10, n_embd=32, n_layer=1, n_head=2, block_size=30, dropout=0.0)
    return BKTransformer(config).eval()


def test_infer_output_shapes(tiny_model):
    T = 5
    S = tiny_model.config.block_size
    obs    = torch.zeros(1, T, 2)
    output = torch.zeros(1, T, 2)
    with torch.no_grad():
        corrects, latents, params = tiny_model.infer(obs, output)
    assert corrects.shape == (1, S, 1)                       # padded to block_size
    assert latents.shape  == (1, S, tiny_model.n_skills)     # padded to block_size
    assert params.shape   == (1, T, tiny_model.n_skills, 4)  # not padded


def test_infer_priors_in_unit_interval(tiny_model):
    obs    = torch.zeros(1, 4, 2)
    output = torch.zeros(1, 4, 2)
    with torch.no_grad():
        _, latents, _ = tiny_model.infer(obs, output)
    assert latents.min().item() >= 0.0
    assert latents.max().item() <= 1.0


def test_forward_loss_is_scalar(tiny_model):
    obs    = torch.zeros(1, 3, 2)
    output = torch.zeros(1, 3, 2)
    with torch.no_grad():
        _, _, _, loss = tiny_model.forward(obs, output)
    assert loss.ndim == 0
