import torch

from src.bkt.config import BKTConfig
from src.bkt.model import BKTransformer


def test_forward_pass_shapes_and_finite_loss():
    config = BKTConfig(n_skills=6, n_embd=16, hidden_dim=32, dropout=0.0)
    model = BKTransformer(config)

    batch_size, seq_len = 2, 5
    obs = torch.zeros(batch_size, seq_len, 2)
    obs[..., 0] = torch.randint(0, 5, (batch_size, seq_len)).float()
    obs[..., 1] = torch.randint(0, 2, (batch_size, seq_len)).float()

    output = torch.zeros(batch_size, seq_len, 2)
    output[..., 0] = torch.randint(0, 5, (batch_size, seq_len)).float()
    output[..., 1] = torch.randint(0, 2, (batch_size, seq_len)).float()

    corrects, latents, params, loss = model(obs, output)

    assert corrects.shape == (batch_size, seq_len, 1)
    assert len(latents) == seq_len
    assert params.shape == (batch_size, seq_len, config.n_skills, 4)
    assert torch.isfinite(loss)
