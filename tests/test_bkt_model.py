import torch

from knowledge_tracing.models.static_neural_bkt import BKTConfig, BKTransformer


def test_forward_pass_shapes_and_finite_loss():
    config = BKTConfig(n_skills=6, n_embd=16, hidden_dim=32, dropout=0.0)
    model = BKTransformer(config)

    batch_size, seq_len = 2, 5
    output = torch.zeros(batch_size, seq_len, 2)
    output[..., 0] = torch.randint(1, 6, (batch_size, seq_len)).float()
    output[..., 1] = torch.randint(0, 2, (batch_size, seq_len)).float()

    corrects, latents, params, loss = model(output)

    assert corrects.shape == (batch_size, seq_len, 1)
    assert len(latents) == seq_len
    assert params.shape == (batch_size, seq_len, config.n_skills, 4)
    assert torch.isfinite(loss)
