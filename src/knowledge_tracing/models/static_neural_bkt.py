from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BKTConfig:
    n_skills: int
    n_embd: int = 64
    hidden_dim: int = 128
    block_size: int = 818
    dropout: float = 0.1
    max_guess: float = 0.5
    max_slip: float = 0.5
    lambda_consistency: float = 0.25
    lambda_guess: float = 0.25
    lambda_slip: float = 0.25


class SkillParameterHead(nn.Module):
    """Three-layer MLP used to generate per-skill BKT parameters."""

    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BKTransformer(nn.Module):
    """Static neural-parameterized BKT model.

    The class name is preserved for compatibility with the tutor pipeline, but
    the implementation is no longer Transformer-based.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.n_skills = config.n_skills
        self.skill_emb = nn.Embedding(config.n_skills, config.n_embd)
        self.skill_params = SkillParameterHead(
            embedding_dim=config.n_embd,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
        )

    def forward(self, output: torch.Tensor, lambd: list[float] | None = None):
        if lambd is None:
            lambd = [
                self.config.lambda_consistency,
                self.config.lambda_guess,
                self.config.lambda_slip,
            ]

        batch_size, seq_len, dims = output.shape
        assert dims == 2, "Expected output shape (B, T, 2)"

        skill_indices = torch.arange(self.n_skills, device=output.device)
        skill_features = self.skill_emb(skill_indices)
        logits = self.skill_params(skill_features)

        prior_logits = logits[..., 0]
        param_logits = logits[..., 1:]

        prior = torch.sigmoid(prior_logits)
        oparams = torch.sigmoid(param_logits)

        params = oparams.view(1, 1, self.n_skills, 4).expand(batch_size, seq_len, self.n_skills, 4)

        regularization = self._regularization_loss(oparams, lambd)

        corrects = torch.zeros(
            batch_size, seq_len, self.n_skills, device=output.device, dtype=output.dtype
        )
        latent = prior.unsqueeze(0).repeat(batch_size, 1)
        latents: list[torch.Tensor] = []

        for step in range(seq_len):
            latent = torch.clamp(latent, min=1e-5, max=1 - 1e-5)
            latents.append(latent)

            correct, latent = self.extract_latent_correct(
                params=params[:, step],
                latent=latent,
                true_correct=output[:, step, -1],
                skills=torch.where(output[:, step, 0] == -1000, 0, output[:, step, 0]).long(),
            )
            corrects[:, step] = correct

        skill_idx = torch.where(output[..., 0] == -1000, 0, output[..., 0]).long()
        target_corrects = torch.gather(corrects, dim=-1, index=skill_idx.unsqueeze(-1))
        mask = output[..., 1] != -1000

        if mask.any():
            prediction_loss = F.binary_cross_entropy(target_corrects[mask], output[..., 1:][mask])
        else:
            prediction_loss = torch.zeros((), device=output.device, dtype=output.dtype)

        loss = prediction_loss + regularization
        return target_corrects, latents, params, loss

    def _regularization_loss(self, params: torch.Tensor, lambd: list[float]) -> torch.Tensor:
        guess = params[..., 2]
        slip = params[..., 3]

        consistency = F.relu(guess + slip - 1).mean()
        guess_penalty = F.relu(guess - self.config.max_guess).mean()
        slip_penalty = F.relu(slip - self.config.max_slip).mean()

        regularization = (
            lambd[0] * consistency
            + lambd[1] * guess_penalty
            + lambd[2] * slip_penalty
        )
        return regularization

    def extract_latent_correct(
        self,
        params: torch.Tensor,
        latent: torch.Tensor,
        true_correct: torch.Tensor,
        skills: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        learn = params[..., 0]
        guess = params[..., 2]
        slip = params[..., 3]

        correct = latent * (1 - slip) + (1 - latent) * guess
        k_t1 = (latent * (1 - slip)) / (latent * (1 - slip) + (1 - latent) * guess + 1e-6)
        k_t0 = (latent * slip) / (latent * slip + (1 - latent) * (1 - guess) + 1e-6)

        updated = latent.clone()
        batch_indices = torch.arange(latent.shape[0], device=latent.device)
        updated[batch_indices, skills] = torch.where(
            true_correct > 0.5,
            k_t1[batch_indices, skills],
            k_t0[batch_indices, skills],
        )
        updated[batch_indices, skills] = (
            updated[batch_indices, skills]
            + (1 - updated[batch_indices, skills]) * learn[batch_indices, skills]
        )

        return correct, torch.clamp(updated, 1e-4, 1 - 1e-4)
