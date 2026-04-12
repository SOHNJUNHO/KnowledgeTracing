import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings


class SwiGLU(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.gate_up_proj = nn.Linear(in_dim, 2 * hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        x = self.dropout(x)
        return self.down_proj(x)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=config.block_size)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        hidden_dim = int(8 / 3 * config.n_embd)
        self.mlp = SwiGLU(config.n_embd, hidden_dim, dropout=config.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BKTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_skills = config.n_skills
        self.skill_emb = nn.Embedding(config.n_skills, config.n_embd)
        self.correct_emb = nn.Embedding(2, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.PReLU(),
            nn.Linear(config.n_embd, config.n_embd),
            nn.PReLU(),
            nn.Linear(config.n_embd, self.n_skills * 4),
        )
        layers = []
        for _ in range(4):
            layers.extend([nn.Linear(config.n_embd, config.n_embd), nn.ReLU()])
        layers.append(nn.Linear(config.n_embd, 5))
        self.skill_params = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Shared encoder: token embedding → transformer → per-skill params
    # ------------------------------------------------------------------

    def _encode(self, obs, output):
        """Shared forward pass up to BKT parameter extraction."""
        B, T, D = obs.shape
        assert D == 2, "Expected obs shape (B, T, 2)"

        skill_ids = obs[..., 0].long()
        corrects = obs[..., 1].long().clamp(0, 1)
        token_embeddings = self.skill_emb(skill_ids) + self.correct_emb(corrects)
        x = self.drop(token_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)

        logit_diff = self.head(x)  # (B, T, n_skills*4)

        skill_indices = torch.arange(self.n_skills, device=obs.device)
        logits = self.skill_params(self.skill_emb(skill_indices))  # (n_skills, 5)

        oparams = logits[..., 1:].reshape(1, -1)
        params = torch.sigmoid(oparams + logit_diff).view(B, T, -1, 4)

        return logits, logit_diff, oparams, params

    def _run_bkt_loop(self, obs, output, params, logits):
        """Iterate the BKT update rule over T timesteps."""
        B, T, _ = obs.shape
        corrects = torch.zeros_like(params[..., -1])
        latent = torch.sigmoid(logits[..., 0].repeat((B, 1)))

        latents = []
        for i in range(T):
            latent = torch.clamp(latent, min=1e-5, max=1 - 1e-5)
            latents.append(latent)
            correct, latent = self.extract_latent_correct(
                params[:, i].view(B, -1, 4),
                latent,
                true_correct=output[:, i, -1],
                skills=torch.where(output[:, i, 0] == -1000, 0, output[:, i, 0]).long(),
            )
            corrects[:, i] = correct.squeeze()

        skill_idx = torch.where(output[..., 0] == -1000, 0, output[..., 0]).long()
        corrects = torch.gather(corrects, dim=-1, index=skill_idx.unsqueeze(-1))
        return corrects, latents

    # ------------------------------------------------------------------
    # Inference-only forward — no loss computation
    # ------------------------------------------------------------------

    def infer(self, obs, output):
        """Inference-only pass. Returns (corrects, latents, params) without loss.

        Use this at serving time. The full forward() method is for training only.
        """
        logits, _, _, params = self._encode(obs, output)
        corrects, latents = self._run_bkt_loop(obs, output, params, logits)
        return corrects, latents, params

    # ------------------------------------------------------------------
    # Training forward — computes physics constraint loss + BCE
    # ------------------------------------------------------------------

    def forward(self, obs, output, lambd=(50, 50, 50, 1)):
        """Full training forward pass. Returns (corrects, latents, params, loss)."""
        logits, logit_diff, oparams, params = self._encode(obs, output)

        oparams_sig = torch.sigmoid(oparams.view(-1, 4))

        loss = (
            lambd[0] * (
                F.relu(params[..., 0] - (1 - params[..., 3]) / (params[..., 2] + 1e-6)).mean()
                + F.relu(oparams_sig[..., 0] - (1 - oparams_sig[..., 3]) / (oparams_sig[..., 2] + 1e-6)).mean()
            )
            + lambd[1] * (F.relu(params[..., 2] - 0.5).mean() + F.relu(oparams_sig[..., 2] - 0.5).mean())
            + lambd[2] * (F.relu(params[..., 3] - 0.5).mean() + F.relu(oparams_sig[..., 3] - 0.5).mean())
            + lambd[3] * torch.mean(logit_diff ** 2)
            + lambd[3] * (
                torch.mean((logit_diff[:, 1:] - logit_diff[:, :-1]) ** 2)
                if logit_diff.shape[1] > 1 else 0
            )
        )

        corrects, latents = self._run_bkt_loop(obs, output, params, logits)

        mask = output[..., 1] != -1000
        loss = loss + F.binary_cross_entropy(corrects[mask], output[..., 1:][mask])

        return corrects, latents, params, loss

    def extract_latent_correct(self, params, latent, true_correct, skills):
        l, g, s = params[..., 0], params[..., 2], params[..., 3]

        correct = latent * (1 - s) + (1 - latent) * g
        k_t1 = (latent * (1 - s)) / (latent * (1 - s) + (1 - latent) * g)
        k_t0 = (latent * s) / (latent * s + (1 - latent) * (1 - g))
        k_t = torch.clone(latent)

        k_t[range(len(k_t)), skills] = torch.where(
            true_correct > 0.5,
            k_t1[range(len(k_t)), skills],
            k_t0[range(len(k_t)), skills],
        )
        k_t[range(len(k_t)), skills] = (
            k_t[range(len(k_t)), skills]
            + (1 - k_t[range(len(k_t)), skills]) * l[range(len(k_t)), skills]
        )
        return correct, torch.clamp(k_t, 1e-4, 1 - 1e-4)
