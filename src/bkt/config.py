from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class BKTConfig:
    n_skills: int
    n_embd: int = 64
    hidden_dim: int = 128
    block_size: int = 512
    dropout: float = 0.1
    max_guess: float = 0.5
    max_slip: float = 0.5
    lambda_consistency: float = 0.25
    lambda_guess: float = 0.25
    lambda_slip: float = 0.25

    @classmethod
    def from_dict(cls, raw: dict) -> "BKTConfig":
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)
