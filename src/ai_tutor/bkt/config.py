"""
BKTransformer model configuration.

Centralised here so that both the training script and the inference
node import from a single source of truth.  Override fields by
subclassing or passing keyword arguments to the constructor.
"""

from dataclasses import dataclass, field


@dataclass
class BKTConfig:
    n_skills: int = 138       # icecream_8th dataset: 137 skills + 1 padding index
    n_embd: int = 256
    n_layer: int = 3
    n_head: int = 4
    block_size: int = 512
    dropout: float = 0.1
