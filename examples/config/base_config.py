from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # transformer
    num_heads: int = 8
    d_model: int = 32
    num_blocks: int = 2
    d_ff: int = 4 * d_model
    dropout: float = 0.1
    # training
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 0.01
    # dataset
    num_atoms: Optional[int] = None
    num_samples: Optional[int] = None
    delta: Optional[float] = None
    # rydberg
    num_states: int = 2
    num_encoder_embedding_dims: int = 4
    # misc
    device: Optional[str] = None
    profiling: Optional[bool] = None
    seed: Optional[int] = None
    prog_bar: Optional[bool] = None
