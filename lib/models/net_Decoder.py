import torch
import torch.nn as nn
from .components.pose_transformer import TransformerDecoder
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class TransformerConfig:
    
    num_tokens : int = 1
    token_dim: int = 1
    dim: int = 256
    depth: int = 4
    heads: int = 4
    mlp_dim: int = 1024
    dim_head: int = 64
    dropout: float = 0.0
    emb_dropout: float = 0.0
    norm: str = 'layer'
    context_dim: int = 256