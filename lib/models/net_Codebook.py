import torch
import torch.nn as nn
from dataclasses import dataclass
from torch import Tensor, nn
from typing import Literal
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange

@dataclass(frozen=True)
class TransformerConfig:

    input_dim: int = 9
    output_dim: int = 1
    d_latent: int = 256
    d_feedforward: int = 1024
    num_encoder_layers: int = 8 
    num_heads: int = 8 
    dropout_p: float = 0.1 
    activation: Literal["gelu", "relu"] = "gelu"
    
class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        d_latent = config.d_latent
        n_heads = config.num_heads
        
        self.layernorm1 = nn.LayerNorm(d_latent)
        self.sattn_qkv_proj = nn.Linear(d_latent, d_latent * 3, bias=False)
        self.sattn_out_proj = nn.Linear(d_latent, d_latent, bias=False)
        
        assert d_latent % n_heads == 0, "d_latent must be divisible by n_heads"
        self.rotary_emb = RotaryEmbedding(dim = d_latent // n_heads)

        self.layernorm2 = nn.LayerNorm(d_latent)
        self.mlp = nn.Sequential(
            nn.Linear(d_latent, config.d_feedforward),
            nn.GELU() if config.activation == "gelu" else nn.ReLU(),
            nn.Dropout(config.dropout_p),
            nn.Linear(config.d_feedforward, d_latent),
            nn.Dropout(config.dropout_p),
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor,
    ) -> Tensor:

        # 1. Self-Attention (with pre-normalization)
        x_norm = self.layernorm1(x)
        sattn_output = self._sattn(x_norm, attn_mask)
        x = x + sattn_output

        # 2. Feed-Forward Network (with pre-normalization)
        x_norm = self.layernorm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output
        
        return x

    def _sattn(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        config = self.config
        
        # Q, K, V 프로젝션
        q, k, v = self.sattn_qkv_proj(x).chunk(3, dim=-1)

        # Multi-head 형태로 재배열
        q, k, v = map(
            lambda t: rearrange(t, "b t (nh dh) -> b nh t dh", nh=config.num_heads),
            (q, k, v),
        )

        # Rotary Positional Embedding 적용
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        # Scaled Dot-Product Attention
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=config.dropout_p if self.training else 0.0
        )

        out = rearrange(out, "b nh t dh -> b t (nh dh)")

        return self.sattn_out_proj(out) 
    

class Transformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.input_layer = nn.Linear(config.input_dim, config.d_latent)

        self.encoder_layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_encoder_layers)]
        )

        self.output_layer = nn.Sequential(
            nn.LayerNorm(config.d_latent),
            nn.Linear(config.d_latent, config.output_dim) 
        )
        
    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
    ) -> tuple[Tensor, Tensor]:

        x = self.input_layer(x)
        
        for layer in self.encoder_layers:
            x = layer(x, attn_mask=mask)
            
        encoder_output = self.output_layer(x)

        return encoder_output