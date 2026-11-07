"""Transformer Nano en PyTorch (Nivel 2).

Arquitectura Decoder-only con:
- RMSNorm
- RoPE (rotary embeddings)
- Atención causal multi-head
- SwiGLU feed-forward
- Embeddings compartidos con LM Head
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_config import ModelConfig, CONFIG


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        x_normed = x / (norm + self.eps)
        return self.weight * x_normed


class RotaryEmbedding(nn.Module):
    """RoPE: aplicada a Q y K antes del producto punto."""

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        # (max_seq_len, dim/2, 2) -> cos, sin
        emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        if seq_len is None:
            seq_len = x.size(1)
        cos = self.cos_cached[:, :seq_len, :]
        sin = self.sin_cached[:, :seq_len, :]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(config.hidden_size, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.n_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.n_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.hidden_size, bias=False)

        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len)

        # Máscara causal registrada como buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.size()

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(q, t)
        q, k = apply_rope(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask = self.causal_mask[:, :, :t, :t]
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        return self.w2(gate * up)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm con residuales
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class NanoTransformer(nn.Module):
    def __init__(self, config: ModelConfig = CONFIG):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)

        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.hidden_size)

        # LM Head atado al embedding
        if config.tie_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight = self.embedding.weight
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        b, t = input_ids.size()
        x = self.embedding(input_ids)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))

        return logits, loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: ModelConfig = CONFIG) -> NanoTransformer:
    return NanoTransformer(config)
