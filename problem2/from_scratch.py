"""Student implementation points for Problem 2.

The assignment explicitly requires these components to be written from tensor
operations; do not replace them with nn.MultiheadAttention or nn.Transformer.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def make_causal_mask(sequence_length: int, device=None) -> torch.Tensor:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    return torch.ones(
        sequence_length, sequence_length, dtype=torch.bool, device=device
    ).tril()


def scaled_dot_product_attention(q, k, v, mask=None):
    """Apply scaled dot-product attention and return output and weights.

    Boolean masks use True for allowed query/key pairs, matching PyTorch's
    scaled_dot_product_attention convention. Floating masks are additive.
    """
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same head width")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("k and v must have the same sequence length")

    logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if mask is not None:
        mask = mask.to(device=logits.device)
        if mask.dtype == torch.bool:
            logits = logits.masked_fill(~mask, float("-inf"))
        else:
            logits = logits + mask.to(dtype=logits.dtype)
    weights = torch.softmax(logits, dim=-1)
    weights = torch.where(torch.isnan(weights), 0.0, weights)
    return torch.matmul(weights, v), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_width = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attention: bool = False):
        # x: (batch, sequence, d_model); split/merge heads in your implementation.
        batch_size, sequence_length, _ = x.shape

        def split_heads(projected):
            return projected.view(
                batch_size, sequence_length, self.num_heads, self.head_width
            ).transpose(1, 2)

        q = split_heads(self.W_Q(x))
        k = split_heads(self.W_K(x))
        v = split_heads(self.W_V(x))
        attended, weights = scaled_dot_product_attention(q, k, v, mask)
        if self.training and self.attention_dropout.p:
            attended = torch.matmul(self.attention_dropout(weights), v)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.d_model
        )
        output = self.W_O(attended)
        if return_attention:
            return output, weights
        return output


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.attention_residual_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.feed_forward_residual_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attention: bool = False):
        attention_output, weights = self.attention(
            self.norm1(x), mask=mask, return_attention=True
        )
        x = x + self.attention_residual_dropout(attention_output)
        x = x + self.feed_forward_residual_dropout(self.feed_forward(self.norm2(x)))
        if return_attention:
            return x, weights
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 4096):
        super().__init__()
        if d_model <= 0 or max_length <= 0:
            raise ValueError("d_model and max_length must be positive")
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scales = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * scales)
        odd_width = encoding[:, 1::2].shape[1]
        encoding[:, 1::2] = torch.cos(positions * scales[:odd_width])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, x):
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError("sequence length exceeds positional encoding capacity")
        return x + self.encoding[:, : x.shape[1]].to(dtype=x.dtype)
