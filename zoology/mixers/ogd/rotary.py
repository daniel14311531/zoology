"""
Rotary positional embeddings (RoPE).
Compatible with attention tensors shaped (B, nh, T, hs).
"""

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate last-dim tensor halves for RoPE.

    For input [x0, x1, x2, x3, ...], returns [-x1, x0, -x3, x2, ...]
    to enable 2D rotation: (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to query and key.

    Args:
        x: (B, nh, T, hs)
        cos, sin: (T, hs) or (1, 1, T, hs)
    """
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0).unsqueeze(0)

    x = (x * cos) + (rotate_half(x) * sin)
    return x


class RotaryEmbedding(nn.Module):
    """RoPE cache and generator."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RotaryEmbedding dim must be even.")
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        offset: int = 0,
        seq_len: int = None
    ) -> torch.Tensor:
        """
        Return the output after RoPE.

        Args:
            x: tensor shaped (B, nh, T, hs) or similar; only length is used.
            seq_len: override sequence length.
            offset: position offset for rotary embeddings.
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        if offset + seq_len > self.cos_cached.shape[0] or self.cos_cached.device != x.device:
            self._set_cos_sin_cache(seq_len=offset + seq_len, device=x.device, dtype=x.dtype)

        cos = self.cos_cached[offset:offset+seq_len, :].to(x.dtype)
        sin = self.sin_cached[offset:offset+seq_len, :].to(x.dtype)
        return apply_rotary_pos_emb(x, cos, sin)
