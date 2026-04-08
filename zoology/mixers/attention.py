import torch 
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math


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


class SelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, qkv):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
        """
        seqlen = qkv.shape[1]
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
        )
        scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


class MHA(nn.Module):
    """Multi-head self-attention
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int=1,
        bias: bool=True,
        dropout: float=0.0,
        layer_idx: int=None,
        use_rope: bool=False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert (
            self.d_model % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        self.Wqkv = nn.Linear(
            d_model, 3 * d_model, bias=bias
        )
        self.inner_attn = SelfAttention(attention_dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model)

        self.use_rope = use_rope
        if use_rope:
            self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor):
        """"""
        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )

        B, L, _, H, D = qkv.shape
        if self.use_rope:
            q, k, v = torch.split(qkv, [1, 1, 1], dim=-3)
            k = k.contiguous().view(B, L, H, D).transpose(1, 2)  # (B, H, L, D)
            q = q.contiguous().view(B, L, H, D).transpose(1, 2)  # (B, H, L, D)
            k = self.rotary(k, offset=0, seq_len=x.shape[1])
            q = self.rotary(q, offset=0, seq_len=x.shape[1])
            k = k.transpose(1, 2).contiguous().view(B, L, 1, H, D)  # (B, L, 1, H, D)
            q = q.transpose(1, 2).contiguous().view(B, L, 1, H, D)  # (B, L, 1, H, D)
            qkv = torch.cat([q, k, v], dim=-3)

        context = self.inner_attn(qkv)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out
    
    def state_size(self, batch_size: int=1, sequence_length: int=2048):
        return 2 * self.d_model * sequence_length