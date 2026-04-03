"""
Gated Linear Attention (GLA) naive PyTorch implementation for Zoology.

Reference: https://arxiv.org/pdf/2312.06635

Core recurrence:
    S_t = Diag(alpha_t) S_{t-1} + k_t^T v_t
    o_t = q_t S_t

where alpha_t is a data-dependent gate vector in (0, 1)^{d_k}.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


GLAState = torch.Tensor

CHUNK_SIZE = 64


def gla_recurrent(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    init_state: GLAState,
    flag: bool = False,
) -> Tuple[torch.Tensor, GLAState]:
    """
    Recurrent (sequential) implementation of GLA for inference.

    Args:
        k: Key tensor of shape (B, L, H, D_k)
        q: Query tensor of shape (B, L, H, D_k)
        v: Value tensor of shape (B, L, H, D_v)
        alpha: Gate tensor of shape (B, L, H, D_k), values in (-inf, 0)
        init_state: S_prev of shape (B, H, D_k, D_v)
        flag: If True, alpha is used directly; if False, exp(alpha) is used

    Returns:
        o: Output tensor of shape (B, L, H, D_v)
        new_state: S_new of shape (B, H, D_k, D_v)
    """
    assert alpha.max() <= 0, f"alpha values should be in (-inf, 0), but found {alpha.max()}"
    B, L, H, D_k = k.shape
    D_v = v.shape[-1]
    device = k.device
    dtype = k.dtype

    S_prev = init_state
    if S_prev is None:
        S_prev = torch.zeros(B, H, D_k, D_v, device=device, dtype=torch.float32)

    S_prev = S_prev.float()

    output_dtype = dtype
    outputs = []

    for t in range(L):
        k_t = k[:, t:t+1, :, :].transpose(1, 2).float()  # (B, H, 1, D_k)
        q_t = q[:, t:t+1, :, :].transpose(1, 2).float()  # (B, H, 1, D_k)
        v_t = v[:, t:t+1, :, :].transpose(1, 2).float()  # (B, H, 1, D_v)
        alpha_t = alpha[:, t:t+1, :, :].transpose(1, 2).float()  # (B, H, 1, D_k)

        # S_t = Diag(alpha_t) S_{t-1} + k_t^T v_t
        if flag == False:
            S_t = alpha_t.transpose(-2, -1).exp() * S_prev + k_t.transpose(-2, -1) @ v_t  # (B, H, D_k, D_v)
        else:
            S_t = alpha_t.transpose(-2, -1) * S_prev + k_t.transpose(-2, -1) @ v_t  # (B, H, D_k, D_v)

        # o_t = q_t S_t
        o_t = q_t @ S_t
        outputs.append(o_t.to(output_dtype).transpose(1, 2))

        S_prev = S_t

    o = torch.cat(outputs, dim=1)
    new_state = S_prev

    return o, new_state


def gla_chunkwise_parallel(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    init_state: GLAState,
    chunk_size: int = CHUNK_SIZE,
) -> Tuple[torch.Tensor, GLAState]:
    """
    Chunkwise parallel implementation of GLA.

    Args:
        k: Key tensor of shape (B, L, H, D_k)
        q: Query tensor of shape (B, L, H, D_k)
        v: Value tensor of shape (B, L, H, D_v)
        alpha: Gate tensor of shape (B, L, H, D_k), valued in (-inf, 0)
        init_state: S_prev of shape (B, H, D_k, D_v)
        chunk_size: Size of each chunk

    Returns:
        o: Output tensor of shape (B, L, H, D_v)
        new_state: S_new of shape (B, H, D_k, D_v)
    """
    assert alpha.max() <= 0, f"alpha values should be in (-inf, 0), but found {alpha.max()}"
    B, L, H, D_k = k.shape
    D_v = v.shape[-1]
    num_chunks = (L + chunk_size - 1) // chunk_size
    device = k.device
    dtype = k.dtype

    S_prev = init_state
    if S_prev is None:
        S_prev = torch.zeros(B, H, D_k, D_v, device=device, dtype=torch.float32)

    S_prev = S_prev.float()

    output_dtype = dtype
    outputs = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, L)

        # Extract chunk data, transpose to (B, H, C, D)
        K_i = k[:, start:end].transpose(1, 2).float()  # (B, H, C, D_k)
        Q_i = q[:, start:end].transpose(1, 2).float()  # (B, H, C, D_k)
        V_i = v[:, start:end].transpose(1, 2).float()  # (B, H, C, D_v)
        alpha_i = alpha[:, start:end].transpose(1, 2).float()  # (B, H, C, D_k)

        cumsum_alpha = torch.cumsum(alpha_i, dim=2)  # (B, H, C, D_k)
        end_cumsum_alpha = cumsum_alpha[:, :, -1, :].contiguous().view(B, H, 1, D_k)  # (B, H, 1, D_k)

        # o_t = q_t^T @ diag(prod_alpha_i[1:t]) @ S_prev + \sum_{s=1}^t diag(prod_alpha_i[s+1:t]) @ k_s @ v_s^T
        weight_i = torch.tril(Q_i @ ((end_cumsum_alpha - cumsum_alpha).exp() * K_i).transpose(-1, -2))  # (B, H, C, C)
        O_i = (Q_i * (end_cumsum_alpha - cumsum_alpha + alpha).exp()) @ S_prev + weight_i @ V_i  # (B, H, C, D_v)
        outputs.append(O_i.to(output_dtype).transpose(1, 2))

        # S_new = diag(prod_alpha_i[1:t]) @ S_prev + (prod_alpha_i * K_i)^T @ V_i
        S_new = end_cumsum_alpha.transpose(-1, -2) * S_prev + ((end_cumsum_alpha - cumsum_alpha).exp() * K_i).transpose(-1, -2) @ V_i

        S_prev = S_new

    o = torch.cat(outputs, dim=1)
    new_state = S_prev

    return o, new_state


class GLANaive(nn.Module):
    """
    Gated Linear Attention layer (naive PyTorch implementation).

    Core recurrence:
        S_t = Diag(alpha_t) S_{t-1} + k_t^T v_t
        o_t = q_t S_t

    where alpha_t is a data-dependent gate vector.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        feature_dim: Feature dimension for key/value (default: d_model // num_heads)
        chunk_size: Chunk size for parallel training
        gate_low_rank: Rank for low-rank gate parameterization
        gate_logit_normalizer: Normalizer for gate logit (default: 1.0)
        layer_idx: Layer index (for state size calculation)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        feature_dim: Optional[int] = None,
        chunk_size: int = 64,
        gate_low_rank: int = 32,
        gate_logit_normalizer: float = 1.0,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.layer_idx = layer_idx

        # Feature dimension for key/value (can be smaller than head_dim)
        self.feature_dim = feature_dim if feature_dim is not None else self.head_dim

        # Projections
        self.k_proj = nn.Linear(d_model, num_heads * self.feature_dim, bias=False)
        self.q_proj = nn.Linear(d_model, num_heads * self.feature_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)

        # Gate projection (low-rank parameterization)
        self.gate_proj = nn.Sequential(
            nn.Linear(d_model, gate_low_rank, bias=False),
            nn.Linear(gate_low_rank, self.head_dim, bias=True)
        )

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Configuration
        self.chunk_size = chunk_size
        self.gate_low_rank_dim = gate_low_rank
        self.gate_logit_normalizer = gate_logit_normalizer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GLA layer.

        Args:
            hidden_states: Input tensor of shape (B, L, D)

        Returns:
            o: Output tensor of shape (B, L, D)
        """
        B, L, D = hidden_states.size()

        # Project to key, query, value
        k = self.k_proj(hidden_states)  # (B, L, H * D_k)
        q = self.q_proj(hidden_states)  # (B, L, H * D_k)
        v = self.v_proj(hidden_states)  # (B, L, H * D)

        # Compute gate (use logsigmoid for numerical stability, alpha in (-inf, 0))
        gate = self.gate_proj(hidden_states)  # (B, L, H * D_k)
        gate = F.logsigmoid(gate) / self.gate_logit_normalizer

        # Reshape to (B, L, H, D_k)
        k = k.view(B, L, self.num_heads, self.feature_dim)
        q = q.view(B, L, self.num_heads, self.feature_dim)
        alpha = gate.view(B, L, self.num_heads, self.feature_dim)

        # Reshape value to (B, L, H, D_v)
        v = v.view(B, L, self.num_heads, self.head_dim)

        # Initialize state as None (no caching for training)
        init_state = None

        # Chunkwise parallel mode for training
        o, _ = gla_chunkwise_parallel(
            k, q, v, alpha, init_state, chunk_size=self.chunk_size
        )

        # Reshape output: (B, L, H, D_v) -> (B, L, D)
        o = o.contiguous().view(B, L, D)

        # Output projection
        o = self.out_proj(o)

        return o

    def state_size(self, sequence_length: int = 0) -> int:
        """
        Compute the state size in bytes.

        State consists of:
        - S: (num_heads, feature_dim, head_dim) float32
        """
        state_size = (
            self.num_heads * self.feature_dim * self.head_dim
        )
        return state_size
