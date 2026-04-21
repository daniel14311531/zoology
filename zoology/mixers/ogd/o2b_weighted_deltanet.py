import torch
import torch.nn as nn
import torch.nn.functional as F
from .shortconvolution import ShortConv
from .norm import RMSNorm
from .rotary import RotaryEmbedding
from typing import Literal
from .delta_rule import delta_rule

CHUNK_SIZE = 64

O2B_state = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def calc_inv(T: torch.Tensor):
    """
    Calculate the combined inverse of a strictly lower triangular tensor.

    Args:
        T: Input tensor of shape (..., C, C), with strictly lower triangular structure

    Returns:
        Inverse of (I + T)^{-1}
    """
    B, H, C, _ = T.size()
    dtype = T.dtype
    res = T + torch.eye(C, device=T.device, dtype=dtype).unsqueeze(0).unsqueeze(0)  # (B, H, C, C)
    return torch.linalg.inv(res.float()).to(dtype)
    # Alternative implementation using exponentiation by squaring:
    # C = T.shape[-1]
    # I = torch.eye(C, device=T.device, dtype=T.dtype).unsqueeze(0).unsqueeze(0)
    # res = I
    # mu = -T
    # base = I
    # t = C
    # while t > 0:
    #     res = mu @ res + res
    #     mu = mu @ mu
    #     t /= 2
    # return res


def o2b_weighted_delta_rule(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    init_state: O2B_state,
):
    """
    Chunkwise parallel version of online-to-batch delta rule.

    Args:
        k: Key tensor of shape (B, L, H, D)
        q: Query tensor of shape (B, L, H, D)
        v: Value tensor of shape (B, L, H, D)
        b: combined tensor of shape (B, L, H, D)
        init_state: (W_t, W_avg, t) where
            W_t: current weight matrix (B, H, D, D)
            W_avg: running average of W_t (B, H, D, D)
            t: current step count

    Returns:
        Output tensor of shape (B, L, H, D)
        new_state: Updated (W_t, W_avg, t)
    """
    B, L, H, D = k.shape
    if init_state is not None:
        W_t, W_avg, t = init_state
    else:
        W_t = torch.zeros(B, H, D, D, dtype=torch.float32, device=k.device)
        W_avg = torch.zeros(B, H, D, D, dtype=torch.float32, device=k.device)
        t = torch.zeros(B, dtype=torch.long, device=k.device)
    C = CHUNK_SIZE
    num_chunks = (L + C - 1) // C
    output_dtype = k.dtype
    state_dtype = torch.float32  # State always float32 for stability

    # Ensure state is float32
    if W_t.dtype != state_dtype:
        W_t = W_t.to(state_dtype)
    if W_avg.dtype != state_dtype:
        W_avg = W_avg.to(state_dtype)

    outputs = []

    for i in range(num_chunks):
        start = i * C
        end = min(start + C, L)
        actual_C = end - start

        # Get chunk data: convert to float32 for critical computations
        K_i = k[:, start:end].transpose(1, 2).to(state_dtype)
        Q_i = q[:, start:end].transpose(1, 2).to(state_dtype)
        V_i = v[:, start:end].transpose(1, 2).to(state_dtype)
        B_i = b[:, start:end].transpose(1, 2).to(state_dtype)

        # U[i] = (I + tril(B[i]K[i]^T, -1))^{-1} (V[i] - B[i]W[i-1])
        # Matrix inversion MUST use float32 for numerical stability
        T = torch.tril(B_i @ K_i.transpose(-2, -1), diagonal=-1)
        inv = calc_inv(T)
        U_i = inv @ (V_i - B_i @ W_t)

        # W[i] = W[i-1] + K[i]^T U[i]
        W_new = W_t + K_i.transpose(-2, -1) @ U_i

        # Compute W_avg using matrix form (efficient parallel computation)
        # Formula: W_avg_new = t/(t+C) * W_avg + C/(t+C) * W_t + (K_i^T U_i) @ diag(w_i)
        # where w_i[j] = alpha[j:C] / (1 + ... + (t + C)), j = 0..C-1 (decreasing weights)
        t_start_f = t.to(torch.float32).view(-1, 1, 1, 1) + i * C  # (B, 1, 1, 1)
        actual_C_f = float(actual_C)

        idx = torch.arange(actual_C, device=k.device, dtype=torch.float32).view(1, 1, -1, 1) + 1 + t_start_f  # (B, 1, C, 1)
        weighted_idx = idx + idx.sum(dim=-2, keepdim=True) - idx.cumsum(dim=-2)  # (B, 1, C, 1)

        w_i = weighted_idx / ((1 + t_start_f + actual_C_f) * (t_start_f + actual_C_f) / 2)  # (B, 1, C, 1)

        coef1 = (t_start_f / (t_start_f + actual_C_f)) * ((1 + t_start_f) / (1 + t_start_f + actual_C_f))
        coef2 = (actual_C_f / (t_start_f + actual_C_f)) * ((t_start_f * 2 + 1 + actual_C_f) / (1 + t_start_f + actual_C_f))
        
        W_avg_new = (coef1 * W_avg + coef2 * W_t +
                     (K_i.transpose(-2, -1) @ (U_i * w_i)))

        # a[i], c[i] for output weighting
        # Compute a, c for output weighting
        a = ((1 + t_start_f) * t_start_f / 2) / (((1 + t_start_f) * t_start_f / 2) + idx.cumsum(dim=-2))  # (B, 1, C, 1)
        c = 1 - a
        # T[i](r,j) = alpha[j:r+1]/alpha[1:r+1] if r>=j else 0
        # Use per-batch time for correct computation
        r = idx.cumsum(dim=-2).view(B, 1, -1, 1)  # (B, 1, C, 1)
        j_idx = (idx.cumsum(dim=-2) - idx).view(B, 1, 1, -1)  # (B, 1, 1, C)
        # T_mat shape: (B, 1, C, C) - each batch uses its own time
        T_mat = torch.where(r >= j_idx,
                           (r - j_idx) / ((t_start_f * (1 + t_start_f) / 2).view(B, 1, 1, 1) + r),
                           torch.zeros_like(r))  # (B, 1, C, C)

        # O[i] = diag(a)Q[i]W_avg[i-1] + diag(c)Q[i]W[i-1] + (T[i] o (Q[i]K[i]^T))U[i]
        QW_avg = Q_i @ W_avg
        QW_t = Q_i @ W_t
        QK_T = Q_i @ K_i.transpose(-2, -1)
        O_i = (QW_avg * a +
               QW_t * c +
               (T_mat * QK_T) @ U_i)

        outputs.append(O_i.to(output_dtype).transpose(1, 2))
        W_t, W_avg = W_new, W_avg_new

    o = torch.cat(outputs, dim=1)

    # Return state in float32
    new_state = (W_t, W_avg, t + L)

    return o, new_state


class O2BWeightedDeltaNetLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        bias: bool = True,
        layer_idx: int = None,
        conv_size: int = 4,
        use_qk_activation: bool = False,
        eta: float = 1.0,
        sync_kv_scale: bool = False,
        use_rope: bool = False,
        ogd_mode: Literal["deltanet", "ogd", "conceptual"] = "deltanet",
        **kwargs
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.n_embd = d_model
        self.n_head = num_heads
        self.head_dim = d_model // num_heads
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        self.beta_proj = nn.Linear(d_model, num_heads, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.k_conv1d = ShortConv(conv_size, d_model)
        self.q_conv1d = ShortConv(conv_size, d_model)
        self.v_conv1d = ShortConv(conv_size, d_model)

        self.eta = eta
        self.use_qk_activation = use_qk_activation
        self.sync_kv_scale = sync_kv_scale
        self.use_rope = use_rope
        self.ogd_mode = ogd_mode

        if self.use_rope:
            self.rotary = RotaryEmbedding(dim=self.head_dim)

    def forward(self, x):
        B, L, D = x.size()
        k = self.k_conv1d(self.k_proj(x))
        q = self.q_conv1d(self.q_proj(x))
        v = self.v_conv1d(self.v_proj(x))
        beta = torch.sigmoid(self.beta_proj(x))
        if self.use_qk_activation:
            k = F.silu(k)
            q = F.silu(q)

        k = k.view(B, L, self.n_head, self.head_dim)
        q = q.view(B, L, self.n_head, self.head_dim)
        v = v.view(B, L, self.n_head, self.head_dim)
        beta = beta.view(B, L, self.n_head)
        v = F.silu(v)

        # Apply RoPE before normalization
        if self.use_rope:
            k = k.transpose(1, 2)  # (B, H, L, D)
            q = q.transpose(1, 2)  # (B, H, L, D)
            k = self.rotary(k, offset=0, seq_len=L)
            q = self.rotary(q, offset=0, seq_len=L)
            k = k.transpose(1, 2)  # (B, L, H, D)
            q = q.transpose(1, 2)  # (B, L, H, D)

        knorm = torch.norm(k, dim=-1, keepdim=True)  # (B, L, n_head, 1)
        qnorm = torch.norm(q, dim=-1, keepdim=True)  # (B, L, n_head, 1)
        k = k / (knorm + 1e-6)
        if self.sync_kv_scale:
            v = v / (knorm + 1e-6)
        q = q / (qnorm + 1e-6)

#         Initialize O2B state: (W_t, W_avg, t)
        init_state = (
            torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32),
            torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32),
            torch.zeros(B, dtype=torch.long, device=x.device),
        )
#         init_state = torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32)

        if self.ogd_mode == "deltanet":
            eta = self.eta * beta
        else:
            k_norm2 = torch.sum(k ** 2, dim=-1)  # (B, L, n_head)
            eta = self.eta * beta / (1 + self.eta * beta * k_norm2)
        
        b = eta.contiguous().view(B, L, self.n_head, 1) * k
        v = eta.contiguous().view(B, L, self.n_head, 1) * v

        o, _ = o2b_weighted_delta_rule(
            k=k,
            q=q,
            v=v,
            b=b,
            init_state=init_state
        )
#         o = delta_rule(
#             k=k,
#             q=q,
#             v=v,
#             beta=eta,
#             init_state=init_state
#         )

        o = self.out_norm(o)
        o = o.contiguous().view(B, L, D)
        o = self.out_proj(o)

        return o

    def state_size(self, sequence_length: int=2048):
        # O2B DeltaNet state: (W_t, W_avg, t)
        # W_t: (H, D, D), W_avg: (H, D, D)
        state_size = (
            2 * self.n_head * self.head_dim * self.head_dim
        )
        return state_size
