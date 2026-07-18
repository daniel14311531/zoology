import torch
import torch.nn as nn
import torch.nn.functional as F
from .shortconvolution import ShortConv
from .norm import RMSNorm
from .rotary import RotaryEmbedding
from typing import Literal
import math


discounted_O2B_decayed_state = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

CHUNK_SIZE = 64


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


def discounted_o2b_decayed_delta_rule_parallel(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    g: torch.Tensor,
    init_state: discounted_O2B_decayed_state,
):
    """
    Chunkwise parallel version of discounted online-to-batch decayed delta rule.

    Args:
        k: Key tensor of shape (B, L, H, D)
        q: Query tensor of shape (B, L, H, D)
        v: Value tensor of shape (B, L, H, D)
        b: Combined tensor of shape (B, L, H, D)
        g: Log-gamma tensor of shape (B, L, H), gamma = exp(g)
        init_state: (W_t, S_t, z_t)

    Returns:
        o: Output tensor of shape (B, L, H, D)
        new_state: Updated (W_t, S_t, z_t)
    """
    B, L, H, D = k.shape
    if init_state is not None:
        W_t, S_t, z_t = init_state
    else:
        W_t = torch.zeros(B, H, D, D, dtype=torch.float32, device=k.device)
        S_t = torch.zeros(B, H, D, D, dtype=torch.float32, device=k.device)
        z_t = torch.zeros(B, H, dtype=torch.float32, device=k.device)
    C = CHUNK_SIZE
    num_chunks = (L + C - 1) // C
    output_dtype = k.dtype
    state_dtype = torch.float32

    if W_t.dtype != state_dtype:
        W_t = W_t.to(state_dtype)
    if S_t.dtype != state_dtype:
        S_t = S_t.to(state_dtype)
    if z_t.dtype != state_dtype:
        z_t = z_t.to(state_dtype)

    outputs = []

    for i in range(num_chunks):
        start = i * C
        end = min(start + C, L)
        actual_C = end - start

        # Extract chunk data
        K_i = k[:, start:end].transpose(1, 2).to(state_dtype)  # (B, H, C, D)
        Q_i = q[:, start:end].transpose(1, 2).to(state_dtype)
        V_i = v[:, start:end].transpose(1, 2).to(state_dtype)
        B_i = b[:, start:end].transpose(1, 2).to(state_dtype)
        G_log = g[:, start:end].float().transpose(1, 2)  # (B, H, C)

        # Cumulative gamma products
        cumG = torch.cumsum(G_log, dim=-1).exp()  # (B, H, C), cumG[j] = prod_{m=0}^{j} Gamma[m]
        p_i = cumG  # p_i[j] = cumG[j], used in U_i computation

        # g_i[j] = prod(Gamma[j+1:C]), suffix product with shift
        suffix_gamma = torch.cumsum(G_log.flip(-1), dim=-1).flip(-1).exp()  # suffix prod including self
        g_i = torch.cat([suffix_gamma[:, :, 1:], torch.ones(B, H, 1, device=k.device, dtype=state_dtype)], dim=-1)  # (B, H, C)

        # Build Gamma matrix for inversion: Gamma_mat[r,c] = cumG[r]/cumG[c] for r > c
        # This captures the full gamma product from c+1 to r (inclusive)
        log_cumG = torch.cumsum(G_log, dim=-1)  # (B, H, C)
        log_gamma_mat = log_cumG.unsqueeze(-1) - log_cumG.unsqueeze(-2)  # (B, H, C, C)
        Gamma_mat = torch.exp(log_gamma_mat) * torch.tril(torch.ones(actual_C, actual_C, device=k.device, dtype=state_dtype), diagonal=-1)

        # U_i = (I + tril(Gamma_mat * BK^T, -1))^{-1} (V_i - diag(p_i) * B_i @ W_t)
        BK = B_i @ K_i.transpose(-2, -1)  # (B, H, C, C)
        T_mat = torch.tril(Gamma_mat * BK, diagonal=-1)
        inv = calc_inv(T_mat)
        U_i = inv @ (V_i - p_i.unsqueeze(-1) * (B_i @ W_t))  # (B, H, C, D)

        # W_new = gamma_chunk * W_t + K^T @ (U_i * g_i)
        gamma_chunk = torch.sum(G_log, dim=-1, keepdim=True).unsqueeze(-1).exp()  # (B, H, 1, 1)
        W_new = gamma_chunk * W_t + K_i.transpose(-2, -1) @ (U_i * g_i.unsqueeze(-1))

        # z_within: parallel computation via suffix_prod + prefix_sum on CxC matrix
        # M[j, k] = Gamma_i[k+1] for k < j, 1 for k >= j (column k uses gamma at position k+1)
        G_shifted = torch.cat([G_log[:, :, 1:], torch.zeros(B, H, 1, device=k.device, dtype=state_dtype)], dim=-1)  # (B, H, C)
        G_expand = G_shifted.unsqueeze(-2).expand(-1, -1, actual_C, actual_C)  # (B, H, C, C)
        mask = torch.tril(torch.ones(actual_C, actual_C, device=k.device), diagonal=-1).bool()
        G_mat = torch.where(mask, G_expand, torch.zeros_like(G_expand))

        suffix_prod = torch.cumsum(G_mat.flip(-1), dim=-1).flip(-1).exp()  # suffix product per row
        z_base = torch.diagonal(torch.cumsum(suffix_prod, dim=-1), dim1=-2, dim2=-1)  # (B, H, C)

        # z_j = cumG[j] * z_start + z_base[j]
        z_within = cumG * z_t.unsqueeze(-1) + z_base  # (B, H, C)
        z_end = z_within[:, :, -1]

        # S update: S_new = gamma_chunk * S_t + C * gamma_chunk * W_t + K^T @ (U_i * w_i)
        # w_i[j] = (C-j) * g_i[j], capturing within-chunk contribution to weighted sum
        pos_weights = torch.arange(actual_C, 0, -1, device=k.device, dtype=state_dtype).view(1, 1, actual_C)
        w_i = pos_weights * g_i  # (B, H, C)
        actual_C_f = float(actual_C)
        S_new = gamma_chunk * S_t + actual_C_f * gamma_chunk * W_t + K_i.transpose(-2, -1) @ (U_i * w_i.unsqueeze(-1))

        # Output coefficients
        # h_i = 1.0 / z_within  # (B, H, C)
        a = cumG * z_t.unsqueeze(-1) / z_within  # (B, H, C), cumG factor for gamma decay of S_start
        d = torch.arange(1, actual_C + 1, device=k.device, dtype=state_dtype).view(1, 1, actual_C) * cumG / z_within

        # T_out[r,c] = (r-c+1) * cumG[r]/cumG[c] / z_within[r] for r >= c (0-indexed)
        coeff = (torch.arange(actual_C, device=k.device, dtype=state_dtype).view(1, 1, -1, 1)
                 - torch.arange(actual_C, device=k.device, dtype=state_dtype).view(1, 1, 1, -1) + 1)
        # cumG[r]/cumG[c] = prod_{m=c+1}^{r} Gamma[m]
        log_prod_mat = log_cumG.unsqueeze(-1) - log_cumG.unsqueeze(-2)
        T_out = coeff * torch.exp(log_prod_mat) / z_within.unsqueeze(-1)  # (B, H, C, C)
        T_out = T_out * torch.tril(torch.ones(actual_C, actual_C, device=k.device))  # include diagonal

        # O_i = diag(a)*Q@W_avg + diag(d)*Q@W_t + (T_out*QK^T)@U_i
        z_safe = z_t.view(B, H, 1, 1).clamp(min=1e-10)  # avoid 0/0 when z_t=0
        W_avg = S_t / z_safe
        QK = Q_i @ K_i.transpose(-2, -1)
        O_i = (a.unsqueeze(-1) * (Q_i @ W_avg)
               + d.unsqueeze(-1) * (Q_i @ W_t)
               + (T_out * QK) @ U_i)

        outputs.append(O_i.to(output_dtype).transpose(1, 2))
        W_t, S_t, z_t = W_new, S_new, z_end

    o = torch.cat(outputs, dim=1)
    new_state = (W_t, S_t, z_t)

    return o, new_state


class DiscountedO2BDecayedDeltaNetLayer(nn.Module):
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

        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.a_proj = nn.Linear(d_model, num_heads, bias=bias)

        self.beta_proj = nn.Linear(d_model, num_heads, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.k_conv1d = ShortConv(conv_size, d_model)
        self.q_conv1d = ShortConv(conv_size, d_model)
        self.v_conv1d = ShortConv(conv_size, d_model)

        A = torch.empty(self.n_head, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.n_head) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True
        
        self.eta = eta
        self.use_qk_activation = use_qk_activation
        self.sync_kv_scale = sync_kv_scale
        self.ogd_mode = ogd_mode
        if self.ogd_mode == "ogd":
            raise NotImplementedError("OGD mode is not implemented, try 'deltanet' or 'conceptual'")
        self.use_RoPE = use_rope
        if self.use_RoPE:
            self.rotary = RotaryEmbedding(dim=self.head_dim)

    def forward(self, x):
        B, L, D = x.size()

        k = self.k_conv1d(self.k_proj(x))
        q = self.q_conv1d(self.q_proj(x))
        v = self.v_conv1d(self.v_proj(x))
        beta = torch.sigmoid(self.beta_proj(x))

        if self.use_qk_activation:
            k, q = F.silu(k), F.silu(q)

        k = k.view(B, L, self.n_head, self.head_dim)
        q = q.view(B, L, self.n_head, self.head_dim)
        v = v.view(B, L, self.n_head, self.head_dim)
        beta = beta.view(B, L, self.n_head)
        v = F.silu(v)

        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)

        # apply rotary embeddings
        if self.use_RoPE:
            k = k.transpose(1, 2)
            q = q.transpose(1, 2)
            k = self.rotary(k, seq_len=L)
            q = self.rotary(q, seq_len=L)
            k = k.transpose(1, 2)
            q = q.transpose(1, 2)

        knorm = torch.norm(k, dim=-1, keepdim=True)  # (B, L, n_head, 1)
        qnorm = torch.norm(q, dim=-1, keepdim=True)  # (B, L, n_head, 1)
        k = k / (knorm + 1e-6)
        if self.sync_kv_scale:
            v = v / (knorm + 1e-6)
        q = q / (qnorm + 1e-6)

        # State uses float32 for numerical stability
        cur_state = (
            torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32),
            torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32),
            torch.zeros(B, self.n_head, dtype=torch.float32, device=x.device),
        )

        if self.ogd_mode == "deltanet":
            eta = self.eta * beta
        else:
            k_norm2 = torch.sum(k ** 2, dim=-1)  # (B, L, n_head)
            eta = self.eta * beta / (1 + self.eta * beta * k_norm2)
        
        b = eta.contiguous().view(B, L, self.n_head, 1) * k
        v = eta.contiguous().view(B, L, self.n_head, 1) * v

        o, cur_state = discounted_o2b_decayed_delta_rule_parallel(
            k = k,
            q = q,
            v = v,
            b = b,
            g = g,
            init_state = cur_state
        )

        o = self.out_norm(o)
        o = o.contiguous().view(B, L, D)
        o = self.out_proj(o)

        return o
    
    def state_size(self, sequence_length: int=2048):
        # O2B DeltaNet state: (W_t, W_avg, t)
        # W_t: (H, D, D), W_avg: (H, D, D)
        state_size = (
            2 * self.n_head * self.head_dim * self.head_dim + self.n_head
        )
        return state_size