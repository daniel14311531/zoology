import torch
import torch.nn as nn
import torch.nn.functional as F
from .ogd.shortconvolution import ShortConv
from .ogd.norm import RMSNorm
import math

CHUNK_SIZE = 64


def ls_chunk_parallel(
    state_KK: torch.Tensor,
    state_KV: torch.Tensor,
    K: torch.Tensor,
    Q: torch.Tensor,
    V: torch.Tensor,
    BETA: torch.Tensor,
    cumsum_G: torch.Tensor,
):
    """
    Parallel computation for a chunk:
    For each position j in chunk, solve state_KK_j @ q_star_j = q_j
    and compute o_j = q_star_j @ state_KV_j.

    This implementation uses full parallelization within the chunk.

    Args:
        state_KK: Previous state_KK of shape (B, H, D, D)
        state_KV: Previous state_KV of shape (B, H, D, D)
        K: Key tensor of shape (B, H, C, D)
        Q: Query tensor of shape (B, H, C, D)
        V: Value tensor of shape (B, H, C, D)
        BETA: Beta tensor of shape (B, H, C)
        cumsum_G: Cumulative sum of gate tensor of shape (B, H, C)

    Returns:
        O: Output tensor of shape (B, H, C, D)
        state_KK_end: Final state_KK after processing chunk
        state_KV_end: Final state_KV after processing chunk
    """
    B, H, C, D = K.shape
    device = K.device

    # Precompute KTK[m] = K[m]^T @ K[m] * beta[m]
    KTK = torch.einsum('bhcd,bhce->bhdec', K, K)  # (B, H, D, D, C)
    KTK = KTK * BETA.unsqueeze(-2).unsqueeze(-2)  # Apply beta scaling
    KTK = KTK.permute(0, 1, 4, 2, 3)  # (B, H, C, D, D)

    # Precompute KTV[m] = K[m]^T @ V[m] * beta[m]
    KTV = torch.einsum('bhcd,bhce->bhdec', K, V)  # (B, H, D, D, C)
    KTV = KTV * BETA.unsqueeze(-2).unsqueeze(-2)  # Apply beta scaling
    KTV = KTV.permute(0, 1, 4, 2, 3)  # (B, H, C, D, D)

    # Create decay matrix: decay[j, m] = exp(cumsum_G[j] - cumsum_G[m]) for j >= m, else 0
    log_decay = cumsum_G.unsqueeze(-1) - cumsum_G.unsqueeze(-2)
    decay = torch.tril(log_decay.exp())

    # Compute cumulative KTK and KTV using parallel matrix multiplication
    KTK_cum = torch.einsum('bhjm,bhmde->bhjde', decay, KTK)
    KTV_cum = torch.einsum('bhjm,bhmde->bhjde', decay, KTV)

    # state_KK[j] = state_KK * exp(cumsum_G[j]) + KTK_cum[j]
    exp_cumsum = cumsum_G.exp()
    state_KK_all = state_KK.unsqueeze(2) * exp_cumsum.unsqueeze(-1).unsqueeze(-1) + KTK_cum
    state_KV_all = state_KV.unsqueeze(2) * exp_cumsum.unsqueeze(-1).unsqueeze(-1) + KTV_cum

    # Solve for each position j: state_KK_all[j] @ q_star_j = Q[j]
    state_KK_flat = state_KK_all.reshape(B * H * C, D, D)
    Q_flat = Q.reshape(B * H * C, D)
    q_star_flat = torch.linalg.solve(state_KK_flat, Q_flat)
    q_star_all = q_star_flat.reshape(B, H, C, D)

    # Compute output: o[j] = q_star_all[j] @ state_KV_all[j]
    O = torch.einsum('bhjd,bhjdc->bhjc', q_star_all, state_KV_all)

    # Final states for next chunk
    state_KK_end = state_KK_all[:, :, -1]
    state_KV_end = state_KV_all[:, :, -1]

    return O, state_KK_end, state_KV_end


def least_square_parallel(
    k: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state_KK: torch.Tensor = None,
    state_KV: torch.Tensor = None,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Parallel implementation of Least Square for inference.
    Chunks are processed sequentially, but computation within each chunk is fully parallel.

    Args:
        k: Key tensor of shape (B, L, H, D)
        q: Query tensor of shape (B, L, H, D)
        v: Value tensor of shape (B, L, H, D)
        g: Gate tensor of shape (B, L, H, D), values in (-inf, 0)
        beta: Beta tensor of shape (B, L, H)
        state_KK: S_prev_KK of shape (B, H, D, D)
        state_KV: S_prev_KV of shape (B, H, D, D)
        chunk_size: Size of each chunk

    Returns:
        o: Output tensor of shape (B, L, H, D)
        new_state_KK: S_new_KK of shape (B, H, D, D)
        new_state_KV: S_new_KV of shape (B, H, D, D)
    """
    B, L, H, D = k.shape
    if state_KK is None:
        state_KK = torch.zeros(B, H, D, D, device=k.device, dtype=torch.float32)
        state_KK = state_KK + torch.eye(D, device=k.device, dtype=torch.float32).view(1, 1, D, D)
    if state_KV is None:
        state_KV = torch.zeros(B, H, D, D, device=k.device, dtype=torch.float32)
    output = []

    num_chunks = (L + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, L)

        K_i = k[:, start:end].transpose(1, 2).float()  # (B, H, C, D)
        Q_i = q[:, start:end].transpose(1, 2).float()  # (B, H, C, D)
        V_i = v[:, start:end].transpose(1, 2).float()  # (B, H, C, D)
        G_i = g[:, start:end].transpose(1, 2).float()  # (B, H, C)
        BETA_i = beta[:, start:end].transpose(1, 2).float()  # (B, H, C)

        cumsum_G = torch.cumsum(G_i, dim=2)  # (B, H, C)

        # Parallel computation within chunk
        O_i, state_KK, state_KV = ls_chunk_parallel(
            state_KK, state_KV, K_i, Q_i, V_i, BETA_i, cumsum_G
        )
        output.append(O_i.to(k.dtype).transpose(1, 2))  # (B, H, C, D) -> (B, C, H, D)

    o = torch.cat(output, dim=1)
    return o, state_KK, state_KV


class LeastSquareLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 2,
        bias: bool = True,
        layer_idx: int = None,
        conv_size: int = 4,
        use_qk_activation: bool = True,
        eta: float = 1.0,
        sync_kv_scale: bool = False,
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
        
        A = torch.empty(num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        self.k_conv1d = ShortConv(conv_size, d_model)
        self.q_conv1d = ShortConv(conv_size, d_model)
        self.v_conv1d = ShortConv(conv_size, d_model)
        
        self.eta = eta
        self.use_qk_activation = use_qk_activation
        self.sync_kv_scale = sync_kv_scale

    def forward(
        self,
        x: torch.Tensor,
    ):
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

        knorm = torch.norm(k, dim=-1, keepdim=True)  # (B, L, n_head, 1)
        qnorm = torch.norm(q, dim=-1, keepdim=True)  # (B, L, n_head, 1)
        k = k / (knorm + 1e-6)
        if self.sync_kv_scale:
            v = v / (knorm + 1e-6)
        q = q / (qnorm + 1e-6)

        o, state_KK, state_KV = least_square_parallel(
            k = k,
            q = q,
            v = v,
            g = g,
            beta = self.eta * beta,
        )

        o = self.out_norm(o)
        o = o.contiguous().view(B, L, D)
        o = self.out_proj(o)

        return o
    
    def state_size(self, sequence_length: int=2048):
        state_size = (
            2 * self.n_head * self.head_dim * self.head_dim
        )
        return state_size 