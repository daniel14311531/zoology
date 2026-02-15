import torch
import torch.nn as nn
import torch.nn.functional as F
from .shortconvolution import ShortConv
from .norm import RMSNorm
from .delta_rule import delta_rule

class DeltaNetLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 1,
        bias: bool = True,
        layer_idx: int = None,
        conv_size: int = 4,
        use_qk_activation: bool = False,
        initial_state: bool = False,
        eta: float = 1.0,
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
        
        self.initial_state = initial_state
        if self.initial_state:
            self.init_state = nn.Parameter(torch.zeros(1, num_heads, self.head_dim, self.head_dim))
        self.eta = eta
        self.use_qk_activation = use_qk_activation

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

        knorm = torch.norm(k, dim=-1, keepdim=True)  # (B, L, n_head, 1)
        qnorm = torch.norm(q, dim=-1, keepdim=True)  # (B, L, n_head, 1)
        k = k / (knorm + 1e-6)
        v = v / (knorm + 1e-6)  # use k's norm for v to maintain scale
        q = q / (qnorm + 1e-6)

        if self.initial_state:
            init_state = self.init_state.repeat(B, 1, 1, 1)
        else:
            init_state = torch.zeros(B, self.n_head, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)

        o = delta_rule(
            k = k,
            q = q,
            v = v,
            beta = self.eta * beta,
            init_state = init_state
        )

        o = self.out_norm(o)
        o = o.contiguous().view(B, L, D)
        o = self.out_proj(o)

        return o
    
    def state_size(self, sequence_length: int=2048):
        state_size = (
            self.n_head * self.head_dim * self.head_dim
        )
        return state_size 