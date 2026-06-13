# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
except:
    assert 0, print(f"Need to install fla: pip install flash-linear-attention")

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class KimiDeltaAttention(nn.Module):
    """
    Kimi Delta Attention (KDA) layer implementation.

    KDA extends the delta rule with per-key-dim gating: the forget gate ``g`` has shape ``[B, T, H, K]``
    (vector per head), compared to GDN's scalar per-head gate ``[B, T, H]``.

    Args:
        d_model (int): Hidden size of the input.
        expand_v (float): Expansion ratio for the value dimension. Default: 1.0.
        num_heads (int): Number of heads. Default: 16.
        num_v_heads (int): Number of value heads (GVA if > num_heads). Default: None (= num_heads).
        mode (str): Kernel mode, `chunk` or `fused_recurrent`. Default: `chunk`.
        use_short_conv (bool): Whether to use short convolutions. Default: True.
        allow_neg_eigval (bool): Allow negative eigenvalues (beta *= 2). Default: False.
        safe_gate (bool): Assume gate values in [lower_bound, 0) for M=16 TensorCore. Default: False.
        lower_bound (float): Lower bound for forget gate in log space. Default: None.
        conv_size (int): Short conv kernel size. Default: 4.
        conv_bias (bool): Short conv bias. Default: False.
        layer_idx (int): Layer index. Default: None.
        norm_eps (float): Norm epsilon. Default: 1e-5.
    """

    def __init__(
        self,
        d_model: int = 2048,
        expand_v: float = 1,
        num_heads: int = 16,
        num_v_heads: Optional[int] = None,
        mode: str = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: Optional[float] = None,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> KimiDeltaAttention:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.safe_gate = safe_gate
        self.lower_bound = lower_bound

        hidden_size = int(d_model)
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        # Compute head_dim from d_model (zoology convention)
        self.head_dim = hidden_size // self.num_heads
        head_dim = self.head_dim
        self.head_k_dim = head_dim
        self.head_v_dim = int(head_dim * expand_v)
        self.key_dim = int(self.num_heads * head_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        if use_short_conv:
            self.q_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.k_conv1d = ShortConvolution(self.key_dim, conv_size, activation='silu')
            self.v_conv1d = ShortConvolution(self.value_dim, conv_size, activation='silu')

        # Gate dim = num_v_heads * head_k_dim: per value-head, per key-dim gating
        self.gate_dim = int(self.num_v_heads * self.head_k_dim)
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.gate_dim, bias=False),
        )
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        # A_log and dt_bias are per value-head for native GVA support
        if safe_gate:
            self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        else:
            self.A_log = nn.Parameter(
                torch.log(torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(1, 16))
            )
        self.A_log._no_weight_decay = True
        dt = torch.exp(
            torch.rand(self.gate_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormGated(self.head_v_dim, activation="sigmoid", eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[Dict],
    ) -> torch.Tensor:
        # Switch to fused_recurrent for short sequences during inference
        mode = "fused_recurrent" if (hidden_states.shape[1] <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            conv_mask = attention_mask[:, -hidden_states.shape[1]:] if attention_mask is not None else None
            position_ids = kwargs.get('position_ids', None)
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states), mask=conv_mask, cache=conv_state_q,
                output_final_state=use_cache, seq_idx=position_ids,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states), mask=conv_mask, cache=conv_state_k,
                output_final_state=use_cache, seq_idx=position_ids,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states), mask=conv_mask, cache=conv_state_v,
                output_final_state=use_cache, seq_idx=position_ids,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        g = self.f_proj(hidden_states)
        beta = self.b_proj(hidden_states).sigmoid()

        q, k = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k))
        # g and v are at value-head dimension (HV); q/k are at qk-head dimension (H)
        g = rearrange(g, "... (h d) -> ... h d", d=self.head_k_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.allow_neg_eigval:
            beta = beta * 2.0

        # breakpoint()
        # convert everything to bf16 
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        beta = beta.to(torch.bfloat16)
        g = g.to(torch.bfloat16)

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        cu_seqlens = kwargs.get('cu_seqlens', None)
        if mode == "chunk":
            o, recurrent_state = chunk_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=recurrent_state, output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                safe_gate=self.safe_gate, lower_bound=self.lower_bound,
                cu_seqlens=cu_seqlens,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_kda(
                q=q, k=k, v=v, g=g, beta=beta,
                A_log=self.A_log, dt_bias=self.dt_bias,
                initial_state=recurrent_state, output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                lower_bound=self.lower_bound,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")
        
        o = o.float()

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx, offset=q.shape[1],
            )

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        return o
