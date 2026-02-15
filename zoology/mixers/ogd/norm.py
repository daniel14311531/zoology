import torch.nn as nn
import torch
from torch.nn import functional as F

class RMSNorm(nn.Module):
    """RMSNorm implementation."""

    def __init__(self, n_embd, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.eps = eps

    def forward(self, x):
        norm_x = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * norm_x * self.weight