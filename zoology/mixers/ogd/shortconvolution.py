import torch.nn as nn
import torch
from torch.nn import functional as F


class ShortConv(nn.Module):
    def __init__(self, kernel_size: int, n_embd: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=n_embd,
            out_channels=n_embd,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=n_embd,
            bias=False,
        )
        self.n_embd = n_embd

    def forward(self, x):
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.conv(x)[..., :x.size(2)]  # (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C)
        return x