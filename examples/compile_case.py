"""Factory for `trainmedic compile-diff examples/compile_case.py:make_case`."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        return x + self.ff(self.ln(x))


def make_case():
    torch.manual_seed(0)
    model = nn.Sequential(Block(), Block(), nn.Linear(32, 5))
    x = torch.randn(8, 32)
    y = torch.randint(0, 5, (8,))
    return {"model": model, "inputs": (x,), "loss_fn": F.cross_entropy, "target": y}
