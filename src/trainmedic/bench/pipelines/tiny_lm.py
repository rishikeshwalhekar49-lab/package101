"""Benchmark pipeline (bug-free): tiny causal transformer LM on Fibonacci-mod-V sequences.

Metric: gen_acc (higher is better) -- greedy continuation accuracy from a two-token
prompt. Exercises hand-written causal self-attention (score scaling, causal mask,
softmax axis), next-token target shifting, functional dropout, gradient
accumulation and a per-step warmup/cosine LR schedule.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import trainmedic as tm

V, T = 16, 20


def make_batch(n, g):
    seq = torch.zeros(n, T, dtype=torch.long)
    seq[:, 0] = torch.randint(0, V, (n,), generator=g)
    seq[:, 1] = torch.randint(0, V, (n,), generator=g)
    for t in range(2, T):
        seq[:, t] = (seq[:, t - 1] + seq[:, t - 2]) % V
    return seq


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_head, p):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.p = p
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))

    def forward(self, x):
        B, L, C = x.shape
        hd = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, L, self.n_head, hd).transpose(1, 2)
        k = k.view(B, L, self.n_head, hd).transpose(1, 2)
        v = v.view(B, L, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, self.p, training=self.training)
        y = (att @ v).transpose(1, 2).contiguous().view(B, L, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d, n_head, p):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, n_head, p)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, d=48, n_head=4, n_layer=2, p=0.0):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, n_head, p) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, idx):
        B, L = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(L, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln(x))


@torch.no_grad()
def generation_accuracy(model, g, n=256, prompt=2):
    model.eval()
    seq = make_batch(n, g)
    out = seq[:, :prompt]
    for _ in range(T - prompt):
        logits = model(out)
        out = torch.cat([out, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return (out[:, prompt:] == seq[:, prompt:]).float().mean().item()


def main():
    tm.seed_everything(0)
    g = torch.Generator().manual_seed(99)
    model = TinyLM(p=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    total_steps = tm.scaled(300)
    warmup = max(1, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps)),
    )
    accum = 2
    model.train()
    for step in range(total_steps):
        optimizer.zero_grad()
        for micro in range(accum):
            x = make_batch(32, g)
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1))
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    tm.report(gen_acc=generation_accuracy(model, torch.Generator().manual_seed(7)))


if __name__ == "__main__":
    main()
