"""Benchmark pipeline (bug-free): DPO-style preference optimisation of a small policy.

Metric: top1_acc (higher is better) -- how often the policy's most likely response
is the truly best one on held-out contexts. Exercises a frozen reference policy,
the DPO log-ratio loss and an explicit KL penalty towards the reference.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import trainmedic as tm

D, K = 8, 8


def make_problem():
    g = torch.Generator().manual_seed(2024)
    W = torch.randn(K, D, generator=g)  # true reward r(c, k) = W[k] . c

    def contexts(n):
        return torch.randn(n, D, generator=g)

    def preferences(c):
        a = torch.randint(0, K, (c.size(0),), generator=g)
        b = (a + torch.randint(1, K, (c.size(0),), generator=g)) % K
        r = c @ W.t()
        ra = r.gather(1, a[:, None]).squeeze(1)
        rb = r.gather(1, b[:, None]).squeeze(1)
        a_wins = torch.rand(c.size(0), generator=g) < torch.sigmoid(2.0 * (ra - rb))
        return torch.where(a_wins, a, b), torch.where(a_wins, b, a)

    return W, contexts, preferences


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, 64), nn.Tanh(), nn.Linear(64, K))

    def forward(self, c):
        return F.log_softmax(self.net(c), dim=-1)


def main():
    tm.seed_everything(0)
    W, contexts, preferences = make_problem()
    train_c = contexts(4096)
    chosen_all, rejected_all = preferences(train_c)
    test_c = contexts(1024)
    policy = Policy()
    ref_model = copy.deepcopy(policy)
    ref_model.eval()
    ref_model.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-3)
    beta, kl_coef = 0.5, 0.05
    for epoch in range(tm.scaled(6)):
        perm = torch.randperm(train_c.size(0))
        for i in range(0, train_c.size(0), 128):
            idx = perm[i : i + 128]
            c, chosen, rejected = train_c[idx], chosen_all[idx], rejected_all[idx]
            logp = policy(c)
            with torch.no_grad():
                ref_logp = ref_model(c)
            pi_logratio = logp.gather(1, chosen[:, None]) - logp.gather(1, rejected[:, None])
            ref_logratio = ref_logp.gather(1, chosen[:, None]) - ref_logp.gather(1, rejected[:, None])
            dpo_loss = -F.logsigmoid(beta * (pi_logratio - ref_logratio)).mean()
            kl = (logp.exp() * (logp - ref_logp)).sum(-1).mean()
            loss = dpo_loss + kl_coef * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        best = (test_c @ W.t()).argmax(1)
        top1 = (policy(test_c).argmax(1) == best).float().mean().item()
    tm.report(top1_acc=top1)


if __name__ == "__main__":
    main()
