"""In-process use: watch a training loop for invariant violations, then protect
another one with the Guardian (rollback-and-skip on a corrupted batch, replica
resync in simulated data parallelism).

    python examples/watch_and_guard.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import trainmedic as tm
from trainmedic.remediate import Guardian, SimulatedDataParallel, run_guarded

torch.manual_seed(0)
X = torch.randn(2048, 16)
Y = (X[:, :4].sum(1) > 0).long()

# 1) watch(): a loop that forgets zero_grad and evaluates in train mode
model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))
opt = torch.optim.SGD(model.parameters(), lr=0.05)
with tm.watch(model, opt, max_steps=70) as w:
    for epoch in range(3):
        for i in range(0, 2048, 64):
            loss = F.cross_entropy(model(X[i:i + 64]), Y[i:i + 64])
            loss.backward()
            opt.step()  # zero_grad() is missing
        with torch.no_grad():  # model.eval() is missing
            model(X[:256])
print("watch() found:", sorted({e.rule for e in w.evidence}))

# 2) Guardian: batch 40 is corrupted (labels scaled by 1000 in a regression task)
reg = nn.Linear(16, 1)
target = X @ torch.randn(16, 1)
opt = torch.optim.Adam(reg.parameters(), lr=1e-2)


def get_batch(i):
    s = (i * 32) % 2048
    x, y = X[s:s + 32], target[s:s + 32]
    return (x, y * 1000) if i == 40 else (x, y)


g = Guardian(reg, opt, snapshot_every=10)
out = run_guarded(g, get_batch, lambda b: F.mse_loss(reg(b[0]), b[1]), total_steps=80)
print("guardian:", {k: v for k, v in out.items() if k != "history"})

# 3) simulated data parallelism: rank 1 silently skips the gradient all-reduce at step 3
sim = SimulatedDataParallel(nn.Linear(16, 1), world_size=4)
sim.inject(rank=1, kind="skip_allreduce", at_step=3)
g = Guardian(sim.replicas[0], sim.optimizers[0], replicas=sim.replicas, replica_optimizers=sim.optimizers,
             snapshot_every=0, check_code_invariants=False)
for step in range(6):
    shards = [(X[k * 32:(k + 1) * 32], target[k * 32:(k + 1) * 32]) for k in range(step * 4, step * 4 + 4)]
    losses = sim.step(shards, lambda m, b: F.mse_loss(m(b[0]), b[1]))
    d = g.post_step(step, sum(losses) / len(losses))
    if d.action.value != "continue":
        print(f"step {step}: {d.action.value} ({d.reason})")
print("replicas consistent:", sim.consistent())
