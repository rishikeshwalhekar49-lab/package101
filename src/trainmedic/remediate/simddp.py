"""A single-process simulation of data-parallel training with fault injection.

Lets students study distributed failure modes (a rank that skips the gradient
all-reduce, silent parameter corruption, NaN gradients on one rank) and the
Guardian's remediation without a multi-GPU cluster.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from ..detect.distributed import replicas_consistent

FAULTS = ("skip_allreduce", "corrupt_param", "nan_grad", "stale_grad")


@dataclass
class Fault:
    rank: int
    kind: str
    at_step: int | None = None
    once: bool = True
    fired: bool = False


class SimulatedDataParallel:
    def __init__(self, model: nn.Module, world_size: int = 2,
                 optimizer_factory: Callable[[Any], torch.optim.Optimizer] | None = None) -> None:
        self.world_size = world_size
        self.replicas = [copy.deepcopy(model) for _ in range(world_size)]
        factory = optimizer_factory or (lambda ps: torch.optim.SGD(ps, lr=0.05))
        self.optimizers = [factory(r.parameters()) for r in self.replicas]
        self.faults: list[Fault] = []
        self.step_count = 0
        self.log: list[dict[str, Any]] = []

    def inject(self, rank: int, kind: str, at_step: int | None = None, once: bool = True) -> None:
        if kind not in FAULTS:
            raise ValueError(f"fault must be one of {FAULTS}")
        self.faults.append(Fault(rank, kind, at_step, once))

    def _active(self, kind: str) -> list[int]:
        out = []
        for f in self.faults:
            if f.kind == kind and (f.at_step is None or f.at_step == self.step_count) and not (f.once and f.fired):
                out.append(f.rank)
                f.fired = True
                self.log.append({"step": self.step_count, "fault": kind, "rank": f.rank})
        return out

    def forward_backward(self, shards: list[Any], loss_fn: Callable[[nn.Module, Any], torch.Tensor]) -> list[float]:
        losses = []
        for m, opt, shard in zip(self.replicas, self.optimizers, shards):
            opt.zero_grad()
            loss = loss_fn(m, shard)
            loss.backward()
            losses.append(float(loss.detach()))
        return losses

    def allreduce(self) -> None:
        skip = set(self._active("skip_allreduce"))
        ranks = [r for r in range(self.world_size) if r not in skip]
        with torch.no_grad():
            for params in zip(*(m.parameters() for m in self.replicas)):
                grads = [params[r].grad for r in ranks if params[r].grad is not None]
                if not grads:
                    continue
                avg = torch.stack(grads).mean(0)
                for r in ranks:
                    if params[r].grad is not None:
                        params[r].grad.copy_(avg)
            for r in self._active("nan_grad"):
                for p in self.replicas[r].parameters():
                    if p.grad is not None:
                        p.grad.fill_(float("nan"))
                        break

    def optimizer_step(self) -> None:
        for opt in self.optimizers:
            opt.step()
        with torch.no_grad():
            for r in self._active("corrupt_param"):
                p = next(self.replicas[r].parameters())
                p.add_(torch.randn_like(p) * p.abs().mean().clamp_min(1e-3))
        self.step_count += 1

    def step(self, shards: list[Any], loss_fn: Callable[[nn.Module, Any], torch.Tensor]) -> list[float]:
        losses = self.forward_backward(shards, loss_fn)
        self.allreduce()
        self.optimizer_step()
        return losses

    def consistent(self, rtol: float = 1e-6) -> bool:
        return replicas_consistent(self.replicas, rtol)[0]

    def resync(self, source: int | None = None) -> int:
        """Copy parameters and optimizer state from ``source`` (default: majority checksum)."""
        if source is None:
            _, sums = replicas_consistent(self.replicas)
            votes: dict[float, list[int]] = {}
            for r, s in enumerate(sums):
                votes.setdefault(round(s, 6), []).append(r)
            source = max(votes.values(), key=len)[0]
        state = copy.deepcopy(self.replicas[source].state_dict())
        opt_state = copy.deepcopy(self.optimizers[source].state_dict())
        for r in range(self.world_size):
            if r != source:
                self.replicas[r].load_state_dict(state)
                self.optimizers[r].load_state_dict(opt_state)
        return source
