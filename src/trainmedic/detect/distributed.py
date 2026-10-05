"""Consistency checks across data-parallel replicas (real or simulated)."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from ..types import Evidence, Severity


def param_checksum(model: nn.Module, max_elems: int = 1_000_000) -> float:
    """Cheap order-sensitive checksum of a model's parameters (float64)."""
    total = 0.0
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            flat = p.detach().flatten()
            stride = max(1, flat.numel() // max_elems)
            s = flat[::stride].double()
            w = torch.arange(1, s.numel() + 1, dtype=torch.float64, device=s.device) % 97 + 1
            total += float((s * w).sum()) * (i + 1)
    return total


def optimizer_checksum(opt: torch.optim.Optimizer) -> float:
    total = 0.0
    with torch.no_grad():
        for i, st in enumerate(opt.state.values()):
            for v in st.values():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    total += float(v.double().sum()) * (i + 1)
    return total


def replicas_consistent(models: Iterable[nn.Module], rtol: float = 1e-6) -> tuple[bool, list[float]]:
    sums = [param_checksum(m) for m in models]
    ref = sums[0] if sums else 0.0
    ok = all(abs(s - ref) <= rtol * max(1.0, abs(ref)) for s in sums)
    return ok, sums


def check_ranks_consistent(model: nn.Module, group=None, rtol: float = 1e-6) -> list[Evidence]:
    """For real ``torch.distributed`` runs: all-gather parameter checksums and compare."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return []
    world = dist.get_world_size(group)
    local = torch.tensor([param_checksum(model)], dtype=torch.float64)
    if dist.get_backend(group) == "nccl":
        local = local.cuda()
    gathered = [torch.zeros_like(local) for _ in range(world)]
    dist.all_gather(gathered, local, group=group)
    sums = [float(g.item()) for g in gathered]
    ref = sums[0]
    bad = [r for r, s in enumerate(sums) if abs(s - ref) > rtol * max(1.0, abs(ref))]
    if not bad:
        return []
    return [Evidence(id="", source="distributed", rule="ranks_consistent",
                     message=f"parameters on rank(s) {bad} differ from rank 0 (checksums {sums})",
                     severity=Severity.ERROR, confidence=0.9, category="distributed", cause="rank_divergence")]
