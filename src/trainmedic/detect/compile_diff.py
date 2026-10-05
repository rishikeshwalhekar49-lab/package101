"""Eager-vs-compiled differential checking for training (a training-time "Ekka").

torch.compile miscompilations are a documented source of *silent* wrong results.
:func:`compile_diff` runs the same model/inputs eagerly and compiled, compares
outputs and parameter gradients, and -- on a mismatch -- localises the culprit
by compiling each sub-module in isolation on the inputs it saw during the eager
run (differential bisection over the module tree). The result attributes the
symptom to the compiler rather than to user code.
"""

from __future__ import annotations

import copy
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn

from ..types import Evidence, Severity


def _flat_tensors(x: Any) -> list[torch.Tensor]:
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, dict):
        x = list(x.values())
    if hasattr(x, "to_tuple"):
        x = x.to_tuple()
    if isinstance(x, (list, tuple)):
        out: list[torch.Tensor] = []
        for v in x:
            out.extend(_flat_tensors(v))
        return out
    return []


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double(), b.detach().double()
    if a.shape != b.shape:
        return float("inf")
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        return 0.0 if torch.equal(torch.nan_to_num(a), torch.nan_to_num(b)) else float("inf")
    denom = max(float(a.abs().max()) if a.numel() else 0.0, 1e-6)
    return float((a - b).abs().max()) / denom if a.numel() else 0.0


def _max_rel(xs: Any, ys: Any) -> float:
    ta, tb = _flat_tensors(xs), _flat_tensors(ys)
    if len(ta) != len(tb):
        return float("inf")
    return max((_rel_err(a, b) for a, b in zip(ta, tb) if a.is_floating_point()), default=0.0)


@dataclass
class CompileDiffReport:
    backend: str
    matches: bool = True
    output_rel_err: float = 0.0
    grad_mismatches: list[tuple[str, float]] = field(default_factory=list)
    module_mismatches: list[tuple[str, float]] = field(default_factory=list)
    culprits: list[str] = field(default_factory=list)
    error: str | None = None
    tolerance: float = 1e-3

    @property
    def attribution(self) -> str:
        if self.error:
            return "compiler_error"
        return "compiler" if not self.matches else "none"

    def to_evidence(self) -> list[Evidence]:
        if self.error:
            return [Evidence(id="", source="compile", rule="compile_error", message=f"torch.compile ({self.backend}) failed: {self.error}",
                             severity=Severity.WARNING, confidence=0.6, category="compiler", cause="compiler_mismatch")]
        if self.matches:
            return []
        where = f"; culprit module(s): {', '.join(self.culprits)}" if self.culprits else ""
        grads = f"; {len(self.grad_mismatches)} parameter gradient(s) differ" if self.grad_mismatches else ""
        return [Evidence(id="", source="compile", rule="eager_compiled_mismatch",
                         message=f"compiled ({self.backend}) and eager results differ (max relative error {self.output_rel_err:.3g}){grads}{where}",
                         severity=Severity.ERROR, confidence=0.85, category="compiler", cause="compiler_mismatch",
                         data={"culprits": self.culprits, "grad_mismatches": self.grad_mismatches[:20]})]

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "matches": self.matches, "output_rel_err": self.output_rel_err,
                "grad_mismatches": self.grad_mismatches, "module_mismatches": self.module_mismatches,
                "culprits": self.culprits, "error": self.error, "attribution": self.attribution}


def compile_diff(
    model: nn.Module,
    inputs: tuple,
    *,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
    target: Any = None,
    backend: str | Callable = "inductor",
    mode: str | None = None,
    tol: float = 1e-3,
    eval_mode: bool = True,
    localize: bool = True,
    seed: int = 0,
) -> CompileDiffReport:
    """Compare eager and compiled execution of ``model(*inputs)`` (and gradients if ``loss_fn`` is given).

    Use ``eval_mode=True`` (default) unless the model has no RNG-dependent layers:
    eager and compiled dropout draw different random numbers."""
    bname = backend if isinstance(backend, str) else getattr(backend, "__name__", "custom")
    rep = CompileDiffReport(backend=bname, tolerance=tol)
    m_e, m_c = copy.deepcopy(model), copy.deepcopy(model)
    for m in (m_e, m_c):
        m.train(not eval_mode)
    captured: dict[str, tuple] = {}
    handles = []
    if localize:
        for name, sub in m_e.named_modules():
            if name:
                handles.append(sub.register_forward_pre_hook(
                    lambda mod, args, _n=name: captured.setdefault(_n, tuple(a.detach().clone() if isinstance(a, torch.Tensor) else a for a in args))
                ))
    torch.manual_seed(seed)
    out_e = m_e(*inputs)
    for h in handles:
        h.remove()
    try:
        torch._dynamo.reset()
        cm = torch.compile(m_c, backend=backend, mode=mode)
        torch.manual_seed(seed)
        out_c = cm(*inputs)
    except Exception as exc:  # compile failures are evidence too
        rep.error = "".join(traceback.format_exception_only(type(exc), exc)).strip()[:500]
        rep.matches = False
        return rep
    rep.output_rel_err = _max_rel(out_e, out_c)
    rep.matches = rep.output_rel_err <= tol
    if loss_fn is not None:
        try:
            loss_fn(out_e, target).backward()
            loss_fn(out_c, target).backward()
            pe = dict(m_e.named_parameters())
            for n, p in m_c.named_parameters():
                q = pe.get(n)
                if p.grad is None or q is None or q.grad is None:
                    continue
                err = _rel_err(q.grad, p.grad)
                if err > tol:
                    rep.grad_mismatches.append((n, err))
            if rep.grad_mismatches:
                rep.matches = False
        except Exception as exc:
            rep.error = f"backward failed: {type(exc).__name__}: {exc}"[:500]
            rep.matches = False
    if not rep.matches and localize and captured:
        rep.module_mismatches = _localize(m_e, captured, backend, mode, tol, seed)
        bad = {n for n, _ in rep.module_mismatches}
        rep.culprits = sorted(n for n in bad if not any(o != n and o.startswith(n + ".") for o in bad))
    return rep


def _localize(model: nn.Module, captured: dict[str, tuple], backend, mode, tol: float, seed: int) -> list[tuple[str, float]]:
    mods = dict(model.named_modules())
    out = []
    for name in sorted(captured, key=lambda n: -n.count(".")):
        sub = mods.get(name)
        if sub is None:
            continue
        args = captured[name]
        try:
            torch.manual_seed(seed)
            a = sub(*args)
            torch._dynamo.reset()
            c = torch.compile(copy.deepcopy(sub), backend=backend, mode=mode)
            torch.manual_seed(seed)
            b = c(*args)
        except Exception:
            continue
        err = _max_rel(a, b)
        if err > tol:
            out.append((name, err))
    return out
