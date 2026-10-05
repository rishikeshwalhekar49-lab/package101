"""Behavioural probes: small experiments on a model that expose silent bugs.

These are used automatically by the tracer (on the first training forward of
each model) and can also be called directly:

* :func:`eval_determinism` -- in eval mode the output must not depend on the RNG
  (catches ``F.dropout`` without ``training=self.training``).
* :func:`batch_independence` -- sample *i*'s output must not depend on sample *j*
  (catches wrong ``view``/``permute`` that mixes the batch dimension).
* :func:`causality` -- for sequence models, position *t* must not depend on
  tokens after *t* (catches missing/inverted causal masks).
* :func:`overfit_one_batch` -- a healthy model+loss+optimizer can drive the loss
  on a single batch close to zero (Karpathy's recipe).
* :func:`kv_cache_consistency` -- incremental decoding must match a full forward.
* :func:`initial_loss_check` -- CE loss at init should be about ln(num_classes).
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable

import torch
import torch.nn as nn

from .tracer import _first_tensor

TOL = 1e-5


def _slice_args(args: tuple, n: int) -> tuple[torch.Tensor, tuple]:
    x = args[0]
    b = x.size(0)
    rest = tuple(a[:n] if isinstance(a, torch.Tensor) and a.dim() > 0 and a.size(0) == b else a for a in args[1:])
    return x[:n], rest


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().float(), b.detach().float()
    scale = max(float(a.abs().max()), float(b.abs().max()), 1e-6)
    return float((a - b).abs().max()) / scale


def _vocab(x: torch.Tensor) -> int:
    return max(int(x.max()) + 1, 2)


def eval_determinism(module: nn.Module, args: tuple) -> dict[str, Any]:
    x, rest = _slice_args(args, 4)
    with torch.no_grad():
        torch.manual_seed(1234)
        o1 = _first_tensor(module(x, *rest), x.size(0))
        torch.manual_seed(98765)
        o2 = _first_tensor(module(x, *rest), x.size(0))
    if o1 is None or o2 is None or not o1.is_floating_point():
        return {"probe": "eval_determinism", "passed": True, "score": 0.0, "detail": "no comparable output"}
    d = _rel(o1, o2)
    return {
        "probe": "eval_determinism", "passed": d < 1e-4, "score": d,
        "detail": "outputs differ between two eval-mode forwards with different RNG seeds" if d >= 1e-4 else "",
    }


def batch_independence(module: nn.Module, args: tuple) -> dict[str, Any]:
    x, rest = _slice_args(args, 4)
    if x.size(0) < 2:
        return {"probe": "batch_independence", "passed": True, "score": 0.0, "detail": "batch of 1"}
    if x.is_floating_point():
        with torch.enable_grad():
            xp = x.detach().clone().requires_grad_(True)
            o = _first_tensor(module(xp, *rest), x.size(0))
            if o is None or not o.is_floating_point() or not o.requires_grad:
                return {"probe": "batch_independence", "passed": True, "score": 0.0, "detail": "output not differentiable"}
            (g,) = torch.autograd.grad(o[0].float().sum(), xp, allow_unused=True)
        if g is None:
            return {"probe": "batch_independence", "passed": True, "score": 0.0, "detail": "no input gradient"}
        own = float(g[0].abs().sum())
        leak = float(g[1:].abs().sum())
        score = leak / (own + 1e-12)
        passed = leak < 1e-8 or score < 1e-4
        kind = "float"
    else:
        x2 = x.clone()
        x2[1:] = (x2[1:] + 1) % _vocab(x)
        with torch.no_grad():
            o1 = _first_tensor(module(x, *rest), x.size(0))
            o2 = _first_tensor(module(x2, *rest), x.size(0))
        if o1 is None or o2 is None or not o1.is_floating_point():
            return {"probe": "batch_independence", "passed": True, "score": 0.0, "detail": "no comparable output"}
        score = _rel(o1[0], o2[0])
        passed = score < TOL
        kind = "int"
    return {
        "probe": "batch_independence", "passed": passed, "score": score, "input_kind": kind,
        "detail": "" if passed else "output for sample 0 changes when other samples in the batch change",
    }


def causality(module: nn.Module, args: tuple) -> dict[str, Any] | None:
    x, rest = _slice_args(args, 2)
    if x.dim() < 2 or x.size(1) < 3:
        return None
    with torch.no_grad():
        o1 = _first_tensor(module(x, *rest), x.size(0))
        if o1 is None or o1.dim() < 2 or o1.size(1) != x.size(1) or not o1.is_floating_point():
            return None
        x2 = x.clone()
        if x.is_floating_point():
            if x.dim() < 3:
                return None
            gen = torch.Generator(device="cpu").manual_seed(7)
            x2[:, -1] = x2[:, -1] + torch.randn(x2[:, -1].shape, generator=gen).to(x2.device, x2.dtype)
            kind = "float_sequence"
        else:
            x2[:, -1] = (x2[:, -1] + 1) % _vocab(x)
            kind = "token_sequence"
        o2 = _first_tensor(module(x2, *rest), x.size(0))
    score = _rel(o1[:, :-1], o2[:, :-1])
    passed = score < TOL
    return {
        "probe": "causality", "passed": passed, "score": score, "input_kind": kind,
        "detail": "" if passed else "outputs at earlier positions change when only the last position changes (future leaks into past)",
    }


def run_all(module: nn.Module, args: tuple) -> list[dict[str, Any]]:
    out = [eval_determinism(module, args), batch_independence(module, args)]
    c = causality(module, args)
    if c is not None:
        out.append(c)
    return out


# --------------------------------------------------------------- user-facing
def overfit_one_batch(
    model: nn.Module,
    batch: tuple,
    loss_fn: Callable[[Any, Any], torch.Tensor],
    optimizer_factory: Callable[[Any], torch.optim.Optimizer] | None = None,
    steps: int = 200,
    target_ratio: float = 0.05,
) -> dict[str, Any]:
    """Train a *copy* of ``model`` on one batch; report whether the loss collapses.

    A pipeline that cannot overfit a single batch has a wiring bug (detached
    graph, wrong targets, frozen parameters, LR of 0, ...)."""
    m = copy.deepcopy(model)
    m.train()
    opt = optimizer_factory(m.parameters()) if optimizer_factory else torch.optim.Adam(m.parameters(), lr=1e-3)
    x, y = batch[0], batch[1]
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(m(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    first, last = losses[0], min(losses[-10:])
    ok = math.isfinite(last) and (last <= first * target_ratio or last < 1e-3)
    return {"probe": "overfit_one_batch", "passed": ok, "first_loss": first, "final_loss": last, "losses": losses}


def initial_loss_check(loss_value: float, num_classes: int, tolerance: float = 2.0) -> dict[str, Any]:
    expected = math.log(num_classes)
    ratio = loss_value / expected if expected > 0 else float("inf")
    return {
        "probe": "initial_loss", "passed": 1 / tolerance <= ratio <= tolerance, "expected": expected,
        "observed": loss_value, "ratio": ratio,
    }


def kv_cache_consistency(
    full_forward: Callable[[torch.Tensor], torch.Tensor],
    incremental_forward: Callable[[torch.Tensor], torch.Tensor],
    tokens: torch.Tensor,
    tol: float = 1e-4,
) -> dict[str, Any]:
    """Compare logits of a full forward with token-by-token cached decoding.

    ``incremental_forward(tokens)`` must return logits of shape (B, T, V) built by
    feeding one token at a time through the model's KV cache."""
    with torch.no_grad():
        a = full_forward(tokens)
        b = incremental_forward(tokens)
    per_pos = [(a[:, t] - b[:, t]).abs().max().item() for t in range(a.size(1))]
    first_bad = next((t for t, d in enumerate(per_pos) if d > tol * max(1.0, a.abs().max().item())), None)
    return {"probe": "kv_cache_consistency", "passed": first_bad is None, "first_bad_position": first_bad, "per_position": per_pos}
