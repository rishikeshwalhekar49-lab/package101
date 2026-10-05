"""Differential debugging against a reference run (buggy vs known-good).

Given a trace of a known-good version (before a refactor, a reference
implementation, the eager model) and a trace of the suspect version run with the
same seed, align them step by step and report the *first* signal that diverges:
train/eval mode, LR, loss, gradient norm, update size, loss-function input
statistics, call structure. The first divergence localises the bug far better
than the final metric does (cf. Ekka's aligned intermediate-state comparison).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..types import Evidence, Severity
from .trace import Trace


@dataclass
class Divergence:
    step: int
    signal: str
    reference: Any
    candidate: Any
    rel_diff: float | None = None


@dataclass
class TraceDiff:
    divergences: list[Divergence] = field(default_factory=list)
    compared_steps: int = 0

    @property
    def first(self) -> Divergence | None:
        return min(self.divergences, key=lambda d: d.step) if self.divergences else None

    def to_evidence(self) -> list[Evidence]:
        f = self.first
        if f is None:
            return []
        others = sorted({d.signal for d in self.divergences if d.step == f.step and d.signal != f.signal})
        msg = (f"first divergence from the reference run at step {f.step}: {f.signal} "
               f"(reference {_fmt(f.reference)}, this run {_fmt(f.candidate)})")
        if others:
            msg += f"; also diverging at that step: {', '.join(others)}"
        return [Evidence(id="", source="invariant", rule="reference_divergence", message=msg, severity=Severity.WARNING,
                         confidence=0.7, step=f.step, category="differential", cause=_cause(f.signal),
                         data={"signal": f.signal, "others": others})]


def _fmt(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def _cause(signal: str) -> str:
    return {
        "training_mode": "missing_train_mode",
        "eval_mode": "missing_eval_mode",
        "lr": "lr_schedule_misuse",
        "ce_input_probs": "double_softmax",
        "backwards_per_step": "grad_accum_unscaled",
        "zero_grad_calls": "missing_zero_grad",
    }.get(signal, "unknown")


def _per_step(trace: Trace) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [{}]
    cur = steps[0]
    for e in trace.events:
        k = e["kind"]
        if k == "forward" and e.get("grad_enabled"):
            cur.setdefault("training_mode", e.get("training"))
        elif k == "forward":
            cur.setdefault("eval_mode", not e.get("training"))
        elif k == "loss" and e.get("fn") == "cross_entropy":
            cur.setdefault("ce_input_probs", bool(e.get("in01") and (e.get("rowsum_err") or 1) < 1e-3))
        elif k == "backward" and isinstance(e.get("value"), (int, float)):
            cur.setdefault("loss", e["value"])
            cur["backwards_per_step"] = cur.get("backwards_per_step", 0) + 1
        elif k == "zero_grad":
            cur["zero_grad_calls"] = cur.get("zero_grad_calls", 0) + 1
        elif k == "opt_pre":
            cur["lr"] = (e.get("lrs") or [None])[0]
            cur["grad_norm"] = e.get("grad_norm")
        elif k == "opt_step":
            cur["update_ratio"] = e.get("update_ratio")
            cur = {}
            steps.append(cur)
    return [s for s in steps if s]


NUMERIC = {"loss": 0.05, "grad_norm": 0.1, "lr": 1e-6, "update_ratio": 0.2}


def diff_traces(reference: Trace, candidate: Trace, *, tolerances: dict[str, float] | None = None) -> TraceDiff:
    tol = dict(NUMERIC, **(tolerances or {}))
    a, b = _per_step(reference), _per_step(candidate)
    out = TraceDiff(compared_steps=min(len(a), len(b)))
    for i, (x, y) in enumerate(zip(a, b)):
        for key in sorted(set(x) | set(y)):
            va, vb = x.get(key), y.get(key)
            if key in tol:
                if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
                    continue
                if not (math.isfinite(va) and math.isfinite(vb)):
                    if math.isfinite(va) != math.isfinite(vb):
                        out.divergences.append(Divergence(i, key, va, vb, None))
                    continue
                rel = abs(va - vb) / max(abs(va), 1e-12)
                if rel > tol[key]:
                    out.divergences.append(Divergence(i, key, va, vb, rel))
            elif va != vb and va is not None and vb is not None:
                out.divergences.append(Divergence(i, key, va, vb, None))
    if len(a) != len(b):
        out.divergences.append(Divergence(min(len(a), len(b)), "number_of_steps", len(a), len(b), None))
    return out
