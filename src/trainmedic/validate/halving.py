"""Compute-budgeted statistical acceptance of candidate patches.

Problem: training is stochastic, so "the patch fixed it" needs several seeds,
and with many candidate patches a fixed k-seed full-budget evaluation of each is
expensive. :class:`SequentialValidator` combines

1. **successive halving** over candidates: cheap rungs (few seeds, reduced
   training budget) rank candidates by their paired improvement over the
   baseline; only the best 1/eta survive each rung, and candidates whose upper
   confidence bound is below the minimum effect are dropped early (futility);
2. **held-out confirmation**: the finalists are re-run on *fresh* seeds that
   played no role in selection, at full budget, and accepted only if a one-sided
   paired sign-flip test rejects "no improvement" at alpha / #finalists.

Because selection and confirmation use disjoint seeds and the final test is
Bonferroni-corrected, P(accepting a candidate that does not improve the metric
by more than ``min_effect``) <= alpha, whatever happened during selection
(under the sign-flip test's symmetry assumption). Early elimination can only
reduce false accepts. Compute saved relative to fixed-k evaluation is reported.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

from .stats import Comparison, compare_paired, mean, seeds_needed, signflip_pvalue, stdev

Evaluator = Callable[[str, int, float], Optional[float]]
BatchEvaluator = Callable[[list[tuple[str, int, float]]], list[Optional[float]]]


@dataclass
class Rung:
    seeds: int
    budget: float


@dataclass
class AcceptanceConfig:
    metric: str = "metric"
    direction: str = "max"
    alpha: float = 0.05
    min_effect: float = 0.0
    rungs: list[Rung] = field(default_factory=lambda: [Rung(2, 0.34), Rung(3, 0.67)])
    eta: float = 3.0
    final_candidates: int = 1
    confirm_seeds: int | None = None
    confirm_budget: float = 1.0
    confirm_seed_offset: int = 1000
    futility_z: float = 2.0

    def n_confirm(self) -> int:
        return self.confirm_seeds or max(5, seeds_needed(self.alpha, self.final_candidates) + 1)


@dataclass
class CandidateRecord:
    key: str
    status: str = "active"
    reason: str = ""
    rungs: list[dict] = field(default_factory=list)
    confirmation: dict | None = None


@dataclass
class ValidationReport:
    baseline: str
    accepted: list[str]
    records: dict[str, CandidateRecord]
    seconds_used: float
    naive_seconds: float
    evaluations: int
    config: AcceptanceConfig
    baseline_failed: bool = False

    @property
    def savings(self) -> float:
        return 1.0 - self.seconds_used / self.naive_seconds if self.naive_seconds > 0 else 0.0

    @property
    def best(self) -> str | None:
        return self.accepted[0] if self.accepted else None

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline,
            "accepted": self.accepted,
            "records": {k: asdict(v) for k, v in self.records.items()},
            "seconds_used": self.seconds_used,
            "naive_seconds": self.naive_seconds,
            "savings": self.savings,
            "evaluations": self.evaluations,
            "config": asdict(self.config),
        }

    def to_markdown(self) -> str:
        c = self.config
        lines = [
            f"**Statistical validation** (metric `{c.metric}`, {c.direction}imise, alpha={c.alpha}, min effect={c.min_effect})",
            "",
            "| candidate | status | last rung mean improvement | confirmation p | reason |",
            "|---|---|---|---|---|",
        ]
        for k, r in self.records.items():
            last = r.rungs[-1]["mean_diff"] if r.rungs else float("nan")
            p = r.confirmation["p_improve"] if r.confirmation else None
            lines.append(f"| {k} | {r.status} | {last:+.4g} | {'' if p is None else f'{p:.4f}'} | {r.reason} |")
        lines.append("")
        lines.append(f"Compute: {self.seconds_used:.1f}s used vs ~{self.naive_seconds:.1f}s for fixed-k full-budget "
                     f"evaluation ({self.savings:.0%} saved), {self.evaluations} runs.")
        return "\n".join(lines)


class SequentialValidator:
    def __init__(
        self,
        evaluate: Evaluator | None = None,
        config: AcceptanceConfig | None = None,
        *,
        evaluate_batch: BatchEvaluator | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if evaluate is None and evaluate_batch is None:
            raise ValueError("need evaluate or evaluate_batch")
        self.evaluate = evaluate
        self.evaluate_batch = evaluate_batch
        self.config = config or AcceptanceConfig()
        self.cache: dict[tuple[str, int, float], Optional[float]] = {}
        self.seconds = 0.0
        self.unit_costs: list[float] = []
        self.log = log or (lambda s: None)

    # ------------------------------------------------------------ evaluation
    def _ensure(self, jobs: list[tuple[str, int, float]]) -> None:
        todo = [j for j in dict.fromkeys(jobs) if j not in self.cache]
        if not todo:
            return
        t0 = time.time()
        if self.evaluate_batch is not None:
            vals = self.evaluate_batch(todo)
        else:
            vals = [self.evaluate(*j) for j in todo]  # type: ignore[misc]
        dt = time.time() - t0
        self.seconds += dt
        total_budget = sum(j[2] for j in todo) or 1.0
        self.unit_costs.append(dt / total_budget)
        for j, v in zip(todo, vals):
            self.cache[j] = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)

    def _diffs(self, key: str, baseline: str, seeds: Sequence[int], budget: float) -> tuple[list[float], bool]:
        sign = 1.0 if self.config.direction == "max" else -1.0
        out: list[float] = []
        crashed = False
        for s in seeds:
            c = self.cache.get((key, s, budget))
            b = self.cache.get((baseline, s, budget))
            if c is None or not math.isfinite(c):
                crashed = True
                continue
            if b is None or not math.isfinite(b):
                out.append(math.inf)  # baseline crashed: any running candidate is an improvement
                continue
            out.append(sign * (c - b))
        return out, crashed

    # ------------------------------------------------------------------ main
    def run(self, candidates: Sequence[str], baseline: str = "baseline") -> ValidationReport:
        cfg = self.config
        recs = {k: CandidateRecord(k) for k in candidates}
        active = list(candidates)
        for ri, rung in enumerate(cfg.rungs):
            if not active:
                break
            seeds = list(range(rung.seeds))
            self._ensure([(k, s, rung.budget) for k in [baseline, *active] for s in seeds])
            scores: dict[str, float] = {}
            for k in list(active):
                d, crashed = self._diffs(k, baseline, seeds, rung.budget)
                finite = [x for x in d if math.isfinite(x)]
                m = mean(d) if d else float("nan")
                sd = stdev(finite) if len(finite) > 1 else 0.0
                recs[k].rungs.append({"rung": ri, "seeds": rung.seeds, "budget": rung.budget, "diffs": d, "mean_diff": m, "sd": sd})
                if crashed:
                    recs[k].status, recs[k].reason = "crashed", f"crashed at rung {ri}"
                    active.remove(k)
                    continue
                if len(finite) >= 2 and m + cfg.futility_z * sd / math.sqrt(len(finite)) < cfg.min_effect and m <= 0:
                    recs[k].status, recs[k].reason = "eliminated", f"futile at rung {ri} (mean {m:+.4g})"
                    active.remove(k)
                    continue
                scores[k] = m
            keep = max(cfg.final_candidates, math.ceil(len(active) / cfg.eta)) if ri < len(cfg.rungs) - 1 else len(active)
            ranked = sorted(active, key=lambda k: scores.get(k, -math.inf), reverse=True)
            for k in ranked[keep:]:
                recs[k].status, recs[k].reason = "eliminated", f"not in top {keep} at rung {ri}"
            active = ranked[:keep]
            self.log(f"rung {ri}: kept {active}")
        finalists = sorted(active, key=lambda k: recs[k].rungs[-1]["mean_diff"] if recs[k].rungs else 0, reverse=True)
        finalists = finalists[: cfg.final_candidates]
        for k in active:
            if k not in finalists:
                recs[k].status, recs[k].reason = "eliminated", "not a finalist"
        accepted: list[str] = []
        if finalists:
            n = cfg.n_confirm()
            seeds = list(range(cfg.confirm_seed_offset, cfg.confirm_seed_offset + n))
            self._ensure([(k, s, cfg.confirm_budget) for k in [baseline, *finalists] for s in seeds])
            level = cfg.alpha / len(finalists)
            for k in finalists:
                d, crashed = self._diffs(k, baseline, seeds, cfg.confirm_budget)
                if crashed:
                    recs[k].status, recs[k].reason = "crashed", "crashed during confirmation"
                    continue
                if any(math.isinf(x) for x in d):
                    p = 0.5 ** len(d) if all(x > cfg.min_effect for x in d) else 1.0
                else:
                    p = signflip_pvalue(d, "greater", cfg.min_effect)
                finite = [x for x in d if math.isfinite(x)]
                recs[k].confirmation = {"seeds": seeds, "diffs": d, "mean_diff": mean(d), "sd": stdev(finite), "p_improve": p, "level": level}
                if p <= level:
                    recs[k].status, recs[k].reason = "accepted", f"p={p:.4f} <= {level:.4f}"
                    accepted.append(k)
                else:
                    recs[k].status, recs[k].reason = "rejected", f"p={p:.4f} > {level:.4f}"
        unit = statistics_median(self.unit_costs)
        naive = (len(candidates) + 1) * cfg.n_confirm() * cfg.confirm_budget * unit
        base_vals = [v for (k, _s, _b), v in self.cache.items() if k == baseline]
        failed = bool(base_vals) and all(v is None for v in base_vals)
        return ValidationReport(baseline, accepted, recs, self.seconds, naive, len(self.cache), cfg, failed)


def statistics_median(xs: list[float]) -> float:
    xs = sorted(x for x in xs if math.isfinite(x))
    if not xs:
        return 0.0
    return xs[len(xs) // 2]


def fixed_k_validate(
    evaluate: Evaluator,
    candidates: Sequence[str],
    baseline: str = "baseline",
    *,
    k: int = 5,
    budget: float = 1.0,
    alpha: float = 0.05,
    direction: str = "max",
    min_effect: float = 0.0,
) -> dict:
    """The naive reference procedure: k seeds at full budget for everyone, Bonferroni."""
    t0 = time.time()
    vals = {(c, s): evaluate(c, s, budget) for c in [baseline, *candidates] for s in range(k)}
    out = {}
    for c in candidates:
        cand = [vals[(c, s)] if vals[(c, s)] is not None else math.nan for s in range(k)]
        base = [vals[(baseline, s)] if vals[(baseline, s)] is not None else math.nan for s in range(k)]
        cmp = compare_paired(cand, base, direction=direction, margin=min_effect)
        out[c] = {"accepted": cmp.p_improve <= alpha / max(1, len(candidates)), **cmp.to_dict()}
    return {"results": out, "seconds": time.time() - t0, "evaluations": len(vals)}


def damage_test(
    patched: Sequence[float], original: Sequence[float], *, direction: str = "max", margin: float = 0.0, alpha: float = 0.05
) -> tuple[bool, Comparison]:
    """Did a patch applied to *correct* code make the metric significantly worse?"""
    cmp = compare_paired(patched, original, direction=direction, margin=margin)
    return (cmp.p_worse <= alpha and cmp.mean_diff < -margin), cmp
