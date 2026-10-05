"""Patch minimisation under stochastic training (delta debugging with a noisy oracle).

``ddmin`` (Zeller & Hildebrandt) finds a 1-minimal subset of a patch's edits that
still repairs the program. For DL training the "does it still repair?" oracle is
noisy, so oracles here are three-valued (True / False / None = inconclusive) and
use majority votes over repeated evaluations; inconclusive outcomes are treated
as "does not pass", which keeps the result sound (never drops a needed edit on a
lucky seed). After a minimal fix is found, the complement is tested for an
*independent* alternative fix, and each fix is classified as code-only,
hyperparameter-only or mixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..detect.invariants import InvariantSet
from ..validate.stats import mean
from .patch import Patch

Test = Callable[[frozenset[int]], Optional[bool]]


def ddmin(n: int, test: Test, *, max_tests: int = 200) -> tuple[list[int], int]:
    """Return (minimal subset of range(n) for which ``test`` is True, number of tests run)."""
    cache: dict[frozenset[int], Optional[bool]] = {}

    def t(items: Sequence[int]) -> bool:
        key = frozenset(items)
        if key not in cache:
            if len(cache) >= max_tests:
                return False
            cache[key] = test(key)
        return cache[key] is True

    items = list(range(n))
    if n == 0 or not t(items):
        return items, len(cache)
    gran = 2
    while len(items) >= 2:
        size = math.ceil(len(items) / gran)
        chunks = [items[i : i + size] for i in range(0, len(items), size)]
        reduced = False
        for c in chunks:
            if t(c):
                items, gran, reduced = c, 2, True
                break
        if not reduced:
            for c in chunks:
                comp = [x for x in items if x not in c]
                if comp and t(comp):
                    items, gran, reduced = comp, max(gran - 1, 2), True
                    break
        if not reduced:
            if gran >= len(items):
                break
            gran = min(len(items), gran * 2)
    if len(items) == 1 or not items:
        return items, len(cache)
    return items, len(cache)


def fix_kind(p: Patch) -> str:
    kinds = p.kinds
    if kinds == {"hyperparameter"}:
        return "hyperparameter"
    if "hyperparameter" in kinds:
        return "mixed"
    return "code"


@dataclass
class MinimizationResult:
    original: Patch
    minimal: Patch
    alternatives: list[Patch] = field(default_factory=list)
    tests: int = 0
    kind: str = "code"

    @property
    def reduction(self) -> float:
        return 1.0 - self.minimal.size / self.original.size if self.original.size else 0.0

    def to_dict(self) -> dict:
        return {
            "original_size": self.original.size,
            "minimal_size": self.minimal.size,
            "reduction": self.reduction,
            "tests": self.tests,
            "kind": self.kind,
            "minimal": self.minimal.to_dict(),
            "alternatives": [a.to_dict() for a in self.alternatives],
        }


def minimize_patch(patch: Patch, oracle: Callable[[Patch], Optional[bool]], *, find_alternatives: bool = True,
                   max_tests: int = 60) -> MinimizationResult:
    def test(idx: frozenset[int]) -> Optional[bool]:
        return oracle(patch.subset(idx)) if idx else False

    keep, n_tests = ddmin(len(patch.edits), test, max_tests=max_tests)
    minimal = patch.subset(keep)
    minimal.meta = dict(patch.meta, minimized=True)
    res = MinimizationResult(patch, minimal, tests=n_tests, kind=fix_kind(minimal))
    if find_alternatives:
        rest = [i for i in range(len(patch.edits)) if i not in keep]
        if rest and oracle(patch.subset(rest)) is True:
            sub = patch.subset(rest)
            alt_keep, t2 = ddmin(len(sub.edits), lambda idx: oracle(sub.subset(idx)) if idx else False, max_tests=max_tests)
            res.alternatives.append(sub.subset(alt_keep))
            res.tests += t2 + 1
    return res


# ------------------------------------------------------------------ oracles
class InvariantOracle:
    """Passes if the patched program runs, the targeted invariant violations are
    gone and no new violations appear (cheap: one short traced run)."""

    def __init__(self, runner, invariants: InvariantSet, target_rules: set[str], baseline_rules: set[str], *,
                 max_steps: int = 40, seed: int = 0, min_confidence: float = 0.5) -> None:
        self.runner = runner
        self.invariants = invariants
        self.target = set(target_rules)
        self.baseline = set(baseline_rules)
        self.max_steps = max_steps
        self.seed = seed
        self.min_confidence = min_confidence
        self.calls = 0

    def violations(self, patch: Patch | None) -> tuple[bool, set[str]]:
        self.calls += 1
        res = self.runner.run(self.seed, patch=patch, max_steps=self.max_steps, tag="inv")
        tr = res.load_trace()
        if tr is None:
            return False, {"crash"}
        rules = {e.rule for e in self.invariants.check(tr) if e.confidence >= self.min_confidence}
        return res.ok and "crash" not in rules, rules

    def __call__(self, patch: Patch) -> Optional[bool]:
        ok, rules = self.violations(patch)
        if not ok:
            return False
        if rules & self.target:
            return False
        if rules - self.baseline - self.target:
            return False
        return True


class StatisticalOracle:
    """Passes if a subset keeps most of the full patch's improvement over the
    baseline across a few seeds (majority vote; None when inconclusive)."""

    def __init__(self, runner, metric: str, direction: str, *, seeds: Sequence[int] = (0, 1, 2), budget: float = 0.5,
                 keep_fraction: float = 0.5, full: Patch | None = None) -> None:
        self.runner = runner
        self.metric = metric
        self.sign = 1.0 if direction == "max" else -1.0
        self.seeds = list(seeds)
        self.budget = budget
        self.keep_fraction = keep_fraction
        self._base = self._values(None)
        self._full = self._values(full) if full is not None else None
        self.calls = 0

    def _values(self, patch: Patch | None) -> list[float]:
        jobs = [{"seed": s, "patch": patch, "budget": self.budget, "trace": False, "tag": "stat"} for s in self.seeds]
        out = []
        for r in self.runner.run_many(jobs):
            v = r.metric(self.metric) if r.ok else float("nan")
            out.append(v)
        return out

    def __call__(self, patch: Patch) -> Optional[bool]:
        self.calls += 1
        vals = self._values(patch)
        if any(not math.isfinite(v) for v in vals):
            return False
        gain = [self.sign * (v - b) for v, b in zip(vals, self._base) if math.isfinite(b)]
        if not gain:
            return None
        if self._full is None:
            return True if mean(gain) > 0 and sum(g > 0 for g in gain) > len(gain) / 2 else False
        full_gain = [self.sign * (f - b) for f, b in zip(self._full, self._base) if math.isfinite(f) and math.isfinite(b)]
        target = mean(full_gain) * self.keep_fraction if full_gain else 0.0
        votes = sum(g >= target for g in gain)
        if votes > len(gain) / 2:
            return True
        if votes < len(gain) / 2:
            return False
        return None


class CombinedOracle:
    def __init__(self, *oracles: Callable[[Patch], Optional[bool]]) -> None:
        self.oracles = oracles

    def __call__(self, patch: Patch) -> Optional[bool]:
        verdict: Optional[bool] = True
        for o in self.oracles:
            v = o(patch)
            if v is False:
                return False
            if v is None:
                verdict = None
        return verdict
