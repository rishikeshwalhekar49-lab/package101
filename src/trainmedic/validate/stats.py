"""Small-sample statistics for comparing training runs across seeds.

Everything is dependency-free (SciPy is used only if installed, for Welch's t).
The default test for paired seed comparisons is the exact sign-flip
(randomisation) test, which is valid for any symmetric noise distribution and
needs no normality assumption -- important with 3-10 seeds.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from dataclasses import dataclass
from typing import Sequence


def mean(xs: Sequence[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def stdev(xs: Sequence[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def signflip_pvalue(diffs: Sequence[float], alternative: str = "greater", margin: float = 0.0, mc: int = 20000, seed: int = 0) -> float:
    """p-value for H0: E[d] <= margin (``greater``), E[d] >= margin (``less``) or
    E[d] == margin (``two-sided``), using the paired sign-flip test on d - margin.

    Exact for n <= 16 (2^n sign patterns), Monte-Carlo above."""
    d = [x - margin for x in diffs if math.isfinite(x)]
    n = len(d)
    if n == 0:
        return 1.0
    obs = sum(d)
    if all(x == 0 for x in d):
        return 1.0

    def extreme(s: float) -> bool:
        if alternative == "greater":
            return s >= obs - 1e-12
        if alternative == "less":
            return s <= obs + 1e-12
        return abs(s) >= abs(obs) - 1e-12

    if n <= 16:
        count = total = 0
        for signs in itertools.product((1, -1), repeat=n):
            total += 1
            if extreme(sum(s * x for s, x in zip(signs, d))):
                count += 1
        return count / total
    rng = random.Random(seed)
    count = 1
    for _ in range(mc):
        if extreme(sum(x if rng.random() < 0.5 else -x for x in d)):
            count += 1
    return count / (mc + 1)


def min_signflip_pvalue(n: int) -> float:
    """Smallest attainable one-sided p-value with n paired seeds."""
    return 0.5**n


def seeds_needed(alpha: float, comparisons: int = 1) -> int:
    """Minimum seeds so that a one-sided sign-flip test can reach alpha / comparisons."""
    target = alpha / max(1, comparisons)
    return max(1, math.ceil(math.log(1 / target, 2)))


def welch_t_pvalue(a: Sequence[float], b: Sequence[float], alternative: str = "greater") -> float:
    """One- or two-sided Welch t-test p-value for mean(a) - mean(b)."""
    a = [x for x in a if math.isfinite(x)]
    b = [x for x in b if math.isfinite(x)]
    if len(a) < 2 or len(b) < 2:
        return 1.0
    ma, mb, va, vb = mean(a), mean(b), statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return 0.0 if (ma > mb and alternative != "less") or (ma < mb and alternative == "less") else 1.0
    t = (ma - mb) / se
    df = (va / len(a) + vb / len(b)) ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    try:
        from scipy import stats as st  # type: ignore

        if alternative == "greater":
            return float(st.t.sf(t, df))
        if alternative == "less":
            return float(st.t.cdf(t, df))
        return float(2 * st.t.sf(abs(t), df))
    except ImportError:
        # normal approximation (slightly anti-conservative for small df)
        z = t
        sf = 0.5 * math.erfc(z / math.sqrt(2))
        if alternative == "greater":
            return sf
        if alternative == "less":
            return 1 - sf
        return 2 * min(sf, 1 - sf)


def bootstrap_ci(xs: Sequence[float], conf: float = 0.95, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    xs = [x for x in xs if math.isfinite(x)]
    if not xs:
        return (float("nan"), float("nan"))
    if len(xs) == 1:
        return (xs[0], xs[0])
    rng = random.Random(seed)
    ms = sorted(mean([rng.choice(xs) for _ in xs]) for _ in range(n))
    lo = ms[int((1 - conf) / 2 * n)]
    hi = ms[min(n - 1, int((1 + conf) / 2 * n))]
    return lo, hi


def holm(pvalues: Sequence[float], alpha: float) -> list[bool]:
    """Holm-Bonferroni step-down: which hypotheses are rejected."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    reject = [False] * len(pvalues)
    m = len(pvalues)
    for rank, i in enumerate(order):
        if pvalues[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


@dataclass
class Comparison:
    """Paired comparison of a candidate against a baseline across seeds."""

    n: int
    mean_diff: float
    sd_diff: float
    p_improve: float
    p_worse: float
    ci: tuple[float, float]

    @property
    def effect_size(self) -> float:
        return self.mean_diff / self.sd_diff if self.sd_diff > 0 else (math.inf if self.mean_diff > 0 else 0.0)

    def to_dict(self) -> dict:
        return {"n": self.n, "mean_diff": self.mean_diff, "sd_diff": self.sd_diff, "p_improve": self.p_improve,
                "p_worse": self.p_worse, "ci": list(self.ci)}


def compare_paired(candidate: Sequence[float], baseline: Sequence[float], *, direction: str = "max", margin: float = 0.0) -> Comparison:
    """Improvement d = sign * (candidate - baseline); ``margin`` = minimum improvement of interest."""
    sign = 1.0 if direction == "max" else -1.0
    diffs = [sign * (c - b) for c, b in zip(candidate, baseline) if math.isfinite(c) and math.isfinite(b)]
    return Comparison(
        n=len(diffs),
        mean_diff=mean(diffs) if diffs else float("nan"),
        sd_diff=stdev(diffs),
        p_improve=signflip_pvalue(diffs, "greater", margin),
        p_worse=signflip_pvalue(diffs, "less", -margin),
        ci=bootstrap_ci(diffs),
    )
