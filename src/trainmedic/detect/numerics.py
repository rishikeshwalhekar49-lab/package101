"""Streaming numeric anomaly detection shared by offline checks and the Guardian.

The detector follows the ZClip idea: keep an exponential moving average of the
mean and variance of a signal (loss or gradient norm) and flag values whose
z-score exceeds a threshold. Flagged values are *not* folded into the EMA, so a
spike cannot inflate the statistics that are supposed to detect it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EmaZScore:
    alpha: float = 0.05
    warmup: int = 20
    z: float = 6.0
    min_std_frac: float = 1e-3
    n: int = 0
    mean: float = 0.0
    var: float = 0.0

    def score(self, x: float) -> float:
        if self.n < self.warmup:
            return 0.0
        std = max(math.sqrt(max(self.var, 0.0)), self.min_std_frac * max(abs(self.mean), 1e-12))
        return (x - self.mean) / std if std > 0 else 0.0

    def update(self, x: float) -> tuple[float, bool]:
        """Return (z-score, is_anomaly). Non-finite values are always anomalies."""
        if not math.isfinite(x):
            return float("inf"), True
        zs = self.score(x)
        anomalous = self.n >= self.warmup and zs > self.z
        if not anomalous:
            if self.n == 0:
                self.mean, self.var = x, 0.0
            elif self.n < self.warmup:
                # Welford during warm-up for a stable start
                delta = x - self.mean
                self.mean += delta / (self.n + 1)
                self.var += (delta * (x - self.mean) - self.var) / (self.n + 1)
            else:
                delta = x - self.mean
                self.mean += self.alpha * delta
                self.var = (1 - self.alpha) * (self.var + self.alpha * delta * delta)
            self.n += 1
        return zs, anomalous

    def state(self) -> dict:
        return {"n": self.n, "mean": self.mean, "var": self.var}

    def load(self, st: dict) -> None:
        self.n, self.mean, self.var = st["n"], st["mean"], st["var"]


def spike_indices(values: list[float], z: float = 6.0, alpha: float = 0.05, warmup: int = 20) -> list[int]:
    det = EmaZScore(alpha=alpha, warmup=warmup, z=z)
    out = []
    for i, v in enumerate(values):
        _, bad = det.update(v)
        if bad and math.isfinite(v):
            out.append(i)
    return out
