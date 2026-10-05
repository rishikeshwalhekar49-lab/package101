"""The training trace: an ordered list of JSON-serialisable events."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Trace:
    events: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ access
    def of(self, *kinds: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("kind") in kinds]

    def first(self, kind: str) -> dict[str, Any] | None:
        for e in self.events:
            if e.get("kind") == kind:
                return e
        return None

    @property
    def n_steps(self) -> int:
        return len(self.of("opt_step"))

    def losses(self) -> list[float]:
        """Per-backward loss values, falling back to the loss-function values."""
        vals = [e["value"] for e in self.of("backward") if isinstance(e.get("value"), (int, float))]
        if not vals:
            vals = [e["value"] for e in self.of("loss") if isinstance(e.get("value"), (int, float))]
        return [v for v in vals if v is not None]

    def step_losses(self) -> list[float]:
        """One (mean) loss per optimizer step."""
        out: list[float] = []
        bucket: list[float] = []
        for e in self.events:
            if e.get("kind") == "backward" and isinstance(e.get("value"), (int, float)):
                bucket.append(e["value"])
            elif e.get("kind") == "opt_step" and bucket:
                out.append(sum(bucket) / len(bucket))
                bucket = []
        return out

    def series(self, kind: str, key: str) -> list[float]:
        return [e[key] for e in self.of(kind) if isinstance(e.get(key), (int, float)) and math.isfinite(e[key])]

    def final_loss(self, frac: float = 0.1) -> float | None:
        ls = self.step_losses() or self.losses()
        if not ls:
            return None
        k = max(1, int(len(ls) * frac))
        tail = [x for x in ls[-k:] if math.isfinite(x)]
        return sum(tail) / len(tail) if tail else float("nan")

    def summary(self) -> dict[str, Any]:
        ls = self.step_losses()
        gn = self.series("opt_step", "grad_norm")
        return {
            "steps": self.n_steps,
            "forwards": len(self.of("forward")),
            "backwards": len(self.of("backward")),
            "first_loss": ls[0] if ls else None,
            "final_loss": self.final_loss(),
            "grad_norm_median": sorted(gn)[len(gn) // 2] if gn else None,
            "stopped_early": bool(self.meta.get("stopped_early")),
            "error": self.meta.get("error"),
        }

    # ------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "__meta__", **self.meta}, default=_json_default) + "\n")
            for e in self.events:
                f.write(json.dumps(e, default=_json_default) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        events: list[dict[str, Any]] = []
        meta: dict[str, Any] = {}
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                if e.get("kind") == "__meta__":
                    e.pop("kind")
                    meta = e
                else:
                    events.append(e)
        return cls(events, meta)

    @classmethod
    def load_many(cls, paths: Iterable[str | Path]) -> list["Trace"]:
        return [cls.load(p) for p in paths]


def _json_default(o: Any):
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            pass
    return str(o)
