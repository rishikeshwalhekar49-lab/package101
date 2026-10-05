"""The tiny API a training script uses to cooperate with trainmedic.

Everything here is optional and is a no-op outside a trainmedic run::

    import trainmedic as tm
    seed = tm.seed_everything()          # honours TRAINMEDIC_SEED (multi-seed validation)
    epochs = tm.scaled(20)               # honours TRAINMEDIC_BUDGET (successive halving)
    ...
    tm.report(val_acc=acc)               # the metric used for statistical patch validation
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Callable

_listeners: list[Callable[[dict[str, Any]], None]] = []
_metrics: dict[str, Any] = {}


def add_listener(fn: Callable[[dict[str, Any]], None]) -> None:
    if fn not in _listeners:
        _listeners.append(fn)


def remove_listener(fn: Callable[[dict[str, Any]], None]) -> None:
    if fn in _listeners:
        _listeners.remove(fn)


def report(metrics: dict[str, Any] | None = None, **kwargs: Any) -> None:
    """Record final (or latest) metrics of this run. Later calls overwrite keys."""
    m = dict(metrics or {})
    m.update(kwargs)
    clean: dict[str, Any] = {}
    for k, v in m.items():
        if hasattr(v, "item"):
            v = v.item()
        clean[k] = v
    _metrics.update(clean)
    for fn in list(_listeners):
        fn(clean)
    path = os.environ.get("TRAINMEDIC_METRICS_FILE")
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_metrics, f)
    if os.environ.get("TRAINMEDIC_PRINT_METRICS"):
        print("TRAINMEDIC_METRIC " + json.dumps(clean), flush=True)


def reported() -> dict[str, Any]:
    return dict(_metrics)


def reset() -> None:
    _metrics.clear()


def seed(default: int = 0) -> int:
    """The seed for this run (``TRAINMEDIC_SEED`` when running under trainmedic)."""
    return int(os.environ.get("TRAINMEDIC_SEED", default))


def seed_everything(default: int = 0) -> int:
    s = seed(default)
    random.seed(s)
    try:
        import numpy as np

        np.random.seed(s % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(s)
    except ImportError:
        pass
    return s


def budget(default: float = 1.0) -> float:
    """Fraction of the full training budget this run may use (``TRAINMEDIC_BUDGET``)."""
    return float(os.environ.get("TRAINMEDIC_BUDGET", default))


def scaled(n: int, minimum: int = 1) -> int:
    """Scale an iteration count (epochs/steps) by :func:`budget`."""
    return max(minimum, int(round(n * budget())))
