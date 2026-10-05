"""Subprocess entry point: run a user training script under trainmedic control.

``python -m trainmedic._bootstrap CONFIG.json``

The config controls the run seed (hard-coded ``torch.manual_seed(42)`` calls in
the script are *mixed* with the run seed so multi-seed validation works on
unmodified scripts), the step/time budget, tracing, and in-memory source
overrides (how candidate patches are executed without touching user files).
"""

from __future__ import annotations

import builtins
import importlib.abc
import importlib.machinery
import json
import os
import random
import sys
import time
import traceback


def _mix(user_seed: object, run_seed: int) -> int:
    try:
        u = int(user_seed)  # type: ignore[arg-type]
    except Exception:
        u = abs(hash(str(user_seed)))
    return hash((u, int(run_seed), 0x5EED)) & 0x7FFFFFFF


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)


def _patch_seed_functions(run_seed: int) -> None:
    import torch

    orig_torch = torch.manual_seed
    orig_random = random.seed

    def manual_seed(s):  # type: ignore[no-untyped-def]
        return orig_torch(_mix(s, run_seed))

    def rseed(a=None, version=2):  # type: ignore[no-untyped-def]
        return orig_random(_mix(a, run_seed) if a is not None else None, version)

    torch.manual_seed = manual_seed  # type: ignore[assignment]
    torch.random.manual_seed = manual_seed  # type: ignore[assignment]
    random.seed = rseed  # type: ignore[assignment]
    try:
        import numpy as np

        orig_np = np.random.seed

        def npseed(s=None):  # type: ignore[no-untyped-def]
            return orig_np(None if s is None else _mix(s, run_seed) % (2**32))

        np.random.seed = npseed  # type: ignore[assignment]
    except ImportError:
        pass


class _SourceLoader(importlib.abc.SourceLoader):
    def __init__(self, fullname: str, path: str, source: str) -> None:
        self.fullname, self.path, self.source = fullname, path, source

    def get_filename(self, fullname: str) -> str:
        return self.path

    def get_data(self, path: str) -> bytes:
        if os.path.realpath(path) == os.path.realpath(self.path):
            return self.source.encode("utf-8")
        with open(path, "rb") as f:
            return f.read()


class _OverrideFinder(importlib.abc.MetaPathFinder):
    def __init__(self, overrides: dict[str, str]) -> None:
        self.overrides = {os.path.realpath(k): v for k, v in overrides.items()}

    def find_spec(self, name, path, target=None):  # type: ignore[no-untyped-def]
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is not None and spec.origin and os.path.realpath(spec.origin) in self.overrides:
            spec.loader = _SourceLoader(name, spec.origin, self.overrides[os.path.realpath(spec.origin)])
            return spec
        return None


def main(config_path: str) -> int:
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    script = os.path.abspath(cfg["script"])
    seed = int(cfg.get("seed", 0))
    os.environ["TRAINMEDIC_SEED"] = str(seed)
    os.environ["TRAINMEDIC_BUDGET"] = str(cfg.get("budget", 1.0))
    if cfg.get("metrics_path"):
        os.environ["TRAINMEDIC_METRICS_FILE"] = cfg["metrics_path"]
    if cfg.get("cwd"):
        os.chdir(cfg["cwd"])

    import torch

    if cfg.get("threads"):
        torch.set_num_threads(int(cfg["threads"]))
    _seed_all(seed)
    if cfg.get("offset_seeds", True):
        _patch_seed_functions(seed)
    overrides = {os.path.abspath(k): v for k, v in (cfg.get("overrides") or {}).items()}
    if overrides:
        sys.meta_path.insert(0, _OverrideFinder(overrides))

    from trainmedic import reporting as tm_report
    from trainmedic.detect.tracer import StopTraining, Tracer

    source = overrides.get(script)
    if source is None:
        with open(script, encoding="utf-8") as f:
            source = f.read()

    tracer = Tracer(
        max_steps=cfg.get("max_steps"),
        max_seconds=cfg.get("max_seconds"),
        probes=bool(cfg.get("probes", True)) and bool(cfg.get("trace", True)),
    )
    sys.argv = [script] + list(cfg.get("argv", []))
    sys.path.insert(0, os.path.dirname(script))
    globs = {"__name__": "__main__", "__file__": script, "__builtins__": builtins, "__spec__": None, "__package__": None}
    status, error = "ok", None
    t0 = time.time()
    try:
        code = compile(source, script, "exec")
        with tracer:
            exec(code, globs)
    except StopTraining:
        tracer.meta["stopped_early"] = True
    except SystemExit as exc:
        if exc.code not in (None, 0):
            status, error = "error", f"SystemExit({exc.code})"
    except BaseException as exc:  # noqa: BLE001 - we must report every failure
        status = "error"
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-6:])
        tracer.meta["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    duration = time.time() - t0
    trace = tracer.trace()
    if cfg.get("trace", True) and cfg.get("trace_path"):
        trace.save(cfg["trace_path"])
    metrics = dict(tm_report.reported())
    fl = trace.final_loss()
    if fl is not None:
        metrics.setdefault("train_loss_final", fl)
    metrics.setdefault("steps", trace.n_steps)
    result = {
        "status": status,
        "error": error,
        "stopped_early": bool(trace.meta.get("stopped_early")),
        "duration": duration,
        "metrics": metrics,
        "seed": seed,
    }
    if cfg.get("result_path"):
        with open(cfg["result_path"], "w", encoding="utf-8") as f:
            json.dump(result, f, default=str)
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
