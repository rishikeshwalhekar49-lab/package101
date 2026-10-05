"""Run training scripts in isolated subprocesses (seeded, budgeted, traced,
optionally with patched sources injected in memory)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..detect.trace import Trace
from ..repair.patch import Patch

_PKG_PARENT = str(Path(__file__).resolve().parents[2])


@dataclass
class RunSpec:
    """What to run: a script plus its CLI arguments, working directory and environment."""

    script: str
    args: list[str] = field(default_factory=list)
    cwd: str | None = None  # default: the directory trainmedic was started from (like `python train.py`)
    python: str = sys.executable
    env: dict[str, str] = field(default_factory=dict)
    timeout: float | None = 3600.0
    root: str | None = None  # patch paths are relative to this (default: script directory)

    @property
    def script_path(self) -> Path:
        return Path(self.script).resolve()

    @property
    def root_path(self) -> Path:
        return Path(self.root).resolve() if self.root else self.script_path.parent

    def rel(self, path: str | Path) -> str:
        p = Path(path).resolve()
        try:
            return str(p.relative_to(self.root_path))
        except ValueError:
            return str(p)

    def read_sources(self, files: list[str] | None = None) -> dict[str, str]:
        files = files or [self.rel(self.script_path)]
        return {f: (self.root_path / f).read_text(encoding="utf-8") for f in files}


@dataclass
class RunResult:
    ok: bool
    seed: int
    budget: float
    metrics: dict[str, Any]
    duration: float
    returncode: int
    error: str | None = None
    stopped_early: bool = False
    trace_path: str | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    tag: str = ""

    def metric(self, name: str) -> float:
        v = self.metrics.get(name)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return float("nan")
        return v

    def load_trace(self) -> Trace | None:
        if self.trace_path and Path(self.trace_path).exists():
            return Trace.load(self.trace_path)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "seed": self.seed, "budget": self.budget, "metrics": self.metrics,
            "duration": self.duration, "returncode": self.returncode, "error": self.error,
            "stopped_early": self.stopped_early, "trace_path": self.trace_path, "tag": self.tag,
        }


class Runner:
    """Launches runs; accumulates the compute (wall-clock seconds) spent."""

    def __init__(
        self,
        spec: RunSpec,
        *,
        workdir: str | Path | None = None,
        workers: int = 1,
        threads_per_run: int | None = None,
        offset_seeds: bool = True,
        verbose: bool = False,
    ) -> None:
        self.spec = spec
        self._cwd = os.getcwd()
        self._tmp = None
        if workdir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="trainmedic-")
            workdir = self._tmp.name
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.workers = max(1, workers)
        self.threads_per_run = threads_per_run or (max(1, (os.cpu_count() or 2) // self.workers) if self.workers > 1 else None)
        self.offset_seeds = offset_seeds
        self.verbose = verbose
        self.total_seconds = 0.0
        self.n_runs = 0
        self._lock = threading.Lock()

    def overrides_for(self, patch: Patch | None, sources: dict[str, str] | None = None) -> dict[str, str]:
        if patch is None or not patch.edits:
            return {}
        sources = sources or self.spec.read_sources(patch.files)
        new = patch.apply(sources)
        return {str(self.spec.root_path / f): text for f, text in new.items()}

    def run(
        self,
        seed: int = 0,
        *,
        patch: Patch | None = None,
        overrides: dict[str, str] | None = None,
        max_steps: int | None = None,
        max_seconds: float | None = None,
        trace: bool = True,
        probes: bool = True,
        budget: float = 1.0,
        tag: str = "",
    ) -> RunResult:
        ov = dict(overrides or {})
        ov.update(self.overrides_for(patch))
        rid = f"{tag or 'run'}-s{seed}-b{budget:g}-{uuid.uuid4().hex[:8]}".replace("/", "_")
        d = self.workdir / rid
        d.mkdir(parents=True, exist_ok=True)
        cfg = {
            "script": str(self.spec.script_path),
            "argv": list(self.spec.args),
            "cwd": self.spec.cwd or self._cwd,
            "seed": seed,
            "budget": budget,
            "max_steps": max_steps,
            "max_seconds": max_seconds,
            "trace": trace,
            "probes": probes,
            "offset_seeds": self.offset_seeds,
            "trace_path": str(d / "trace.jsonl") if trace else None,
            "metrics_path": str(d / "metrics.json"),
            "result_path": str(d / "result.json"),
            "overrides": ov,
            "threads": self.threads_per_run,
        }
        cfg_path = d / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        env = dict(os.environ)
        env.update(self.spec.env)
        env["PYTHONPATH"] = _PKG_PARENT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.setdefault("PYTHONHASHSEED", "0")
        if self.threads_per_run:
            env["OMP_NUM_THREADS"] = str(self.threads_per_run)
            env["MKL_NUM_THREADS"] = str(self.threads_per_run)
        t0 = time.time()
        try:
            proc = subprocess.run(
                [self.spec.python, "-m", "trainmedic._bootstrap", str(cfg_path)],
                capture_output=True, text=True, env=env, timeout=self.spec.timeout,
            )
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            rc, out, err = -9, (exc.stdout or "") if isinstance(exc.stdout, str) else "", f"timeout after {self.spec.timeout}s"
        dt = time.time() - t0
        with self._lock:
            self.total_seconds += dt
            self.n_runs += 1
        result: dict[str, Any] = {}
        rp = d / "result.json"
        if rp.exists():
            try:
                result = json.loads(rp.read_text())
            except json.JSONDecodeError:
                result = {}
        ok = rc == 0 and result.get("status") == "ok"
        error = result.get("error") or (None if ok else (err.strip().splitlines()[-1] if err.strip() else f"exit code {rc}"))
        res = RunResult(
            ok=ok, seed=seed, budget=budget, metrics=result.get("metrics", {}), duration=dt, returncode=rc, error=error,
            stopped_early=bool(result.get("stopped_early")), trace_path=cfg["trace_path"] if trace else None,
            stdout_tail=out[-2000:], stderr_tail=err[-4000:], tag=tag,
        )
        if self.verbose:
            print(f"[trainmedic] {rid}: ok={ok} {dt:.1f}s metrics={ {k: v for k, v in res.metrics.items() if isinstance(v, (int, float))} }",
                  file=sys.stderr)
        return res

    def run_many(self, jobs: list[dict[str, Any]]) -> list[RunResult]:
        if self.workers <= 1 or len(jobs) <= 1:
            return [self.run(**j) for j in jobs]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(lambda j: self.run(**j), jobs))

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def __enter__(self) -> "Runner":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def metric_value(res: RunResult, metric: str, direction: str) -> float:
    """Metric for a run; crashed / non-finite runs get the worst possible value."""
    v = res.metric(metric) if res.ok else float("nan")
    if not math.isfinite(v):
        return float("-inf") if direction == "max" else float("inf")
    return v
