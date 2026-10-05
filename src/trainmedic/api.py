"""High-level entry points for use from Python.

    import trainmedic as tm

    with tm.watch(model, optimizer, max_steps=50) as w:   # in-process tracing
        for x, y in loader:
            ...
    print(w.evidence)

    result = tm.check_script("train.py", max_steps=40)    # subprocess: lint + trace + invariants
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from .detect.invariants import InvariantSet
from .detect.lint import lint_source
from .detect.trace import Trace
from .detect.tracer import StopTraining, Tracer
from .types import Evidence, EvidenceIds, Severity


class SilentBugError(RuntimeError):
    def __init__(self, evidence: list[Evidence]) -> None:
        self.evidence = evidence
        super().__init__("trainmedic found likely silent bugs:\n" + "\n".join(e.short() for e in evidence))


class Watch:
    def __init__(self, tracer: Tracer) -> None:
        self.tracer = tracer
        self.evidence: list[Evidence] = []
        self.trace: Trace | None = None


@contextmanager
def watch(
    model: Any = None,
    optimizer: Any = None,
    *,
    invariants: InvariantSet | None = None,
    max_steps: int | None = None,
    probes: bool = True,
    report: bool = True,
    raise_on_error: bool = False,
    min_confidence: float = 0.5,
) -> Iterator[Watch]:
    """Trace the training code run inside the ``with`` block and check invariants on exit.

    Reaching ``max_steps`` ends the block quietly (the rest of the block is skipped)."""
    tracer = Tracer(max_steps=max_steps, probes=probes)
    w = Watch(tracer)
    tracer.install()
    if model is not None:
        tracer.register_module(model)
    if optimizer is not None:
        tracer.register_optimizer(optimizer)
    try:
        yield w
    except StopTraining:
        tracer.meta["stopped_early"] = True
    finally:
        tracer.uninstall()
        w.trace = tracer.trace()
        w.evidence = EvidenceIds().renumber((invariants or InvariantSet.default()).check(w.trace))
        shown = [e for e in w.evidence if e.confidence >= min_confidence]
        if report and shown:
            print("[trainmedic] possible silent bugs:", file=sys.stderr)
            for e in shown:
                print("  " + e.short(), file=sys.stderr)
        if raise_on_error and any(e.severity == Severity.ERROR for e in shown):
            raise SilentBugError(shown)


@dataclass
class CheckResult:
    script: str
    evidence: list[Evidence] = field(default_factory=list)
    trace: Trace | None = None
    ok: bool = True
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def actionable(self, min_confidence: float = 0.5) -> list[Evidence]:
        return [e for e in self.evidence if e.confidence >= min_confidence and e.severity != Severity.INFO]


def check_script(
    script: str,
    args: list[str] | tuple[str, ...] = (),
    *,
    max_steps: int | None = 40,
    invariants: InvariantSet | None = None,
    lint: bool = True,
    probes: bool = True,
    seed: int = 0,
    save_trace: str | None = None,
    cwd: str | None = None,
) -> CheckResult:
    """Lint ``script`` and run it (in a subprocess, up to ``max_steps`` optimizer steps) under the tracer."""
    from .validate.runner import Runner, RunSpec

    spec = RunSpec(script, list(args), cwd=cwd)
    evidence: list[Evidence] = []
    if lint:
        rel = spec.rel(spec.script_path)
        evidence.extend(lint_source(spec.script_path.read_text(encoding="utf-8"), rel))
    with Runner(spec) as runner:
        res = runner.run(seed, max_steps=max_steps, probes=probes, tag="check")
        trace = res.load_trace()
        if save_trace and trace is not None:
            trace.save(save_trace)
    if trace is not None:
        evidence.extend((invariants or InvariantSet.default()).check(trace))
    elif not res.ok:
        evidence.append(Evidence(id="", source="runtime_error", rule="crash", message=res.error or "run failed",
                                 severity=Severity.ERROR, confidence=1.0, cause="crash", category="crash"))
    return CheckResult(script, EvidenceIds().renumber(evidence), trace, res.ok, res.error, res.metrics)
