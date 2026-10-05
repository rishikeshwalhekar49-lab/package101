"""The closed loop: detect -> diagnose -> propose -> prefilter -> validate -> minimise.

The violated invariant is used three times: as diagnosis context, as a cheap
pre-filter (a candidate must make the violated invariants hold without
introducing new violations), and inside the minimisation oracle. Statistical
multi-seed acceptance (successive halving + held-out confirmation) runs only on
candidates that survive the pre-filter.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..detect.invariants import InvariantSet
from ..detect.lint import all_rules, fix_finding, lint_source
from ..detect.trace import Trace
from ..diagnose import diagnose as rules_diagnose
from ..diagnose import taxonomy
from ..llm.backend import LLMBackend, LLMError, get_backend
from ..sourceedit import parses
from ..types import Diagnosis, Evidence, EvidenceIds, Severity
from ..validate.halving import AcceptanceConfig, Rung, SequentialValidator, ValidationReport
from ..validate.runner import Runner, RunSpec
from .abstention import AbstentionPolicy
from .minimize import CombinedOracle, InvariantOracle, MinimizationResult, StatisticalOracle, minimize_patch
from .patch import Patch, PatchConflict

MODES = ("rules", "llm", "hybrid", "naive-llm")


@dataclass
class RepairConfig:
    mode: str = "hybrid"
    llm: str | None = None
    metric: str | None = None
    direction: str = "max"
    max_steps: int = 40
    validate: bool = True
    minimize: bool = True
    alpha: float = 0.05
    min_effect: float = 0.0
    rungs: list[Rung] | None = None
    confirm_seeds: int | None = None
    confirm_budget: float = 1.0
    workers: int = 2
    max_candidates: int = 6
    llm_candidates: int = 3
    allow_ungrounded: bool = False
    abstention: AbstentionPolicy = field(default_factory=AbstentionPolicy)
    invariants_path: str | None = None
    reference_traces: list[str] = field(default_factory=list)
    use_memory: bool = False
    memory_path: str | None = None
    probes: bool = True
    files: list[str] = field(default_factory=list)
    verbose: bool = False

    def acceptance(self) -> AcceptanceConfig:
        cfg = AcceptanceConfig(metric=self.metric or "metric", direction=self.direction, alpha=self.alpha,
                               min_effect=self.min_effect, confirm_seeds=self.confirm_seeds, confirm_budget=self.confirm_budget)
        if self.rungs is not None:
            cfg.rungs = self.rungs
        return cfg


@dataclass
class CandidateOutcome:
    key: str
    patch: Patch
    status: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "status": self.status, "reason": self.reason, "origin": self.patch.origin,
                "description": self.patch.description, "size": self.patch.size, "evidence_ids": self.patch.evidence_ids,
                "patch": self.patch.to_dict()}


@dataclass
class RepairReport:
    status: str
    script: str
    sources: dict[str, str]
    evidence: list[Evidence] = field(default_factory=list)
    diagnosis: Diagnosis | None = None
    abstained: bool = False
    abstain_reason: str = ""
    candidates: list[CandidateOutcome] = field(default_factory=list)
    accepted: Patch | None = None
    minimized: MinimizationResult | None = None
    validation: ValidationReport | None = None
    seconds: float = 0.0
    runs: int = 0
    notes: list[str] = field(default_factory=list)
    trace_summary: dict[str, Any] | None = None

    @property
    def final_patch(self) -> Patch | None:
        if self.minimized is not None:
            return self.minimized.minimal
        return self.accepted

    def diff(self) -> str:
        p = self.final_patch
        return p.to_unified_diff(self.sources) if p is not None else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "script": self.script,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "evidence": [e.to_dict() for e in self.evidence],
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "final_patch": self.final_patch.to_dict() if self.final_patch else None,
            "diff": self.diff(),
            "minimization": self.minimized.to_dict() if self.minimized else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "seconds": self.seconds,
            "runs": self.runs,
            "notes": self.notes,
            "trace_summary": self.trace_summary,
        }

    def to_markdown(self) -> str:
        out = [f"# trainmedic report: `{self.script}`", "", f"**Status:** {self.status}"]
        if self.abstained:
            out.append(f"**Abstained:** {self.abstain_reason}")
        out += ["", f"Compute: {self.seconds:.1f}s across {self.runs} run(s).", ""]
        if self.evidence:
            out += ["## Evidence", "", "| id | source | rule | conf | where | message |", "|---|---|---|---|---|---|"]
            for e in self.evidence:
                out.append(f"| {e.id} | {e.source} | {e.rule} | {e.confidence:.2f} | {e.location()} | {e.message.replace('|', '/')} |")
            out.append("")
        if self.diagnosis:
            out += ["## Diagnosis", "", self.diagnosis.summary, "", f"Attribution: **{self.diagnosis.attribution}**", ""]
            for c in self.diagnosis.causes[:6]:
                where = f" ({c.file}:{c.line})" if c.file and c.line else ""
                out.append(f"- **{c.title}**{where} -- confidence {c.confidence:.2f}; evidence {', '.join(c.evidence_ids)}. Fix: {c.suggested_fix}")
            out.append("")
        if self.candidates:
            out += ["## Candidate patches", "", "| key | origin | size | status | reason |", "|---|---|---|---|---|"]
            for c in self.candidates:
                out.append(f"| {c.key} | {c.patch.origin} | {c.patch.size} | {c.status} | {c.reason} |")
            out.append("")
        if self.validation:
            out += ["## Validation", "", self.validation.to_markdown(), ""]
        if self.minimized:
            m = self.minimized
            out += ["## Minimisation", "", f"{m.original.size} -> {m.minimal.size} changed lines ({m.reduction:.0%} smaller), "
                    f"{m.tests} oracle calls, fix kind: **{m.kind}**, {len(m.alternatives)} alternative fix(es).", ""]
        d = self.diff()
        if d:
            out += ["## Patch", "", "```diff", d.rstrip(), "```", ""]
        if self.notes:
            out += ["## Notes", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(out)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        if p.suffix == ".md":
            p.write_text(self.to_markdown())
        else:
            p.write_text(json.dumps(self.to_dict(), indent=2, default=str))


class RepairSession:
    def __init__(
        self,
        script: str,
        args: list[str] | tuple[str, ...] = (),
        config: RepairConfig | None = None,
        *,
        backend: LLMBackend | None = None,
        runner: Runner | None = None,
        cwd: str | None = None,
    ) -> None:
        self.config = config or RepairConfig()
        if self.config.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.spec = RunSpec(script, list(args), cwd=cwd)
        self.runner = runner or Runner(self.spec, workers=self.config.workers, verbose=self.config.verbose)
        self.files = [self.spec.rel(self.spec.script_path)] + [f for f in self.config.files if f]
        self.sources = self.spec.read_sources(self.files)
        self.backend = backend
        if self.backend is None and self.config.mode in ("llm", "hybrid", "naive-llm"):
            self.backend = get_backend(self.config.llm)
        if self.backend is None and self.config.mode in ("llm", "naive-llm"):
            raise LLMError(f"mode {self.config.mode!r} needs an LLM backend (--llm anthropic or --llm cmd:...)")
        if self.config.invariants_path:
            self.invariants = InvariantSet.load(self.config.invariants_path)
        elif self.config.reference_traces:
            self.invariants = InvariantSet.infer(Trace.load_many(self.config.reference_traces))
        else:
            self.invariants = InvariantSet.default()
        self.memory = None
        if self.config.use_memory:
            from ..memory import ExperienceMemory

            self.memory = ExperienceMemory(self.config.memory_path)
        self.notes: list[str] = []

    def log(self, msg: str) -> None:
        if self.config.verbose:
            import sys

            print(f"[trainmedic] {msg}", file=sys.stderr)

    # --------------------------------------------------------------- detect
    def lint(self, sources: dict[str, str] | None = None) -> list[Evidence]:
        out: list[Evidence] = []
        for f, text in (sources or self.sources).items():
            out.extend(lint_source(text, f))
        return out

    def detect(self) -> tuple[list[Evidence], Trace | None]:
        static = self.lint()
        res = self.runner.run(0, max_steps=self.config.max_steps, probes=self.config.probes, tag="detect")
        trace = res.load_trace()
        runtime = self.invariants.check(trace) if trace is not None else []
        if trace is None and not res.ok:
            runtime.append(Evidence(id="", source="runtime_error", rule="crash", message=res.error or "run failed",
                                    severity=Severity.ERROR, confidence=1.0, category="crash", cause="crash"))
        evidence = EvidenceIds().renumber(static + runtime)
        return evidence, trace

    # ------------------------------------------------------------- diagnose
    def diagnose(self, evidence: list[Evidence], trace: Trace | None) -> Diagnosis:
        d = rules_diagnose(evidence, threshold=self.config.abstention.min_cause_confidence)
        if self.backend is not None and self.config.mode in ("llm", "hybrid"):
            from ..llm.tasks import llm_diagnose, merge_diagnoses

            try:
                ld = llm_diagnose(self.backend, self.sources, evidence, trace_summary=trace.summary() if trace else None,
                                  rules_diagnosis=d)
                d = merge_diagnoses(d, ld, evidence, allow_ungrounded=self.config.allow_ungrounded,
                                    threshold=self.config.abstention.min_cause_confidence)
            except (LLMError, ValueError) as exc:
                self.notes.append(f"LLM diagnosis unavailable: {exc}")
        return d

    # --------------------------------------------------------------- propose
    def propose(self, evidence: list[Evidence], diagnosis: Diagnosis) -> list[Patch]:
        cfg = self.config
        policy = cfg.abstention
        patches: list[Patch] = []
        rules = all_rules()
        if cfg.mode in ("rules", "hybrid"):
            actionable = {e.id for e in policy.actionable(evidence)}
            for ev in evidence:
                if ev.source != "static" or not ev.fixable or ev.id not in actionable or ev.file not in self.sources:
                    continue
                new = fix_finding(self.sources[ev.file], ev)
                if new is None:
                    continue
                support = [x.id for x in evidence if x.id != ev.id and x.cause and ev.cause and taxonomy.related(x.cause, ev.cause)]
                patches.append(Patch.from_sources(ev.file, self.sources[ev.file], new, origin=f"rule:{ev.rule}",
                                                  description=rules[ev.rule].description, evidence_ids=[ev.id, *support]))
            if len(patches) >= 2:
                combined = patches[0]
                for p in patches[1:]:
                    try:
                        combined = combined.merge(p)
                    except PatchConflict:
                        continue
                if len(combined.edits) > len(patches[0].edits):
                    combined.origin = "rules:combined"
                    patches.insert(0, combined)
        if cfg.mode in ("rules", "hybrid"):
            patches.extend(self._lr_candidates(evidence))
        if self.memory is not None:
            patches.extend(self.memory.suggest(evidence, self.sources))
        if self.backend is not None and cfg.mode in ("llm", "hybrid"):
            from ..llm.tasks import llm_repair

            hints = self.memory.hints(evidence) if self.memory is not None else None
            try:
                abstained, reason, lp, errs = llm_repair(self.backend, self.sources, evidence, diagnosis,
                                                         max_candidates=cfg.llm_candidates, hints=hints)
                if abstained:
                    self.notes.append(f"LLM abstained: {reason}")
                self.notes.extend(f"LLM patch could not be applied: {e}" for e in errs)
                patches.extend(lp)
            except (LLMError, ValueError) as exc:
                self.notes.append(f"LLM repair unavailable: {exc}")
        if cfg.mode == "naive-llm" and self.backend is not None:
            from ..llm.tasks import naive_llm_repair

            lp, errs = naive_llm_repair(self.backend, self.sources)
            patches.extend(lp)
            self.notes.extend(errs)
        uniq: list[Patch] = []
        seen: set[str] = set()
        for p in patches:
            try:
                key = p.to_unified_diff(self.sources)
            except PatchConflict:
                continue
            if key and key not in seen:
                seen.add(key)
                uniq.append(p)
        return uniq[: cfg.max_candidates]

    def _lr_candidates(self, evidence: list[Evidence]) -> list[Patch]:
        """Hyperparameter-only candidates (LR / 10, LR / 100) when the evidence says the LR is far too high.
        Like every candidate they must pass the statistical gate, so a wrong guess is rejected, not applied."""
        import ast as _ast

        from ..detect.lint.analysis import Analysis, const_value, kwarg

        act = [e for e in self.config.abstention.actionable(evidence) if e.cause == "lr_scale"]
        if not act:
            return []
        out: list[Patch] = []
        for f, text in self.sources.items():
            try:
                an = Analysis(_ast.parse(text), text)
            except SyntaxError:
                continue
            for name in sorted(an.optimizers):
                for _stmt, value in an.assignments.get(name, []):
                    kw = kwarg(value, "lr") if isinstance(value, _ast.Call) else None
                    lr = const_value(kw.value) if kw is not None else None
                    if not isinstance(lr, (int, float)) or lr <= 0:
                        continue
                    seg = _ast.get_source_segment(text, kw.value)
                    line = text.splitlines(keepends=True)[kw.value.lineno - 1]
                    for factor in (0.1, 0.01):
                        new_line = line.replace(seg, repr(float(f"{lr * factor:.3g}")), 1)
                        lines = text.splitlines(keepends=True)
                        lines[kw.value.lineno - 1] = new_line
                        out.append(Patch.from_sources(f, text, "".join(lines), origin=f"rule:lr_scale_x{factor:g}",
                                                      description=f"scale the learning rate of `{name}` by {factor:g}",
                                                      evidence_ids=[e.id for e in act]))
        return out

    # ------------------------------------------------------------- prefilter
    def prefilter(self, patches: list[Patch], evidence: list[Evidence]) -> list[CandidateOutcome]:
        policy = self.config.abstention
        base_runtime = {e.rule for e in evidence if e.source != "static" and e.confidence >= policy.min_evidence_confidence}
        base_static = [e for e in evidence if e.source == "static" and e.confidence >= policy.min_evidence_confidence]
        out: list[CandidateOutcome] = []
        for i, p in enumerate(patches):
            key = f"c{i}"
            if self.config.mode != "naive-llm":
                ok, why = policy.gate_patch(p, evidence)
                if not ok:
                    out.append(CandidateOutcome(key, p, "rejected", why))
                    continue
            try:
                new_sources = p.apply(self.sources)
            except PatchConflict as exc:
                out.append(CandidateOutcome(key, p, "rejected", f"does not apply: {exc}"))
                continue
            if not all(parses(t) for t in new_sources.values()):
                out.append(CandidateOutcome(key, p, "rejected", "patched code does not parse"))
                continue
            new_static = [e for e in self.lint(new_sources) if e.confidence >= policy.min_evidence_confidence]
            introduced = {e.rule for e in new_static} - {e.rule for e in base_static}
            if introduced:
                out.append(CandidateOutcome(key, p, "rejected", f"introduces static findings {sorted(introduced)}"))
                continue
            res = self.runner.run(0, patch=p, max_steps=self.config.max_steps, probes=self.config.probes, tag=f"pre-{key}")
            tr = res.load_trace()
            if not res.ok or tr is None:
                out.append(CandidateOutcome(key, p, "rejected", f"patched run failed: {res.error}"))
                continue
            after = {e.rule for e in self.invariants.check(tr) if e.confidence >= policy.min_evidence_confidence}
            new_v = after - base_runtime
            if new_v:
                out.append(CandidateOutcome(key, p, "rejected", f"introduces invariant violations {sorted(new_v)}"))
                continue
            resolved_runtime = base_runtime - after
            resolved_static = len(base_static) - len([e for e in new_static if e.rule in {b.rule for b in base_static}])
            if base_runtime and not resolved_runtime and resolved_static <= 0:
                out.append(CandidateOutcome(key, p, "rejected", "violated invariants still violated"))
                continue
            p.meta["resolved_invariants"] = sorted(resolved_runtime)
            p.meta["remaining_invariants"] = sorted(after & base_runtime)
            out.append(CandidateOutcome(key, p, "prefiltered", f"resolves {sorted(resolved_runtime) or 'static findings'}"))
        return out

    # --------------------------------------------------------------- validate
    def validate(self, cands: list[CandidateOutcome]) -> ValidationReport:
        patches = {c.key: c.patch for c in cands}
        metric = self.config.metric

        def evaluate_batch(jobs: list[tuple[str, int, float]]) -> list[float | None]:
            runs = self.runner.run_many([
                {"seed": s, "patch": patches.get(k), "budget": b, "trace": False, "probes": False, "tag": f"val-{k}"}
                for k, s, b in jobs
            ])
            vals: list[float | None] = []
            for r in runs:
                v = r.metric(metric) if r.ok else float("nan")
                vals.append(v if math.isfinite(v) else None)
            return vals

        v = SequentialValidator(config=self.config.acceptance(), evaluate_batch=evaluate_batch, log=self.log)
        return v.run([c.key for c in cands], "baseline")

    # ------------------------------------------------------------------ run
    def run(self) -> RepairReport:
        t0 = time.time()
        cfg = self.config
        evidence, trace = self.detect()
        report = RepairReport("no_bug_found", self.files[0], self.sources, evidence,
                              trace_summary=trace.summary() if trace else None)
        self.log(f"{len(evidence)} evidence item(s)")
        diagnosis = self.diagnose(evidence, trace)
        report.diagnosis = diagnosis
        if cfg.mode != "naive-llm":
            abstain, why = cfg.abstention.gate_before_repair(evidence, diagnosis)
            if abstain:
                report.abstained, report.abstain_reason = True, why
                return self._finish(report, t0)
        patches = self.propose(evidence, diagnosis)
        if not patches:
            report.status = "diagnosed_only"
            report.notes.append("no candidate patch could be generated (enable an LLM backend for non-template bugs)")
            return self._finish(report, t0)
        outcomes = self.prefilter(patches, evidence)
        report.candidates = outcomes
        survivors = [c for c in outcomes if c.status == "prefiltered"]
        if not survivors:
            report.status = "candidates_rejected"
            report.abstained, report.abstain_reason = True, "no candidate passed grounding and invariant pre-filters"
            self._remember(evidence, outcomes, None)
            return self._finish(report, t0)
        accepted_key: str | None = None
        if cfg.metric and cfg.validate:
            val = self.validate(survivors)
            report.validation = val
            if val.baseline_failed and not val.accepted:
                report.notes.append(f"no run reported a finite `{cfg.metric}`: call trainmedic.report({cfg.metric}=...) in the "
                                    "script, or use --metric train_loss_final --direction min")
            for c in survivors:
                rec = val.records.get(c.key)
                if rec is not None:
                    c.status, c.reason = rec.status, rec.reason
            accepted_key = val.best
            if accepted_key is None:
                report.status = "candidates_rejected"
                report.abstained, report.abstain_reason = True, "no candidate showed a statistically significant improvement"
        else:
            accepted_key = survivors[0].key
            survivors[0].status = "accepted"
            survivors[0].reason += " (invariant-validated only; pass --metric for statistical validation)"
        if accepted_key is not None:
            acc = next(c for c in outcomes if c.key == accepted_key)
            report.accepted = acc.patch
            report.status = "repaired" if (cfg.metric and cfg.validate) else "unvalidated_fix"
            if cfg.minimize and len(acc.patch.edits) > 1:
                report.minimized = self._minimize(acc.patch, evidence)
        self._remember(evidence, outcomes, accepted_key)
        return self._finish(report, t0)

    def _minimize(self, patch: Patch, evidence: list[Evidence]) -> MinimizationResult:
        policy = self.config.abstention
        base = {e.rule for e in evidence if e.source != "static" and e.confidence >= policy.min_evidence_confidence}
        resolved = set(patch.meta.get("resolved_invariants", []))
        inv = InvariantOracle(self.runner, self.invariants, resolved, base, max_steps=self.config.max_steps,
                              min_confidence=policy.min_evidence_confidence)
        oracles: list[Any] = [inv]
        if self.config.metric:
            oracles.append(StatisticalOracle(self.runner, self.config.metric, self.config.direction, full=patch))
        return minimize_patch(patch, CombinedOracle(*oracles))

    def _remember(self, evidence: list[Evidence], outcomes: list[CandidateOutcome], accepted: str | None) -> None:
        if self.memory is None:
            return
        for c in outcomes:
            causes = [e.cause for e in evidence if e.id in c.patch.evidence_ids and e.cause]
            status = "accepted" if c.key == accepted else "rejected"
            self.memory.record(evidence, causes, c.patch, self.sources, status, self.files[0])

    def _finish(self, report: RepairReport, t0: float) -> RepairReport:
        report.seconds = time.time() - t0
        report.runs = self.runner.n_runs
        report.notes.extend(self.notes)
        return report


def repair(script: str, args: list[str] | tuple[str, ...] = (), **kw: Any) -> RepairReport:
    """Convenience wrapper: ``trainmedic.repair.agent.repair("train.py", metric="val_acc")``."""
    backend = kw.pop("backend", None)
    cfg = RepairConfig(**kw)
    return RepairSession(script, args, cfg, backend=backend).run()
