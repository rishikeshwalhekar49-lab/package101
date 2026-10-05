"""Do-no-harm evaluation of repair agents on buggy and bug-free pipelines.

For every buggy twin we measure detection, repair (validated against the clean
twin, not just against the agent's own claim) and patch size; for every clean
pipeline and benign mutant we measure abstention, false alarms and *damage*
(a proposed patch that makes a correct pipeline measurably worse or introduces
invariant violations). Headline numbers: repair rate (alpha), damage rate (beta),
abstention accuracy, and patch minimality -- per category and per
contamination split.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..detect.invariants import InvariantSet
from ..repair.agent import RepairConfig, RepairSession
from ..repair.patch import Patch
from ..validate.halving import Rung, damage_test
from ..validate.runner import Runner, RunSpec
from ..validate.stats import mean
from .suite import Instance, Suite

AGENTS = ("rules", "llm", "hybrid", "naive-llm", "oracle", "abstain")


@dataclass
class EvalConfig:
    agent: str = "rules"
    llm: str | None = None
    labels: tuple[str, ...] = ("silent_failure", "clean", "benign", None)  # None = unverified buggy twins
    seeds: int = 4
    budget: float = 1.0
    min_drop: float = 0.02
    alpha: float = 0.1
    workers: int = 2
    max_steps: int = 40
    validate: bool = True
    minimize: bool = True
    rungs: list[Rung] = field(default_factory=lambda: [Rung(2, 0.34)])
    confirm_seeds: int = 5
    confirm_budget: float = 0.67


@dataclass
class InstanceResult:
    id: str
    pipeline: str
    kind: str
    label: str | None
    operator: str | None
    category: str | None
    split: str
    status: str = ""
    abstained: bool = False
    proposed: bool = False
    detected: bool = False
    detected_expected: bool = False
    patch_size: int = 0
    minimal_size: int = 0
    fix_kind: str | None = None
    repaired: bool | None = None
    exact_match: bool | None = None
    false_alarm: bool | None = None
    damage: bool | None = None
    damage_reason: str = ""
    seconds: float = 0.0
    runs: int = 0
    error: str | None = None


def _norm(src: str) -> str:
    return "\n".join(l.rstrip() for l in src.strip().splitlines() if l.strip())


class Evaluator:
    def __init__(self, suite: Suite, cfg: EvalConfig, *, backend: Any = None, log: Callable[[str], None] = print) -> None:
        if cfg.agent not in AGENTS:
            raise ValueError(f"agent must be one of {AGENTS}")
        self.suite, self.cfg, self.backend, self.log = suite, cfg, backend, log
        self._clean_vals: dict[str, list[float]] = {}

    def _values(self, inst: Instance, path: str, patch: Patch | None = None) -> list[float]:
        with Runner(RunSpec(path), workers=self.cfg.workers) as r:
            res = r.run_many([{"seed": 100 + s, "patch": patch, "budget": self.cfg.budget, "trace": False, "probes": False}
                              for s in range(self.cfg.seeds)])
        return [x.metric(inst.metric) if x.ok else float("nan") for x in res]

    def clean_values(self, inst: Instance) -> list[float]:
        if inst.pipeline not in self._clean_vals:
            self._clean_vals[inst.pipeline] = self._values(inst, inst.clean_path)
        return self._clean_vals[inst.pipeline]

    def _agent(self, inst: Instance) -> tuple[Any, Patch | None]:
        cfg = self.cfg
        if cfg.agent == "abstain":
            return None, None
        if cfg.agent == "oracle":
            return None, Patch.from_dict(inst.ground_truth) if inst.ground_truth else None
        rc = RepairConfig(mode=cfg.agent, llm=cfg.llm, metric=inst.metric if cfg.validate else None, direction=inst.direction,
                          max_steps=cfg.max_steps, validate=cfg.validate, minimize=cfg.minimize, rungs=cfg.rungs,
                          confirm_seeds=cfg.confirm_seeds, confirm_budget=cfg.confirm_budget, workers=cfg.workers)
        rep = RepairSession(inst.path, config=rc, backend=self.backend).run()
        return rep, rep.final_patch

    def run_instance(self, inst: Instance) -> InstanceResult:
        t0 = time.time()
        r = InstanceResult(inst.id, inst.pipeline, inst.kind, inst.label, inst.operator, inst.category, inst.split)
        try:
            rep, patch = self._agent(inst)
        except Exception as exc:  # an agent failure is a result, not a crash of the harness
            r.error = f"{type(exc).__name__}: {exc}"
            r.seconds = time.time() - t0
            return r
        if rep is not None:
            r.status, r.abstained, r.runs = rep.status, rep.abstained, rep.runs
            act = [e for e in rep.evidence if e.confidence >= 0.5]
            r.detected = bool(act)
            want = set(inst.expected_rules) | set(inst.expected_invariants)
            r.detected_expected = bool(want & {e.rule for e in act}) if want else r.detected
            if rep.minimized is not None:
                r.minimal_size = rep.minimized.minimal.size
        else:
            r.status = "oracle" if patch is not None else "abstained"
            r.abstained = patch is None
        r.proposed = patch is not None
        if patch is not None:
            r.patch_size = patch.size if not r.minimal_size else rep.minimized.original.size  # type: ignore[union-attr]
            r.minimal_size = r.minimal_size or patch.size
            r.fix_kind = "hyperparameter" if patch.kinds == {"hyperparameter"} else ("mixed" if "hyperparameter" in patch.kinds else "code")
        src = Path(inst.path).read_text()
        if inst.kind == "buggy" and inst.label != "benign":
            r.repaired = False
            if patch is not None:
                patched = patch.apply({"train.py": src})["train.py"]
                r.exact_match = _norm(patched) == _norm(Path(inst.clean_path).read_text())
                if r.exact_match:
                    r.repaired = True
                else:
                    vals = self._values(inst, inst.path, patch)
                    clean = self.clean_values(inst)
                    worse, cmp = damage_test(vals, clean, direction=inst.direction, alpha=self.cfg.alpha)
                    r.repaired = not worse and -cmp.mean_diff < self.cfg.min_drop
        else:  # clean pipeline or benign mutant: any proposal is a false alarm; check for damage
            r.false_alarm = patch is not None
            r.damage = False
            if patch is not None:
                orig = self.clean_values(inst) if inst.kind == "clean" else self._values(inst, inst.path)
                vals = self._values(inst, inst.path, patch)
                worse, cmp = damage_test(vals, orig, direction=inst.direction, alpha=self.cfg.alpha)
                consistent = all((v < o) if inst.direction == "max" else (v > o) for v, o in zip(vals, orig))
                if worse or (-cmp.mean_diff >= self.cfg.min_drop and consistent):
                    r.damage, r.damage_reason = True, f"metric drop {-cmp.mean_diff:.4f} (p={cmp.p_worse:.3f})"
                else:
                    new_v = self._new_violations(inst, patch)
                    if new_v:
                        r.damage, r.damage_reason = True, f"introduces invariant violations {sorted(new_v)}"
        r.seconds = time.time() - t0
        return r

    def _new_violations(self, inst: Instance, patch: Patch) -> set[str]:
        inv = InvariantSet.default()
        with Runner(RunSpec(inst.path)) as runner:
            a = runner.run(0, max_steps=self.cfg.max_steps, tag="dmg-orig").load_trace()
            b = runner.run(0, patch=patch, max_steps=self.cfg.max_steps, tag="dmg-patched").load_trace()
        if a is None or b is None:
            return {"crash"} if b is None else set()
        before = {e.rule for e in inv.check(a) if e.confidence >= 0.5}
        after = {e.rule for e in inv.check(b) if e.confidence >= 0.5}
        return after - before

    def run(self, instances: list[Instance] | None = None, out: str | Path | None = None) -> dict[str, Any]:
        insts = instances if instances is not None else [i for i in self.suite.instances if i.label in self.cfg.labels]
        results = []
        for k, inst in enumerate(insts, 1):
            self.log(f"[{k}/{len(insts)}] {inst.id} ({inst.kind}, {inst.label})")
            res = self.run_instance(inst)
            self.log(f"    -> status={res.status} proposed={res.proposed} repaired={res.repaired} damage={res.damage} "
                     f"({res.seconds:.0f}s){' ERROR ' + res.error if res.error else ''}")
            results.append(res)
        summary = summarize(results, self.cfg)
        if out is not None:
            p = Path(out)
            p.mkdir(parents=True, exist_ok=True)
            (p / "results.json").write_text(json.dumps({"config": asdict(self.cfg), "summary": summary,
                                                        "results": [asdict(r) for r in results]}, indent=2, default=str))
            (p / "summary.md").write_text(summary_markdown(summary))
        return {"summary": summary, "results": results}


def _rate(xs: list[bool]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def summarize(results: list[InstanceResult], cfg: EvalConfig | None = None) -> dict[str, Any]:
    """Headline metrics. False alarms / abstention are measured on truly clean pipelines;
    damage on clean pipelines *and* benign mutants (code with a defect that does not move
    the metric -- restoring it is fine, making it worse is not)."""
    ok = [r for r in results if r.error is None]
    buggy = [r for r in ok if r.kind == "buggy" and r.label != "benign"]
    clean = [r for r in ok if r.kind == "clean"]
    benign = [r for r in ok if r.kind == "buggy" and r.label == "benign"]

    def block(b: list[InstanceResult], c: list[InstanceResult], bn: list[InstanceResult]) -> dict[str, Any]:
        proposed = [r for r in b if r.proposed]
        return {
            "n_buggy": len(b),
            "n_clean": len(c),
            "n_benign": len(bn),
            "detection_rate": _rate([r.detected for r in b]),
            "expected_signal_rate": _rate([r.detected_expected for r in b]),
            "repair_rate": _rate([bool(r.repaired) for r in b]),
            "exact_match_rate": _rate([bool(r.exact_match) for r in proposed]),
            "proposal_rate_buggy": _rate([r.proposed for r in b]),
            "false_alarm_rate": _rate([bool(r.false_alarm) for r in c]),
            "abstention_accuracy": _rate([not r.proposed for r in c]),
            "damage_rate": _rate([bool(r.damage) for r in c + bn]),
            "benign_patched_rate": _rate([r.proposed for r in bn]),
            "mean_patch_size": mean([r.patch_size for r in proposed]) if proposed else None,
            "mean_minimal_size": mean([r.minimal_size for r in proposed]) if proposed else None,
            "hyperparameter_fixes": sum(r.fix_kind == "hyperparameter" for r in proposed),
        }

    out: dict[str, Any] = {"agent": cfg.agent if cfg else None, "overall": block(buggy, clean, benign),
                           "errors": sum(r.error is not None for r in results),
                           "seconds": sum(r.seconds for r in results)}
    cats = sorted({r.category for r in buggy if r.category})
    out["by_category"] = {c: block([r for r in buggy if r.category == c], [], []) for c in cats}
    splits = sorted({r.split for r in ok})
    out["by_split"] = {s: block([r for r in buggy if r.split == s], [r for r in clean if r.split == s],
                                [r for r in benign if r.split == s]) for s in splits}
    return out


def results_from_json(path: str | Path) -> list[InstanceResult]:
    d = json.loads(Path(path).read_text())
    return [InstanceResult(**r) for r in d["results"]]


def summary_markdown(s: dict[str, Any]) -> str:
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.2f}" if not math.isnan(v) else "-"
        return str(v)

    o = s["overall"]
    rows = ["detection_rate", "expected_signal_rate", "repair_rate", "exact_match_rate", "false_alarm_rate", "abstention_accuracy",
            "damage_rate", "benign_patched_rate", "mean_patch_size", "mean_minimal_size", "hyperparameter_fixes"]
    lines = [f"# trainmedic benchmark: agent `{s.get('agent')}`", "",
             f"{o['n_buggy']} buggy pipelines (silent failures), {o['n_clean']} clean pipelines, {o['n_benign']} benign mutants, "
             f"{s['errors']} agent errors, {s['seconds']:.0f}s total.", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {fmt(o[k])} |" for k in rows]
    if s.get("by_category"):
        lines += ["", "## By category", "", "| category | n | detection | repair |", "|---|---|---|---|"]
        for c, b in s["by_category"].items():
            lines.append(f"| {c} | {b['n_buggy']} | {fmt(b['detection_rate'])} | {fmt(b['repair_rate'])} |")
    if s.get("by_split"):
        lines += ["", "## By contamination split", "", "| split | buggy | repair | clean+benign | damage |", "|---|---|---|---|---|"]
        for sp, b in s["by_split"].items():
            lines.append(f"| {sp} | {b['n_buggy']} | {fmt(b['repair_rate'])} | {b['n_clean'] + b['n_benign']} | {fmt(b['damage_rate'])} |")
    return "\n".join(lines) + "\n"
