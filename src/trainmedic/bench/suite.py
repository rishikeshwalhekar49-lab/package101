"""Benchmark construction: buggy/bug-free twins with ground truth and metadata.

Each pipeline contributes one *clean* negative control plus one *buggy* twin per
applicable mutation operator. ``verify`` runs clean and buggy twins over a few
seeds and labels every mutant:

* ``silent_failure`` -- runs without error and the metric is significantly worse,
* ``benign``         -- runs and is not significantly worse (equivalent mutant;
                        kept as an extra negative control),
* ``crash``          -- raises (not a *silent* bug; excluded).

Metadata records provenance (``source``: bundled / user / agent-generated) and
creation date so results can be split by contamination risk (``split``).
"""

from __future__ import annotations

import datetime as _dt
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..repair.patch import Patch
from ..validate.halving import damage_test
from ..validate.runner import Runner, RunSpec
from .mutations import OPERATORS, mutate

PIPELINE_DIR = Path(__file__).parent / "pipelines"
PIPELINES: dict[str, dict[str, str]] = {
    "mlp_classify": {"metric": "val_acc", "direction": "max"},
    "cnn_patterns": {"metric": "val_acc", "direction": "max"},
    "tiny_lm": {"metric": "gen_acc", "direction": "max"},
    "dpo_toy": {"metric": "top1_acc", "direction": "max"},
}


@dataclass
class Instance:
    id: str
    pipeline: str
    kind: str  # "buggy" | "clean"
    path: str
    clean_path: str
    metric: str
    direction: str
    operator: str | None = None
    description: str = ""
    category: str | None = None
    cause: str | None = None
    fix_kind: str = "code"
    line: int | None = None
    expected_rules: list[str] = field(default_factory=list)
    expected_invariants: list[str] = field(default_factory=list)
    ground_truth: dict[str, Any] | None = None
    source: str = "bundled"
    created: str = field(default_factory=lambda: _dt.date.today().isoformat())
    split: str = "fresh"
    label: str | None = None
    verify: dict[str, Any] | None = None


@dataclass
class Suite:
    root: str
    instances: list[Instance] = field(default_factory=list)

    @property
    def path(self) -> Path:
        return Path(self.root)

    def save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "suite.json").write_text(json.dumps({"instances": [asdict(i) for i in self.instances]}, indent=2))

    @classmethod
    def load(cls, root: str | Path) -> "Suite":
        d = json.loads((Path(root) / "suite.json").read_text())
        return cls(str(root), [Instance(**i) for i in d["instances"]])

    def select(self, *, kinds: Iterable[str] | None = None, labels: Iterable[str] | None = None,
               pipelines: Iterable[str] | None = None, operators: Iterable[str] | None = None) -> list[Instance]:
        out = self.instances
        if kinds:
            out = [i for i in out if i.kind in set(kinds)]
        if labels:
            out = [i for i in out if i.label in set(labels) or (i.kind == "clean" and "clean" in set(labels))]
        if pipelines:
            out = [i for i in out if i.pipeline in set(pipelines)]
        if operators:
            out = [i for i in out if i.operator in set(operators) or i.kind == "clean"]
        return out


def _split(created: str, cutoff: str | None) -> str:
    if cutoff is None:
        return "fresh"
    return "post_cutoff" if created > cutoff else "pre_cutoff"


def build_suite(
    out_dir: str | Path,
    *,
    pipelines: Iterable[str] | None = None,
    operators: Iterable[str] | None = None,
    extra_scripts: dict[str, dict[str, str]] | None = None,
    max_sites: int = 1,
    model_cutoff: str | None = None,
    source: str = "bundled",
) -> Suite:
    """Create ``out_dir/<instance>/train.py`` files plus ``suite.json``.

    ``extra_scripts`` maps a name to ``{"path", "metric", "direction"}`` for user
    pipelines (each must report its metric via ``trainmedic.report``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    srcs: dict[str, dict[str, str]] = {}
    for name in pipelines or PIPELINES:
        srcs[name] = {"path": str(PIPELINE_DIR / f"{name}.py"), **PIPELINES[name], "source": source}
    for name, info in (extra_scripts or {}).items():
        srcs[name] = {"source": "user", **info}
    suite = Suite(str(out))
    for name, info in srcs.items():
        original = Path(info["path"]).read_text(encoding="utf-8")
        cdir = out / f"{name}__clean"
        cdir.mkdir(exist_ok=True)
        (cdir / "train.py").write_text(original)
        created = _dt.date.today().isoformat()
        suite.instances.append(Instance(
            id=f"{name}__clean", pipeline=name, kind="clean", path=str(cdir / "train.py"), clean_path=str(cdir / "train.py"),
            metric=info["metric"], direction=info["direction"], source=info.get("source", source), created=created,
            split=_split(created, model_cutoff), label="clean",
        ))
        for k, m in enumerate(mutate(original, f"{name}.py", operators, max_sites=max_sites)):
            iid = f"{name}__{m.operator}" + (f"_{k}" if max_sites > 1 else "")
            if any(i.id == iid for i in suite.instances):
                iid = f"{iid}_{k}"
            d = out / iid
            d.mkdir(exist_ok=True)
            (d / "train.py").write_text(m.source)
            gt = Patch.from_sources("train.py", m.source, original, origin="ground_truth", description=f"undo {m.operator}")
            suite.instances.append(Instance(
                id=iid, pipeline=name, kind="buggy", path=str(d / "train.py"), clean_path=str(cdir / "train.py"),
                metric=info["metric"], direction=info["direction"], operator=m.operator, description=m.description,
                category=m.category, cause=m.cause, fix_kind=m.kind, line=m.line, expected_rules=m.expected_rules,
                expected_invariants=m.expected_invariants, ground_truth=gt.to_dict(), source=info.get("source", source),
                created=created, split=_split(created, model_cutoff),
            ))
    suite.save()
    return suite


def verify_suite(suite: Suite, *, seeds: int = 4, budget: float = 1.0, alpha: float = 0.1, workers: int = 2,
                 min_drop: float = 0.02, log=print) -> Suite:
    """Label each buggy twin as silent_failure / benign / crash (see module docstring)."""
    clean_vals: dict[str, list[float]] = {}
    for inst in suite.instances:
        if inst.kind != "clean":
            continue
        with Runner(RunSpec(inst.path), workers=workers) as r:
            res = r.run_many([{"seed": s, "budget": budget, "trace": False, "probes": False} for s in range(seeds)])
        clean_vals[inst.pipeline] = [x.metric(inst.metric) for x in res]
        inst.verify = {"values": clean_vals[inst.pipeline]}
        log(f"clean {inst.pipeline}: {inst.metric}={clean_vals[inst.pipeline]}")
    for inst in suite.instances:
        if inst.kind != "buggy":
            continue
        with Runner(RunSpec(inst.path), workers=workers) as r:
            res = r.run_many([{"seed": s, "budget": budget, "trace": False, "probes": False} for s in range(seeds)])
        if not all(x.ok for x in res):
            inst.label = "crash"
            inst.verify = {"error": next((x.error for x in res if not x.ok), None)}
        else:
            vals = [x.metric(inst.metric) for x in res]
            worse, cmp = damage_test(vals, clean_vals[inst.pipeline], direction=inst.direction, alpha=alpha)
            drop = -cmp.mean_diff
            consistent = all((v < c) if inst.direction == "max" else (v > c) for v, c in zip(vals, clean_vals[inst.pipeline]))
            inst.label = "silent_failure" if worse or (drop >= min_drop and consistent) else "benign"
            inst.verify = {"values": vals, "mean_drop": drop, "p_worse": cmp.p_worse}
        log(f"{inst.id}: {inst.label} {inst.verify}")
    suite.save()
    return suite


def copy_pipeline(name: str, dest: str | Path) -> Path:
    dest = Path(dest)
    shutil.copy(PIPELINE_DIR / f"{name}.py", dest)
    return dest


def list_operators() -> list[tuple[str, str]]:
    return [(o.name, o.description) for o in OPERATORS.values()]
