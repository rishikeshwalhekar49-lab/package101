"""Command-line interface: ``trainmedic <command> ...``.

Arguments after a bare ``--`` are passed to the training script, e.g.
``trainmedic check train.py --max-steps 30 -- --epochs 2``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import __version__
from .types import Evidence, Severity


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return argv, []


def _print_evidence(evidence: list[Evidence], fmt: str, min_conf: float) -> None:
    shown = [e for e in evidence if e.confidence >= min_conf]
    if fmt == "json":
        print(json.dumps([e.to_dict() for e in shown], indent=2, default=str))
        return
    if not shown:
        print("No issues found.")
        return
    for e in shown:
        print(e.short())


def _fails(evidence: list[Evidence], threshold: float) -> bool:
    return any(e.confidence >= threshold and e.severity != Severity.INFO for e in evidence)


def _rungs(spec: str | None):
    if not spec:
        return None
    from .validate.halving import Rung

    out = []
    for part in spec.split(","):
        s, b = part.split(":")
        out.append(Rung(int(s), float(b)))
    return out


# ----------------------------------------------------------------- commands
def cmd_lint(a, extra) -> int:
    from .detect.lint import fix_finding, lint_paths
    from .repair.patch import Patch

    ev = lint_paths(a.paths, select=a.select, ignore=a.ignore, min_confidence=a.min_confidence)
    from .types import EvidenceIds

    EvidenceIds().renumber(ev)
    _print_evidence(ev, a.format, a.min_confidence)
    if a.diff:
        for e in ev:
            if e.fixable and e.file:
                src = Path(e.file).read_text()
                new = fix_finding(src, e)
                if new:
                    print(f"\n# fix for {e.id} ({e.rule})")
                    print(Patch.from_sources(e.file, src, new).to_unified_diff({e.file: src}), end="")
    return 1 if (not a.exit_zero and _fails(ev, a.fail_under)) else 0


def cmd_rules(a, extra) -> int:
    from .detect.lint import all_rules

    for rid, r in sorted(all_rules().items()):
        print(f"{rid}  {r.name:34s} {r.category:18s} {'fix' if r.fixable else '   '}  {r.description}")
    return 0


def cmd_taxonomy(a, extra) -> int:
    from .diagnose.taxonomy import CAUSES

    for c in CAUSES.values():
        print(f"{c.id:28s} {c.category:18s} {c.attribution:22s} {c.title}")
    return 0


def _invariants(a):
    from .detect.invariants import InvariantSet
    from .detect.trace import Trace

    if getattr(a, "invariants", None):
        return InvariantSet.load(a.invariants)
    if getattr(a, "reference", None):
        return InvariantSet.infer(Trace.load_many(a.reference))
    return None


def cmd_check(a, extra) -> int:
    from .api import check_script

    res = check_script(a.script, extra, max_steps=a.max_steps or None, invariants=_invariants(a), lint=not a.no_lint,
                       probes=not a.no_probes, seed=a.seed, save_trace=a.save_trace)
    if res.trace is not None and a.format == "text":
        s = res.trace.summary()
        print(f"traced {s['steps']} optimizer step(s); first loss {s['first_loss']}, final loss {s['final_loss']}"
              f"{'; stopped at --max-steps' if s['stopped_early'] else ''}")
    if res.error and a.format == "text":
        print(f"run failed: {res.error}")
    _print_evidence(res.evidence, a.format, a.min_confidence)
    return 1 if (not a.exit_zero and _fails(res.evidence, a.fail_under)) else 0


def cmd_trace(a, extra) -> int:
    from .validate.runner import Runner, RunSpec

    with Runner(RunSpec(a.script, extra)) as r:
        res = r.run(a.seed, max_steps=a.max_steps or None, probes=not a.no_probes, tag="trace")
        tr = res.load_trace()
    if tr is None:
        print(f"no trace produced: {res.error}", file=sys.stderr)
        return 2
    tr.save(a.output)
    print(f"wrote {a.output} ({len(tr.events)} events, {tr.n_steps} steps){'' if res.ok else ' -- run failed: ' + str(res.error)}")
    return 0


def cmd_infer(a, extra) -> int:
    from .detect.invariants import InvariantSet
    from .detect.trace import Trace

    inv = InvariantSet.infer(Trace.load_many(a.traces))
    inv.save(a.output)
    print(f"inferred {len(inv)} invariant(s) from {len(a.traces)} healthy trace(s) -> {a.output}")
    for i in inv.invariants:
        print(f"  {i.name} {i.params if i.params else ''}")
    return 0


def cmd_trace_diff(a, extra) -> int:
    from .detect.trace import Trace
    from .detect.trace_diff import diff_traces

    d = diff_traces(Trace.load(a.reference), Trace.load(a.candidate))
    if d.first is None:
        print(f"no divergence over {d.compared_steps} aligned step(s)")
        return 0
    for e in d.to_evidence():
        print(e.message)
    for dv in sorted(d.divergences, key=lambda v: (v.step, v.signal))[: a.limit]:
        print(f"  step {dv.step:4d}  {dv.signal:20s} ref={dv.reference!s:>12.12} new={dv.candidate!s:>12.12}"
              + (f"  rel={dv.rel_diff:.3g}" if dv.rel_diff is not None else ""))
    return 1


def _repair_config(a, **over):
    from .repair.agent import RepairConfig

    cfg = RepairConfig(
        mode=getattr(a, "mode", "hybrid"), llm=getattr(a, "llm", None), metric=getattr(a, "metric", None),
        direction=getattr(a, "direction", "max"), max_steps=a.max_steps, alpha=getattr(a, "alpha", 0.05),
        min_effect=getattr(a, "min_effect", 0.0), rungs=_rungs(getattr(a, "rungs", None)),
        confirm_seeds=getattr(a, "confirm_seeds", None), confirm_budget=getattr(a, "confirm_budget", 1.0),
        workers=getattr(a, "workers", 2), minimize=not getattr(a, "no_minimize", False),
        invariants_path=getattr(a, "invariants", None), reference_traces=getattr(a, "reference", None) or [],
        use_memory=getattr(a, "memory", False), files=getattr(a, "files", None) or [], verbose=getattr(a, "verbose", False),
        allow_ungrounded=getattr(a, "allow_ungrounded", False),
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def cmd_diagnose(a, extra) -> int:
    from .repair.agent import RepairSession

    sess = RepairSession(a.script, extra, _repair_config(a, mode="hybrid" if a.llm else "rules"))
    ev, tr = sess.detect()
    d = sess.diagnose(ev, tr)
    if a.format == "json":
        print(json.dumps({"evidence": [e.to_dict() for e in ev], "diagnosis": d.to_dict(), "notes": sess.notes}, indent=2, default=str))
        return 0
    _print_evidence(ev, "text", 0.0)
    print()
    print(d.summary)
    print(f"attribution: {d.attribution}")
    for c in d.causes[:8]:
        where = f" {c.file}:{c.line}" if c.file and c.line else ""
        print(f"  - [{c.confidence:.2f}] {c.cause_id}{where}: {c.title}\n      fix: {c.suggested_fix}")
    for n in sess.notes:
        print(f"note: {n}")
    return 0


def cmd_repair(a, extra) -> int:
    from .repair.agent import RepairSession

    sess = RepairSession(a.script, extra, _repair_config(a))
    rep = sess.run()
    if a.report:
        rep.save(a.report)
    if a.format == "json":
        print(json.dumps(rep.to_dict(), indent=2, default=str))
    else:
        print(rep.to_markdown())
    diff = rep.diff()
    if diff and a.out:
        Path(a.out).write_text(diff)
        print(f"\npatch written to {a.out}")
    if diff and a.apply:
        rep.final_patch.apply_files(sess.spec.root_path)  # type: ignore[union-attr]
        print(f"\napplied patch to {', '.join(rep.final_patch.files)}")  # type: ignore[union-attr]
    return 0 if rep.status in ("repaired", "unvalidated_fix", "no_bug_found") else 3


def _load_patches(paths: list[str], spec) -> list:
    from .repair.patch import Patch

    sources = {}
    for f in spec.root_path.rglob("*.py"):
        if len(sources) > 500:
            break
        sources[spec.rel(f)] = f.read_text(encoding="utf-8")
    out = []
    for p in paths:
        text = Path(p).read_text()
        if p.endswith(".json"):
            out.append(Patch.from_dict(json.loads(text)))
        else:
            out.append(Patch.from_unified_diff(text, sources, origin=f"file:{p}"))
    return out


def cmd_validate(a, extra) -> int:
    from .validate.halving import AcceptanceConfig, SequentialValidator
    from .validate.runner import Runner, RunSpec

    spec = RunSpec(a.script, extra)
    patches = {f"p{i}": p for i, p in enumerate(_load_patches(a.patch, spec))}
    cfg = AcceptanceConfig(metric=a.metric, direction=a.direction, alpha=a.alpha, min_effect=a.min_effect,
                           confirm_seeds=a.confirm_seeds, confirm_budget=a.confirm_budget)
    if a.rungs:
        cfg.rungs = _rungs(a.rungs)
    with Runner(spec, workers=a.workers, verbose=a.verbose) as runner:
        def batch(jobs):
            res = runner.run_many([{"seed": s, "patch": patches.get(k), "budget": b, "trace": False, "probes": False} for k, s, b in jobs])
            return [r.metric(a.metric) if r.ok else None for r in res]

        rep = SequentialValidator(config=cfg, evaluate_batch=batch).run(list(patches), "baseline")
    for i, p in enumerate(a.patch):
        print(f"p{i} = {p}")
    print(rep.to_markdown())
    return 0 if rep.accepted else 3


def cmd_minimize(a, extra) -> int:
    from .detect.invariants import InvariantSet
    from .repair.minimize import CombinedOracle, InvariantOracle, StatisticalOracle, minimize_patch
    from .validate.runner import Runner, RunSpec

    spec = RunSpec(a.script, extra)
    patch = _load_patches([a.patch], spec)[0]
    inv = InvariantSet.default()
    with Runner(spec, workers=a.workers) as runner:
        base = runner.run(0, max_steps=a.max_steps, tag="base").load_trace()
        full = runner.run(0, patch=patch, max_steps=a.max_steps, tag="full").load_trace()
        b_rules = {e.rule for e in inv.check(base) if e.confidence >= 0.5} if base else set()
        f_rules = {e.rule for e in inv.check(full) if e.confidence >= 0.5} if full else set()
        oracles: list[Any] = [InvariantOracle(runner, inv, b_rules - f_rules, b_rules, max_steps=a.max_steps)]
        if a.metric:
            oracles.append(StatisticalOracle(runner, a.metric, a.direction, full=patch))
        res = minimize_patch(patch, CombinedOracle(*oracles))
    sources = spec.read_sources(res.minimal.files) if res.minimal.edits else {}
    print(f"{res.original.size} -> {res.minimal.size} changed lines ({res.reduction:.0%} smaller), {res.tests} oracle calls, kind={res.kind}")
    if res.minimal.edits:
        print(res.minimal.to_unified_diff(sources))
    for i, alt in enumerate(res.alternatives):
        print(f"# alternative fix {i}")
        print(alt.to_unified_diff(spec.read_sources(alt.files)))
    if a.out and res.minimal.edits:
        Path(a.out).write_text(res.minimal.to_unified_diff(sources))
    return 0


def cmd_compile_diff(a, extra) -> int:
    import importlib.util

    from .detect.compile_diff import compile_diff

    path, _, fn = a.target.partition(":")
    spec = importlib.util.spec_from_file_location("_tm_compile_target", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.path.insert(0, str(Path(path).resolve().parent))
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    case = getattr(mod, fn or "make_case")()
    rep = compile_diff(case["model"], tuple(case["inputs"]), loss_fn=case.get("loss_fn"), target=case.get("target"),
                       backend=a.backend, tol=a.tol, eval_mode=not a.train_mode)
    if a.format == "json":
        print(json.dumps(rep.to_dict(), indent=2))
    else:
        print(f"backend={rep.backend} matches={rep.matches} max_rel_err={rep.output_rel_err:.3g} attribution={rep.attribution}")
        for e in rep.to_evidence():
            print(" ", e.message)
    return 0 if rep.matches else 1


def cmd_config(a, extra) -> int:
    from .detect.config_check import check_config_file, fix_config_file

    kind, ev = check_config_file(a.file, a.world_size)
    from .types import EvidenceIds

    EvidenceIds().renumber(ev)
    print(f"{a.file}: detected {kind} config")
    _print_evidence(ev, a.format, 0.0)
    if a.fix:
        changes = fix_config_file(a.file, a.fix, a.world_size)
        print(f"wrote {a.fix}: " + ("; ".join(changes) if changes else "no automatic changes"))
    return 1 if _fails(ev, 0.5) else 0


def cmd_memory(a, extra) -> int:
    from .memory import ExperienceMemory

    mem = ExperienceMemory(a.path)
    if a.action == "clear":
        mem.clear()
        print(f"cleared {mem.path}")
        return 0
    stats = mem.fix_stats()
    print(f"{mem.path}: {len(mem.records())} record(s), {len(stats)} distinct fix(es)")
    for key, s in sorted(stats.items(), key=lambda kv: -kv[1]["support"]):
        print(f"  {key} support={s['support']} precision={s['precision']:.2f} damage={s['damage']} :: {s['description'][:80]}")
    return 0


def cmd_bench(a, extra) -> int:
    from .bench import suite as S

    if a.bench_cmd == "list":
        print("pipelines:")
        for name, info in S.PIPELINES.items():
            print(f"  {name:14s} metric={info['metric']} ({info['direction']})")
        print("mutation operators:")
        for name, desc in S.list_operators():
            print(f"  {name:28s} {desc}")
        return 0
    if a.bench_cmd == "pipelines":
        dest = Path(a.dest)
        dest.mkdir(parents=True, exist_ok=True)
        for name in S.PIPELINES:
            S.copy_pipeline(name, dest)
        print(f"copied {len(S.PIPELINES)} pipelines to {dest}")
        return 0
    if a.bench_cmd == "build":
        extra_scripts = {}
        for item in a.script or []:
            name, rest = item.split("=", 1)
            path, metric, direction = (rest.split(":") + ["max"])[:3]
            extra_scripts[name] = {"path": path, "metric": metric, "direction": direction}
        suite = S.build_suite(a.out, pipelines=a.pipelines or ([] if extra_scripts and a.only_scripts else None),
                              operators=a.operators, extra_scripts=extra_scripts, max_sites=a.max_sites, model_cutoff=a.cutoff)
        print(f"built {len(suite.instances)} instance(s) in {a.out} "
              f"({sum(i.kind == 'buggy' for i in suite.instances)} buggy, {sum(i.kind == 'clean' for i in suite.instances)} clean)")
        return 0
    if a.bench_cmd == "verify":
        suite = S.verify_suite(S.Suite.load(a.suite), seeds=a.seeds, workers=a.workers)
        labels: dict[str, int] = {}
        for i in suite.instances:
            labels[str(i.label)] = labels.get(str(i.label), 0) + 1
        print(f"labels: {labels}")
        return 0
    if a.bench_cmd == "run":
        from .bench.evaluate import EvalConfig, Evaluator, summary_markdown

        suite = S.Suite.load(a.suite)
        insts = [i for i in suite.instances if (not a.only or i.id in a.only) and (not a.pipelines or i.pipeline in a.pipelines)]
        labels = tuple(a.labels) if a.labels else ("silent_failure", "clean", "benign", None)
        insts = [i for i in insts if i.label in labels or (None in labels and i.label is None)]
        cfg = EvalConfig(agent=a.agent, llm=a.llm, seeds=a.seeds, workers=a.workers, validate=not a.no_validate,
                         minimize=not a.no_minimize)
        out = Evaluator(suite, cfg).run(insts, a.out)
        print(summary_markdown(out["summary"]))
        return 0
    if a.bench_cmd == "report":
        from .bench.evaluate import results_from_json, summarize, summary_markdown

        res = results_from_json(Path(a.results) / "results.json")
        summary = summarize(res)
        summary["agent"] = json.loads((Path(a.results) / "results.json").read_text()).get("config", {}).get("agent")
        print(summary_markdown(summary))
        return 0
    if a.bench_cmd == "mutate":
        from .bench.mutations import mutate

        src = Path(a.script).read_text()
        dest = Path(a.out)
        dest.mkdir(parents=True, exist_ok=True)
        muts = mutate(src, a.script, a.operators, max_sites=a.max_sites)
        for k, m in enumerate(muts):
            p = dest / f"{Path(a.script).stem}__{m.operator}_{k}.py"
            p.write_text(m.source)
            print(f"{p}  ({m.description}, line {m.line})")
        return 0
    if a.bench_cmd == "agentgen":
        from .bench.agentgen import TASK_SPECS, generate
        from .llm.backend import get_backend

        backend = get_backend(a.llm)
        if backend is None:
            print("bench agentgen needs --llm", file=sys.stderr)
            return 2
        specs = [s for s in TASK_SPECS if not a.specs or s["id"] in a.specs]
        generate(backend, a.out, specs=specs, n=a.n)
        return 0
    if a.bench_cmd == "audit":
        from .bench.agentgen import audit

        audit(a.dir, max_steps=a.max_steps)
        print(f"wrote {Path(a.dir) / 'labels.jsonl'} and LABELING.md")
        return 0
    if a.bench_cmd == "labels":
        from .bench.agentgen import summarize_labels

        print(json.dumps(summarize_labels(a.dir), indent=2))
        return 0
    return 2


def cmd_demo(a, extra) -> int:
    from .bench.mutations import mutate
    from .bench.suite import PIPELINE_DIR
    from .repair.agent import RepairConfig, RepairSession
    from .validate.halving import Rung

    src = (PIPELINE_DIR / "mlp_classify.py").read_text()
    m = mutate(src, operators=[a.operator])
    if not m:
        print(f"operator {a.operator} does not apply to the demo pipeline", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "train.py"
        p.write_text(m[0].source)
        print(f"Injected bug: {m[0].description} (line {m[0].line}). Running detect -> diagnose -> repair -> validate ...\n")
        cfg = RepairConfig(mode="rules", metric="val_acc", rungs=[Rung(2, 0.25)], confirm_seeds=5, confirm_budget=0.5,
                           workers=a.workers)
        rep = RepairSession(str(p), config=cfg).run()
        print(rep.to_markdown())
    return 0


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="trainmedic", description="Detect, diagnose, repair and validate silent bugs in DL training code.")
    p.add_argument("--version", action="version", version=f"trainmedic {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common_out(sp, conf=0.3):
        sp.add_argument("--format", choices=["text", "json"], default="text")
        sp.add_argument("--min-confidence", type=float, default=conf)
        sp.add_argument("--fail-under", type=float, default=0.5, help="exit 1 if a finding has at least this confidence")
        sp.add_argument("--exit-zero", action="store_true")

    sp = sub.add_parser("lint", help="static analysis of training code")
    sp.add_argument("paths", nargs="+")
    sp.add_argument("--select", nargs="*")
    sp.add_argument("--ignore", nargs="*")
    sp.add_argument("--diff", action="store_true", help="print template fixes as unified diffs")
    common_out(sp)
    sp.set_defaults(fn=cmd_lint)

    sub.add_parser("rules", help="list static rules").set_defaults(fn=cmd_rules)
    sub.add_parser("taxonomy", help="list root causes").set_defaults(fn=cmd_taxonomy)

    def run_opts(sp, steps=40):
        sp.add_argument("script")
        sp.add_argument("--max-steps", type=int, default=steps)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--no-probes", action="store_true")

    sp = sub.add_parser("check", help="lint + run with tracing + check training invariants")
    run_opts(sp)
    sp.add_argument("--invariants", help="inferred invariants JSON (from `trainmedic infer`)")
    sp.add_argument("--reference", nargs="*", help="healthy reference traces to infer invariants from")
    sp.add_argument("--save-trace")
    sp.add_argument("--no-lint", action="store_true")
    common_out(sp)
    sp.set_defaults(fn=cmd_check)

    sp = sub.add_parser("trace", help="record a training trace")
    run_opts(sp)
    sp.add_argument("-o", "--output", required=True)
    sp.set_defaults(fn=cmd_trace)

    sp = sub.add_parser("trace-diff", help="first divergence between a reference trace and a new trace")
    sp.add_argument("reference")
    sp.add_argument("candidate")
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(fn=cmd_trace_diff)

    sp = sub.add_parser("infer", help="infer invariants from healthy traces")
    sp.add_argument("traces", nargs="+")
    sp.add_argument("-o", "--output", required=True)
    sp.set_defaults(fn=cmd_infer)

    def repair_opts(sp):
        sp.add_argument("script")
        sp.add_argument("--llm", help="anthropic[:MODEL] | cmd:COMMAND | none (default: $TRAINMEDIC_LLM)")
        sp.add_argument("--max-steps", type=int, default=40)
        sp.add_argument("--invariants")
        sp.add_argument("--reference", nargs="*")
        sp.add_argument("--files", nargs="*", help="extra source files (relative to the script) to analyse and patch")
        sp.add_argument("--workers", type=int, default=2)
        sp.add_argument("--allow-ungrounded", action="store_true")
        sp.add_argument("--verbose", "-v", action="store_true")
        sp.add_argument("--format", choices=["text", "json"], default="text")

    sp = sub.add_parser("diagnose", help="detect + root-cause diagnosis")
    repair_opts(sp)
    sp.set_defaults(fn=cmd_diagnose)

    def stat_opts(sp):
        sp.add_argument("--metric", help="metric reported via trainmedic.report(); enables statistical validation")
        sp.add_argument("--direction", choices=["max", "min"], default="max")
        sp.add_argument("--alpha", type=float, default=0.05)
        sp.add_argument("--min-effect", type=float, default=0.0)
        sp.add_argument("--rungs", help="successive-halving rungs as seeds:budget,... (default 2:0.34,3:0.67)")
        sp.add_argument("--confirm-seeds", type=int)
        sp.add_argument("--confirm-budget", type=float, default=1.0)

    sp = sub.add_parser("repair", help="full loop: detect, diagnose, propose, validate, minimise")
    repair_opts(sp)
    stat_opts(sp)
    sp.add_argument("--mode", choices=["rules", "llm", "hybrid", "naive-llm"], default="hybrid")
    sp.add_argument("--no-minimize", action="store_true")
    sp.add_argument("--memory", action="store_true", help="use and update the experience memory")
    sp.add_argument("--out", help="write the final patch (unified diff)")
    sp.add_argument("--report", help="write the report (.md or .json)")
    sp.add_argument("--apply", action="store_true", help="apply the final patch to the files")
    sp.set_defaults(fn=cmd_repair)

    sp = sub.add_parser("validate", help="statistically validate candidate patches")
    sp.add_argument("script")
    sp.add_argument("--patch", action="append", required=True, help="unified diff or patch JSON (repeatable)")
    stat_opts(sp)
    sp.add_argument("--workers", type=int, default=2)
    sp.add_argument("--verbose", "-v", action="store_true")
    sp.set_defaults(fn=cmd_validate)

    sp = sub.add_parser("minimize", help="delta-debug a patch to its minimal repairing subset")
    sp.add_argument("script")
    sp.add_argument("--patch", required=True)
    sp.add_argument("--metric")
    sp.add_argument("--direction", choices=["max", "min"], default="max")
    sp.add_argument("--max-steps", type=int, default=40)
    sp.add_argument("--workers", type=int, default=2)
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_minimize)

    sp = sub.add_parser("compile-diff", help="eager vs torch.compile differential check")
    sp.add_argument("target", help="file.py:factory returning {'model', 'inputs', ['loss_fn', 'target']}")
    sp.add_argument("--backend", default="inductor")
    sp.add_argument("--tol", type=float, default=1e-3)
    sp.add_argument("--train-mode", action="store_true")
    sp.add_argument("--format", choices=["text", "json"], default="text")
    sp.set_defaults(fn=cmd_compile_diff)

    sp = sub.add_parser("config", help="check (and fix) DeepSpeed / HF TrainingArguments JSON")
    sp.add_argument("file")
    sp.add_argument("--world-size", type=int)
    sp.add_argument("--fix", help="write a repaired config here")
    sp.add_argument("--format", choices=["text", "json"], default="text")
    sp.set_defaults(fn=cmd_config)

    sp = sub.add_parser("memory", help="inspect or clear the experience memory")
    sp.add_argument("action", choices=["show", "clear"])
    sp.add_argument("--path")
    sp.set_defaults(fn=cmd_memory)

    sp = sub.add_parser("demo", help="inject a bug into a bundled pipeline and repair it")
    sp.add_argument("--operator", default="remove_zero_grad")
    sp.add_argument("--workers", type=int, default=4)
    sp.set_defaults(fn=cmd_demo)

    b = sub.add_parser("bench", help="benchmark tooling")
    bs = b.add_subparsers(dest="bench_cmd", required=True)
    bs.add_parser("list")
    x = bs.add_parser("pipelines", help="copy the bundled pipelines")
    x.add_argument("dest")
    x = bs.add_parser("build", help="build buggy/clean twins")
    x.add_argument("out")
    x.add_argument("--pipelines", nargs="*")
    x.add_argument("--operators", nargs="*")
    x.add_argument("--script", nargs="*", help="NAME=path.py:metric:direction (your own pipelines)")
    x.add_argument("--only-scripts", action="store_true", help="skip bundled pipelines when --script is given")
    x.add_argument("--max-sites", type=int, default=1)
    x.add_argument("--cutoff", help="model training-data cutoff YYYY-MM-DD for the contamination split")
    x = bs.add_parser("verify", help="label mutants as silent_failure / benign / crash")
    x.add_argument("suite")
    x.add_argument("--seeds", type=int, default=4)
    x.add_argument("--workers", type=int, default=2)
    x = bs.add_parser("run", help="evaluate an agent")
    x.add_argument("suite")
    x.add_argument("--agent", choices=["rules", "llm", "hybrid", "naive-llm", "oracle", "abstain"], default="rules")
    x.add_argument("--llm")
    x.add_argument("--out")
    x.add_argument("--only", nargs="*")
    x.add_argument("--pipelines", nargs="*")
    x.add_argument("--labels", nargs="*")
    x.add_argument("--seeds", type=int, default=4)
    x.add_argument("--workers", type=int, default=2)
    x.add_argument("--no-validate", action="store_true")
    x.add_argument("--no-minimize", action="store_true")
    x = bs.add_parser("report", help="re-summarise a results directory")
    x.add_argument("results")
    x = bs.add_parser("mutate", help="write mutated copies of a script")
    x.add_argument("script")
    x.add_argument("-o", "--out", required=True)
    x.add_argument("--operators", nargs="*")
    x.add_argument("--max-sites", type=int, default=1)
    x = bs.add_parser("agentgen", help="have an LLM write training scripts to audit")
    x.add_argument("out")
    x.add_argument("--llm", required=True)
    x.add_argument("--n", type=int, default=1)
    x.add_argument("--specs", nargs="*")
    x = bs.add_parser("audit", help="run detectors over generated scripts; write a labelling sheet")
    x.add_argument("dir")
    x.add_argument("--max-steps", type=int, default=40)
    x = bs.add_parser("labels", help="summarise a filled-in labelling sheet")
    x.add_argument("dir")
    b.set_defaults(fn=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ours, extra = _split_argv(argv)
    args = build_parser().parse_args(ours)
    try:
        return int(args.fn(args, extra) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # user-facing errors without a traceback wall
        from .llm.backend import LLMError

        if isinstance(exc, (LLMError, FileNotFoundError, ValueError)):
            print(f"trainmedic: error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
