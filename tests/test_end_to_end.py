"""Subprocess-based tests: runner, mutations, full repair loop, benchmark harness, CLI."""

import json

import pytest

from trainmedic.bench.evaluate import EvalConfig, Evaluator, summarize
from trainmedic.bench.mutations import OPERATORS, mutate
from trainmedic.bench.suite import Suite, build_suite
from trainmedic.cli import main
from trainmedic.detect.lint import lint_source
from trainmedic.llm.backend import ScriptedBackend
from trainmedic.repair.agent import RepairConfig, RepairSession
from trainmedic.repair.patch import Patch
from trainmedic.sourceedit import parses
from trainmedic.validate.halving import Rung
from trainmedic.validate.runner import Runner, RunSpec

SEEDED = """
import torch
import trainmedic as tm
torch.manual_seed(42)
w = torch.randn(1).item()
tm.report(value=w)
"""


def test_every_operator_applies_somewhere(pipelines_dir):
    hit = set()
    for p in pipelines_dir.glob("*.py"):
        src = p.read_text()
        for m in mutate(src, p.name):
            assert parses(m.source) and m.source != src
            fired = {e.rule for e in lint_source(m.source) if e.confidence >= 0.5}
            for r in m.expected_rules:
                if m.operator not in ("remove_eval", "remove_train", "unscale_grad_accum"):  # context-dependent
                    assert r in fired, f"{m.operator} on {p.name}: expected {r}"
            hit.add(m.operator)
    assert hit == set(OPERATORS)


@pytest.mark.slow
def test_runner_seeds_budget_overrides_and_crash(write_script):
    script = write_script(SEEDED)
    with Runner(RunSpec(str(script)), workers=2) as r:
        a, b = r.run_many([{"seed": 0, "trace": False}, {"seed": 1, "trace": False}])
        assert a.ok and b.ok and a.metric("value") != b.metric("value")  # hard-coded seed is mixed with the run seed
        src = script.read_text()
        patch = Patch.from_sources("train.py", src, src.replace("tm.report(value=w)", "tm.report(value=123.0)"))
        assert r.run(0, patch=patch, trace=False).metric("value") == 123.0
        assert script.read_text() == src  # user file untouched
    bad = write_script("raise RuntimeError('boom')\n", "bad.py")
    with Runner(RunSpec(str(bad))) as r:
        res = r.run(0)
        assert not res.ok and "boom" in (res.error or "")


@pytest.mark.slow
def test_repair_loop_fixes_mutated_pipeline_and_abstains_on_clean(pipelines_dir, write_script):
    src = (pipelines_dir / "mlp_classify.py").read_text()
    buggy = write_script(mutate(src, operators=["remove_zero_grad"])[0].source)
    cfg = RepairConfig(mode="rules", metric="val_acc", rungs=[Rung(2, 0.25)], confirm_seeds=5, confirm_budget=0.5, workers=4)
    rep = RepairSession(str(buggy), config=cfg).run()
    assert rep.status == "repaired", rep.to_markdown()
    assert "zero_grad()" in rep.diff()
    clean = write_script(src, "clean.py")
    rep2 = RepairSession(str(clean), config=RepairConfig(mode="rules", metric="val_acc", workers=2)).run()
    assert rep2.status == "no_bug_found" and rep2.abstained and rep2.final_patch is None


@pytest.mark.slow
def test_hybrid_rejects_ungrounded_llm_patch_on_clean_code(pipelines_dir, write_script):
    src = (pipelines_dir / "mlp_classify.py").read_text()
    script = write_script(src)
    # an over-eager LLM that "finds" a bug in correct code and changes the learning rate
    backend = ScriptedBackend([
        {"bug_found": True, "summary": "lr too high", "attribution": "user_code",
         "causes": [{"cause_id": "lr_scale", "title": "LR", "description": "guess", "confidence": 0.95,
                     "evidence_ids": [], "file": "train.py", "line": 62}]},
        {"abstain": False, "reason": "", "candidates": [{"description": "lower lr", "evidence_ids": [],
                     "edits": [{"file": "train.py", "search": "lr=3e-3", "replace": "lr=3e-4"}]}]},
    ])
    rep = RepairSession(str(script), config=RepairConfig(mode="hybrid", metric="val_acc", workers=2), backend=backend).run()
    assert rep.final_patch is None and rep.abstained


@pytest.mark.slow
def test_bench_build_and_evaluate_oracle_and_abstain(tmp_path):
    suite = build_suite(tmp_path / "suite", pipelines=["mlp_classify"], operators=["remove_zero_grad", "softmax_before_ce"])
    assert len(suite.instances) == 3
    loaded = Suite.load(tmp_path / "suite")
    assert [i.kind for i in loaded.instances].count("buggy") == 2
    oracle = Evaluator(loaded, EvalConfig(agent="oracle", seeds=2, workers=2), log=lambda s: None)
    res = [oracle.run_instance(i) for i in loaded.instances if i.kind == "buggy"]
    assert all(r.repaired and r.exact_match for r in res)
    abstain = Evaluator(loaded, EvalConfig(agent="abstain"), log=lambda s: None)
    clean = [abstain.run_instance(i) for i in loaded.instances if i.kind == "clean"]
    s = summarize(res + clean)
    assert s["overall"]["repair_rate"] == 1.0 and s["overall"]["damage_rate"] == 0.0 and s["overall"]["abstention_accuracy"] == 1.0


def test_cli_commands(tmp_path, capsys):
    assert main(["rules"]) == 0
    assert main(["taxonomy"]) == 0
    f = tmp_path / "t.py"
    f.write_text("import torch\nfrom torch.utils.data import DataLoader\ntrain_loader = DataLoader(train_ds, batch_size=8)\n"
                 "val_loader = DataLoader(train_ds)\n")
    assert main(["lint", str(f)]) == 1
    assert main(["lint", str(f), "--exit-zero", "--format", "json"]) == 0
    out = capsys.readouterr().out
    assert "TM013" in out
    cfg = tmp_path / "ds.json"
    cfg.write_text(json.dumps({"train_batch_size": 10, "train_micro_batch_size_per_gpu": 4, "fp16": {"enabled": True},
                               "bf16": {"enabled": True}}))
    assert main(["config", str(cfg), "--fix", str(tmp_path / "fixed.json")]) == 1
    assert (tmp_path / "fixed.json").exists()
    assert main(["bench", "list"]) == 0


@pytest.mark.slow
def test_cli_check_and_trace(pipelines_dir, tmp_path, capsys):
    src = (pipelines_dir / "mlp_classify.py").read_text()
    buggy = tmp_path / "train.py"
    buggy.write_text(mutate(src, operators=["softmax_before_ce"])[0].source)
    assert main(["check", str(buggy), "--max-steps", "20"]) == 1
    out = capsys.readouterr().out
    assert "cross_entropy_gets_logits" in out and "TM008" in out
    assert main(["trace", str(pipelines_dir / "mlp_classify.py"), "--max-steps", "15", "-o", str(tmp_path / "ref.jsonl")]) == 0
    assert main(["infer", str(tmp_path / "ref.jsonl"), "-o", str(tmp_path / "inv.json")]) == 0
    assert main(["check", str(pipelines_dir / "mlp_classify.py"), "--max-steps", "15", "--invariants", str(tmp_path / "inv.json")]) == 0
