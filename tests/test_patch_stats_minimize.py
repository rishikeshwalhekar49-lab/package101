import random

import pytest

from trainmedic.repair.minimize import ddmin, fix_kind, minimize_patch
from trainmedic.repair.patch import Patch, PatchConflict, apply_unified_diff, classify_edit, search_replace
from trainmedic.validate.halving import AcceptanceConfig, Rung, SequentialValidator, damage_test, fixed_k_validate
from trainmedic.validate.stats import compare_paired, holm, seeds_needed, signflip_pvalue

OLD = "a = 1\nb = 2\nc = 3\nd = 4\n"
NEW = "a = 1\nb = 20\nc = 3\nx = 9\nd = 4\n"


def test_patch_roundtrip_and_subsets():
    p = Patch.from_sources("f.py", OLD, NEW)
    assert p.apply({"f.py": OLD})["f.py"] == NEW
    assert len(p.edits) == 2
    for i in range(2):
        out = p.subset([i]).apply({"f.py": OLD})["f.py"]
        assert out != OLD and out != NEW
    diff = p.to_unified_diff({"f.py": OLD})
    assert apply_unified_diff(diff, {"f.py": OLD})["f.py"] == NEW
    assert Patch.from_unified_diff(diff, {"f.py": OLD}).apply({"f.py": OLD})["f.py"] == NEW
    assert Patch.from_dict(p.to_dict()).apply({"f.py": OLD})["f.py"] == NEW


def test_patch_conflicts():
    p = Patch.from_sources("f.py", OLD, NEW)
    with pytest.raises(PatchConflict):
        p.apply({"f.py": "a = 1\nb = 3\nc = 3\nd = 4\n"})
    q = Patch.from_sources("f.py", OLD, OLD.replace("b = 2", "b = 7"))
    with pytest.raises(PatchConflict):
        p.merge(q)
    assert len(p.merge(p).edits) == len(p.edits)


def test_search_replace_variants():
    src = "def f():\n    x = 1\n    return x\n"
    assert search_replace(src, "x = 1", "x = 2") == src.replace("x = 1", "x = 2")
    assert search_replace(src, "x = 1\nreturn x", "x = 3\nreturn x") == "def f():\n    x = 3\n    return x\n"
    with pytest.raises(PatchConflict):
        search_replace("a\na\n", "a", "b")
    with pytest.raises(PatchConflict):
        search_replace(src, "missing", "b")


def test_edit_classification_and_risks():
    hp = Patch.from_sources("f.py", "opt = Adam(p, lr=1e-3)\n", "opt = Adam(p, lr=1e-4)\n")
    assert hp.kinds == {"hyperparameter"}
    code = Patch.from_sources("f.py", "    opt.step()\n", "    opt.step()\n    opt.zero_grad()\n")
    assert code.kinds == {"code"}
    risky = Patch.from_sources("f.py", "self.drop = nn.Dropout(0.5)\nx = 1\n", "x = 1\n")
    assert "removes_dropout" in risky.risks()
    assert classify_edit(hp.edits[0]) == "hyperparameter"


def test_signflip_and_helpers():
    assert signflip_pvalue([1, 1, 1, 1, 1]) == pytest.approx(1 / 32)
    assert signflip_pvalue([1, -1, 1, -1]) > 0.3
    assert signflip_pvalue([0, 0, 0]) == 1.0
    assert seeds_needed(0.05) == 5
    assert holm([0.001, 0.04, 0.5], 0.05) == [True, False, False]
    cmp = compare_paired([0.9, 0.91, 0.92], [0.8, 0.82, 0.81])
    assert cmp.mean_diff > 0 and cmp.p_improve == pytest.approx(1 / 8)
    worse, _ = damage_test([0.5] * 5, [0.9, 0.91, 0.89, 0.9, 0.92])
    assert worse


def _evaluator(effects, noise=0.02, trial=0):
    def ev(key, seed, budget):
        if effects[key] is None:
            return None
        shared = random.Random(hash((seed, budget, trial))).gauss(0, noise)
        own = random.Random(hash((key, seed, budget, trial))).gauss(0, noise / 2)
        return 0.7 * budget + effects[key] * budget + shared + own
    return ev


def test_validator_controls_false_accepts():
    accepted = 0
    for t in range(60):
        eff = {"baseline": 0.0, **{f"c{i}": 0.0 for i in range(5)}}
        rep = SequentialValidator(_evaluator(eff, trial=t), AcceptanceConfig()).run(list(eff)[1:])
        accepted += bool(rep.accepted)
    assert accepted / 60 <= 0.15


def test_validator_finds_good_candidate_and_handles_crashes():
    eff = {"baseline": 0.0, "bad": -0.05, "crash": None, "good": 0.06, "null": 0.0}
    rep = SequentialValidator(_evaluator(eff, trial=1), AcceptanceConfig(rungs=[Rung(2, 0.5)])).run(["bad", "crash", "good", "null"])
    assert rep.accepted == ["good"]
    assert rep.records["crash"].status == "crashed"
    assert rep.seconds_used >= 0 and rep.evaluations > 0
    assert "accepted" in rep.to_markdown()


def test_baseline_crash_counts_as_improvement():
    eff = {"baseline": None, "fix": 0.0}
    rep = SequentialValidator(_evaluator(eff), AcceptanceConfig(rungs=[], confirm_seeds=5)).run(["fix"])
    assert rep.accepted == ["fix"]


def test_fixed_k_reference():
    eff = {"baseline": 0.0, "good": 0.1}
    out = fixed_k_validate(_evaluator(eff), ["good"], k=5)
    assert out["results"]["good"]["accepted"]


def test_ddmin_minimal_subset():
    need = {2, 5}
    keep, n = ddmin(8, lambda s: need <= s)
    assert set(keep) == need and n > 0
    keep, _ = ddmin(4, lambda s: False)
    assert keep == [0, 1, 2, 3]


def test_ddmin_inconclusive_is_conservative():
    need = {1}
    keep, _ = ddmin(4, lambda s: None if len(s) == 1 else need <= s)
    assert set(need) <= set(keep)


def test_minimize_patch_and_alternatives():
    names = ["lr", "momentum", "weight_decay", "eps", "gamma", "beta"]
    old = "".join(f"{n} = {i}\n" for i, n in enumerate(names))
    new = old.replace("momentum = 1", "momentum = 10").replace("eps = 3", "eps = 30").replace("beta = 5", "beta = 50")
    p = Patch.from_sources("f.py", old, new)

    def oracle(sub):
        text = sub.apply({"f.py": old})["f.py"]
        return "momentum = 10" in text or "beta = 50" in text

    res = minimize_patch(p, oracle)
    assert len(res.minimal.edits) == 1
    assert res.alternatives and len(res.alternatives[0].edits) == 1
    assert res.kind == "hyperparameter" == fix_kind(res.minimal)
    assert res.reduction > 0


def test_numeric_code_changes_are_not_hyperparameters():
    for old, new in [("att = F.softmax(att, dim=-2)\n", "att = F.softmax(att, dim=-1)\n"),
                     ("att = att.masked_fill(m == 1, x)\n", "att = att.masked_fill(m == 0, x)\n"),
                     ("loader = DataLoader(ds, batch_size=64, shuffle=False)\n", "loader = DataLoader(ds, batch_size=64, shuffle=True)\n")]:
        assert Patch.from_sources("f.py", old, new).kinds == {"code"}, old
    assert Patch.from_sources("f.py", "opt = SGD(p, lr=0.5, momentum=0.9)\n", "opt = SGD(p, lr=0.05, momentum=0.9)\n").kinds == {"hyperparameter"}
    assert Patch.from_sources("f.py", "weight_decay = 0.1\n", "weight_decay = 0.01\n").kinds == {"hyperparameter"}
