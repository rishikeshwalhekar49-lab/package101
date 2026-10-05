import json
from types import SimpleNamespace

import pytest

from trainmedic.detect.config_check import check_config_file, check_deepspeed_config, check_hf_training_args, fix_config_file
from trainmedic.diagnose import diagnose
from trainmedic.llm.backend import AnthropicBackend, LLMError, ScriptedBackend, get_backend, parse_json
from trainmedic.llm.tasks import DIAGNOSIS_SCHEMA, REPAIR_SCHEMA, llm_diagnose, llm_repair, merge_diagnoses, naive_llm_repair
from trainmedic.memory import ExperienceMemory
from trainmedic.repair.abstention import AbstentionPolicy
from trainmedic.repair.patch import Patch
from trainmedic.types import Evidence, Severity

SRC = "import torch\nfor x, y in loader:\n    loss = f(x, y)\n    loss.backward()\n    opt.step()\n"


def ev(id, source, rule, cause, conf=0.9, sev=Severity.ERROR, **kw):
    return Evidence(id=id, source=source, rule=rule, message=f"{rule} msg", severity=sev, confidence=conf, cause=cause, **kw)


def test_fusion_corroboration_and_attribution():
    evidence = [ev("S1", "static", "TM001", "missing_zero_grad", 0.8, file="t.py", line=5),
                ev("V1", "invariant", "grads_cleared_between_steps", "missing_zero_grad", 0.8)]
    d = diagnose(evidence)
    assert d.bug_found and d.top.cause_id == "missing_zero_grad"
    assert d.top.confidence > 0.95 and d.top.line == 5
    assert d.attribution == "user_code"
    d2 = diagnose([ev("C1", "compile", "eager_compiled_mismatch", "compiler_mismatch", 0.85)])
    assert d2.attribution == "compiler"
    d3 = diagnose([ev("K1", "config", "DS002", "framework_config", 0.95)])
    assert d3.attribution == "framework_config"
    assert not diagnose([ev("S1", "static", "TM030", "unseeded", 0.3, Severity.INFO)]).bug_found


def test_abstention_gates():
    pol = AbstentionPolicy()
    weak = [ev("S1", "static", "TM012", "no_shuffle", 0.4, Severity.WARNING)]
    assert pol.gate_before_repair(weak, None)[0] is True
    strong = [ev("S1", "static", "TM001", "missing_zero_grad")]
    assert pol.gate_before_repair(strong, diagnose(strong))[0] is False
    p = Patch.from_sources("t.py", SRC, SRC.replace("    opt.step()\n", "    opt.step()\n    opt.zero_grad()\n"), evidence_ids=["S1"])
    assert pol.gate_patch(p, strong)[0]
    ungrounded = Patch.from_sources("t.py", SRC, SRC + "x = 1\n", evidence_ids=[])
    assert not pol.gate_patch(ungrounded, strong)[0]
    lr_change = Patch.from_sources("t.py", "opt = SGD(p, lr=0.1)\n", "opt = SGD(p, lr=0.01)\n", evidence_ids=["S1"])
    ok, why = pol.gate_patch(lr_change, strong)
    assert not ok and "changes_lr" in why


def test_llm_diagnosis_merge_penalises_ungrounded():
    evidence = [ev("S1", "static", "TM001", "missing_zero_grad", 0.9)]
    backend = ScriptedBackend([{
        "bug_found": True, "summary": "x", "attribution": "user_code",
        "causes": [
            {"cause_id": "missing_zero_grad", "title": "t", "description": "d", "confidence": 0.9, "evidence_ids": ["S1"], "file": "t.py", "line": 5},
            {"cause_id": "lr_scale", "title": "lr", "description": "guess", "confidence": 0.9, "evidence_ids": [], "file": "", "line": 0},
        ],
    }])
    ld = llm_diagnose(backend, {"t.py": SRC}, evidence)
    merged = merge_diagnoses(diagnose(evidence), ld, evidence)
    by = {c.cause_id: c for c in merged.causes}
    assert by["lr_scale"].confidence < 0.5 and by["lr_scale"].title.startswith("[ungrounded]")
    assert merged.bug_found
    assert "S1" in backend.calls[0]["prompt"] and backend.calls[0]["schema"] is DIAGNOSIS_SCHEMA


def test_llm_repair_parses_candidates_and_abstain():
    backend = ScriptedBackend([
        {"abstain": False, "reason": "", "candidates": [
            {"description": "add zero_grad", "evidence_ids": ["S1"],
             "edits": [{"file": "t.py", "search": "    opt.step()", "replace": "    opt.step()\n    opt.zero_grad()"}]},
            {"description": "broken", "evidence_ids": ["S1"], "edits": [{"file": "t.py", "search": "nope", "replace": "x"}]},
        ]},
        {"abstain": True, "reason": "code is fine", "candidates": []},
    ])
    abstained, _, patches, errors = llm_repair(backend, {"t.py": SRC}, [], None)
    assert not abstained and len(patches) == 1 and len(errors) == 1
    assert "opt.zero_grad()" in patches[0].apply({"t.py": SRC})["t.py"]
    assert backend.calls[0]["schema"] is REPAIR_SCHEMA
    abstained, reason, patches, _ = llm_repair(backend, {"t.py": SRC}, [], None)
    assert abstained and not patches and reason == "code is fine"
    naive = ScriptedBackend([{"explanation": "e", "edits": [{"file": "t.py", "search": "opt.step()", "replace": "opt.step()  # x"}]}])
    patches, _ = naive_llm_repair(naive, {"t.py": SRC})
    assert patches and patches[0].origin.startswith("naive-llm")


class _FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


def test_anthropic_backend_request_shape():
    calls = {}
    msg = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text='{"ok": true}')],
                          usage=SimpleNamespace(input_tokens=1, output_tokens=2, cache_read_input_tokens=0))

    def stream(**kw):
        calls.update(kw)
        return _FakeStream(msg)

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    b = AnthropicBackend(client=client)
    pytest.importorskip("anthropic")
    assert b.complete_json("sys", "prompt", {"type": "object"}) == {"ok": True}
    assert calls["model"] == "claude-opus-5-5"
    assert calls["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in calls["betas"]
    assert calls["output_config"]["format"]["type"] == "json_schema"
    assert calls["thinking"] == {"type": "adaptive"}
    msg.stop_reason = "refusal"
    msg.stop_details = SimpleNamespace(category="cyber")
    with pytest.raises(LLMError):
        b.complete_json("sys", "prompt", {})


def test_backend_specs_and_json_parsing():
    assert get_backend("none") is None
    assert get_backend("cmd:echo {}").name == "cmd:echo {}"
    assert get_backend("anthropic:claude-sonnet-5-5").model == "claude-sonnet-5-5"
    with pytest.raises(ValueError):
        get_backend("openai")
    assert parse_json('text ```json\n{"a": 1}\n``` more') == {"a": 1}
    assert get_backend("cmd:cat").complete_json("s", "p", {"x": 1})["schema"] == {"x": 1}


def test_memory_precision_gate(tmp_path):
    mem = ExperienceMemory(tmp_path / "m.jsonl")
    evidence = [ev("S1", "static", "TM001", "missing_zero_grad")]
    new = SRC.replace("    opt.step()\n", "    opt.step()\n    opt.zero_grad()\n")
    p = Patch.from_sources("t.py", SRC, new, description="add zero_grad", evidence_ids=["S1"])
    mem.record(evidence, ["missing_zero_grad"], p, {"t.py": SRC}, "accepted")
    assert mem.suggest(evidence, {"t.py": SRC}) == []  # support 1 < 2
    mem.record(evidence, ["missing_zero_grad"], p, {"t.py": SRC}, "accepted")
    sugg = mem.suggest(evidence, {"t.py": SRC})
    assert sugg and sugg[0].apply({"t.py": SRC})["t.py"] == new
    mem.record(evidence, ["missing_zero_grad"], p, {"t.py": SRC}, "damage")
    assert mem.suggest(evidence, {"t.py": SRC}) == []  # 2/3 < 0.8 bar after damage
    assert mem.hints(evidence)
    with pytest.raises(ValueError):
        mem.record(evidence, [], p, {"t.py": SRC}, "great")


def test_config_checks(tmp_path):
    ds = {"train_batch_size": 64, "train_micro_batch_size_per_gpu": 4, "gradient_accumulation_steps": 2,
          "fp16": {"enabled": True}, "bf16": {"enabled": True}, "zero_optimization": {"stage": 2, "offload_param": {"device": "cpu"}}}
    rules = {e.rule for e in check_deepspeed_config(ds, world_size=4)}
    assert {"DS001", "DS002", "DS003"} <= rules
    f = tmp_path / "ds.json"
    f.write_text(json.dumps(ds))
    kind, evs = check_config_file(f, 4)
    assert kind == "deepspeed" and evs
    out = tmp_path / "fixed.json"
    fix_config_file(f, out, 4)
    assert not {e.rule for e in check_deepspeed_config(json.loads(out.read_text()), 4)} & {"DS001", "DS002", "DS003"}
    hf = {"output_dir": "o", "load_best_model_at_end": True, "metric_for_best_model": "eval_loss", "greater_is_better": True}
    assert any(e.rule == "HF003" for e in check_hf_training_args(hf))


def test_lr_candidates_only_when_lr_evidence(write_script):
    from trainmedic.repair.agent import RepairConfig, RepairSession

    src = "import torch\nmodel = torch.nn.Linear(2, 2)\nopt = torch.optim.Adam(model.parameters(), lr=0.3)\n"
    sess = RepairSession(str(write_script(src)), config=RepairConfig(mode="rules"))
    assert sess._lr_candidates([ev("S1", "static", "TM001", "missing_zero_grad")]) == []
    lr_ev = [ev("V1", "invariant", "update_ratio_sane", "lr_scale", 0.65, Severity.WARNING)]
    cands = sess._lr_candidates(lr_ev)
    texts = [c.apply(sess.sources)["train.py"] for c in cands]
    assert any("lr=0.03" in t for t in texts) and any("lr=0.003" in t for t in texts)
    assert all(c.kinds == {"hyperparameter"} and AbstentionPolicy().gate_patch(c, lr_ev)[0] for c in cands)
