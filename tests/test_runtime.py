"""In-process runtime tests: tracer + invariants + probes, Guardian, compile diff, watch()."""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import trainmedic as tm
from trainmedic.detect import probes
from trainmedic.detect.compile_diff import compile_diff
from trainmedic.detect.invariants import InvariantSet
from trainmedic.detect.trace import Trace
from trainmedic.detect.tracer import Tracer
from trainmedic.remediate import Action, Guardian, SimulatedDataParallel, run_guarded


def _data(sorted_labels=False, n=512):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(n, 10, generator=g)
    y = (X[:, 0] > 0).long() + 2 * (X[:, 1] > 0).long()
    if sorted_labels:
        idx = y.argsort()
        X, y = X[idx], y[idx]
    return TensorDataset(X, y)


class Net(nn.Module):
    def __init__(self, softmax=False, mix=False, fdrop=False):
        super().__init__()
        self.fc1, self.bn, self.do, self.fc2 = nn.Linear(10, 32), nn.BatchNorm1d(32), nn.Dropout(0.1), nn.Linear(32, 4)
        self.softmax, self.mix, self.fdrop = softmax, mix, fdrop

    def forward(self, x):
        h = self.do(F.relu(self.bn(self.fc1(x))))
        if self.fdrop:
            h = F.dropout(h, 0.3)
        if self.mix:
            h = h.t().reshape(h.size(0), -1)
        o = self.fc2(h)
        return F.softmax(o, 1) if self.softmax else o


def run_loop(bug=None, steps=30):
    torch.manual_seed(0)
    ds = _data(sorted_labels=bug == "noshuffle")
    tl = DataLoader(ds, batch_size=32, shuffle=bug != "noshuffle")
    vl = DataLoader(ds if bug == "leak" else _data(n=256), batch_size=64)
    m = Net(softmax=bug == "softmax", mix=bug == "mix", fdrop=bug == "fdrop")
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    accum = 2 if bug in ("accum", "accum_ok") else 1
    with Tracer(max_steps=steps) as tr:
        for _ in range(100):
            if bug != "notrain":
                m.train()
            for i, (x, y) in enumerate(tl):
                if bug != "nozero" and i % accum == 0 and bug != "zero_late":
                    opt.zero_grad()
                loss = F.cross_entropy(m(x), y)
                if bug == "accum_ok":
                    loss = loss / accum
                loss.backward()
                if bug == "zero_late":
                    opt.zero_grad()
                if (i + 1) % accum == 0:
                    opt.step()
            if bug != "noeval":
                m.eval()
            with torch.no_grad():
                for x, _ in vl:
                    m(x)
    return tr.trace()


EXPECT = {
    "nozero": "grads_cleared_between_steps",
    "zero_late": "gradients_present_at_step",
    "softmax": "cross_entropy_gets_logits",
    "noeval": "eval_mode_during_evaluation",
    "notrain": "train_mode_during_training",
    "accum": "grad_accumulation_scaled",
    "noshuffle": "training_data_shuffled",
    "leak": "no_train_eval_overlap",
    "mix": "batch_independence",
    "fdrop": "eval_mode_deterministic",
}


def _rules(trace, conf=0.5):
    return {e.rule for e in InvariantSet.default().check(trace) if e.confidence >= conf}


def test_clean_loop_has_no_violations():
    assert _rules(run_loop(None)) == set()
    assert _rules(run_loop("accum_ok")) == set()


@pytest.mark.parametrize("bug", sorted(EXPECT))
def test_invariant_detects_bug(bug):
    assert EXPECT[bug] in _rules(run_loop(bug))


def test_trace_save_load_and_inference(tmp_path):
    tr = run_loop(None)
    p = tmp_path / "t.jsonl"
    tr.save(p)
    back = Trace.load(p)
    assert back.n_steps == tr.n_steps == 30 and back.meta.get("stopped_early")
    inv = InvariantSet.infer([back])
    names = set(inv.names())
    assert {"grads_cleared_between_steps", "update_ratio_sane", "loss_progress"} <= names
    inv.save(tmp_path / "inv.json")
    again = InvariantSet.load(tmp_path / "inv.json")
    assert len(again) == len(inv)
    assert not {e.rule for e in again.check(back) if e.confidence >= 0.5}
    assert "grads_cleared_between_steps" in {e.rule for e in again.check(run_loop("nozero"))}


class TinyAttn(nn.Module):
    def __init__(self, causal=True, V=11, T=8, d=16):
        super().__init__()
        self.emb, self.pos, self.qkv, self.out = nn.Embedding(V, d), nn.Embedding(T, d), nn.Linear(d, 3 * d), nn.Linear(d, V)
        self.causal = causal
        self.register_buffer("mask", torch.tril(torch.ones(T, T)))

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.size(1)))
        q, k, v = self.qkv(x).chunk(3, -1)
        att = (q @ k.transpose(-2, -1)) / 4.0
        if self.causal:
            L = idx.size(1)
            att = att.masked_fill(self.mask[:L, :L] == 0, float("-inf"))
        return self.out(F.softmax(att, -1) @ v)


def test_probes_causality_and_independence():
    idx = torch.randint(0, 11, (4, 8))
    assert probes.causality(TinyAttn(True).eval(), (idx,))["passed"]
    assert not probes.causality(TinyAttn(False).eval(), (idx,))["passed"]
    assert probes.batch_independence(TinyAttn(True).eval(), (idx,))["passed"]
    m = Net(fdrop=True).eval()
    assert not probes.eval_determinism(m, (torch.randn(4, 10),))["passed"]
    ok = probes.overfit_one_batch(nn.Linear(10, 4), (torch.randn(8, 10), torch.randint(0, 4, (8,))), F.cross_entropy,
                                  lambda p: torch.optim.Adam(p, lr=0.05), steps=150)
    assert ok["passed"]
    assert probes.initial_loss_check(math.log(10), 10)["passed"]


def test_causality_invariant_from_tracer():
    m = TinyAttn(causal=False)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    with Tracer(max_steps=3) as tr:
        for _ in range(5):
            idx = torch.randint(0, 11, (8, 8))
            opt.zero_grad()
            F.cross_entropy(m(idx[:, :-1]).reshape(-1, 11), idx[:, 1:].reshape(-1)).backward()
            opt.step()
    assert "causality" in _rules(tr.trace())


def test_watch_context_reports_and_raises():
    model = nn.Sequential(nn.Linear(10, 8), nn.Linear(8, 4))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    X, y = torch.randn(64, 10), torch.randint(0, 4, (64,))
    with tm.watch(model, opt, max_steps=5, report=False) as w:
        for i in range(10):
            F.cross_entropy(model(X), y).backward()
            opt.step()
    assert "grads_cleared_between_steps" in {e.rule for e in w.evidence}
    from trainmedic.api import SilentBugError

    with pytest.raises(SilentBugError):
        with tm.watch(model, opt, max_steps=5, report=False, raise_on_error=True):
            for i in range(10):
                F.cross_entropy(model(X), y).backward()
                opt.step()


# ------------------------------------------------------------------ Guardian
def test_guardian_rolls_back_and_skips_bad_batches():
    torch.manual_seed(0)
    X = torch.randn(3200, 10)
    Y = X @ torch.randn(10, 1)

    def get_batch(i):
        x, y = X[(i * 32) % 3200:(i * 32) % 3200 + 32], Y[(i * 32) % 3200:(i * 32) % 3200 + 32]
        return (x, y * 1e3) if i == 60 else (x, y)

    model = nn.Linear(10, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    g = Guardian(model, opt, snapshot_every=10)
    out = run_guarded(g, get_batch, lambda b: F.mse_loss(model(b[0]), b[1]), total_steps=100)
    assert out["rollbacks"] >= 1 and 60 in out["skipped_batches"]
    assert out["steps"] == 100


def test_guardian_skips_nonfinite_grad_and_alerts_on_eval_mode():
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1), nn.Linear(4, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    g = Guardian(model, opt)
    for p in model.parameters():
        p.grad = torch.full_like(p, float("nan"))
    assert g.pre_step(0) is False
    model.eval()
    loss = F.mse_loss(model(torch.randn(8, 4)), torch.randn(8, 1))
    loss.backward()
    assert g.pre_step(1)
    opt.step()
    d = g.post_step(1, loss)
    assert d.action == Action.ALERT and d.cause == "missing_train_mode"


def test_guardian_resyncs_diverged_replicas():
    sim = SimulatedDataParallel(nn.Linear(4, 1), world_size=3)
    sim.inject(1, "skip_allreduce", at_step=2)
    g = Guardian(sim.replicas[0], sim.optimizers[0], replicas=sim.replicas, replica_optimizers=sim.optimizers,
                 snapshot_every=0, check_code_invariants=False)
    actions = []
    for step in range(5):
        shards = [(torch.randn(8, 4), torch.randn(8, 1)) for _ in range(3)]
        sim.step(shards, lambda m, b: F.mse_loss(m(b[0]), b[1]))
        actions.append(g.post_step(step, 1.0).action)
    assert Action.RESYNC in actions and sim.consistent()


# ------------------------------------------------------------- compile diff
def test_compile_diff_detects_and_localizes():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Sequential(nn.Linear(6, 8), nn.GELU())
            self.b = nn.Linear(8, 3)

        def forward(self, x):
            return self.b(self.a(x))

    x = torch.randn(4, 6)
    assert compile_diff(M(), (x,), backend="aot_eager").matches

    def faulty(gm, example_inputs):
        has_gelu = any("gelu" in str(n.target) for n in gm.graph.nodes)

        def run(*args):
            out = gm(*args)
            if has_gelu:
                return [o * 1.05 if isinstance(o, torch.Tensor) else o for o in out] if isinstance(out, (list, tuple)) else out * 1.05
            return out

        return run

    rep = compile_diff(M(), (x,), backend=faulty)
    assert not rep.matches and rep.culprits == ["a.1"]
    assert rep.to_evidence()[0].cause == "compiler_mismatch"


def test_trace_diff_finds_first_divergence():
    from trainmedic.detect.trace_diff import diff_traces

    ref, bad = run_loop(None), run_loop("nozero")
    assert diff_traces(ref, run_loop(None)).first is None
    d = diff_traces(ref, bad)
    assert d.first is not None and d.first.step <= 1
    assert d.to_evidence()[0].rule == "reference_divergence"
