"""Standalone validation of the runtime hooks across standard architectures.

    python -m silent_bug_agent.runtime.validate            # all architectures, CPU (or CUDA if available)
    python -m silent_bug_agent.runtime.validate --device cuda --steps 8 --json report.json

For each architecture (MLP, ResNet, LSTM, GRU, Transformer encoder, GPT-style
causal decoder, autoencoder) it checks that

1. the hooks are **non-intrusive**: losses and final weights are bit-identical
   with and without hooks;
2. one ``TrainingState`` is published per step, with loss, global gradient norm
   and weight delta matching an independent measurement;
3. activation statistics cover the selected layers and gradient statistics flow
   back through them;
4. ``detach()`` removes every hook and restores ``Tensor.backward``;

and reports the per-step overhead. It then injects faults (NaN input, dead
ReLUs, saturated tanh, a huge learning rate, training in eval mode, a loss
spike) and checks the matching anomaly is raised, that a consumer thread
receives every snapshot through the stream, and that a ``torch.compile``-d
model still produces snapshots.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hooks import HookConfig, RuntimeHooks, TrainingState

ORIGINAL_BACKWARD = torch.Tensor.backward


# ------------------------------------------------------------ architectures
class SeqClassifier(nn.Module):
    def __init__(self, cell: str = "lstm", vocab: int = 50, d: int = 32, classes: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(d, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(64, classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(self.emb(tokens))
        return self.head(out[:, -1])


class BasicBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class TinyResNet(nn.Module):
    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(BasicBlock(16), BasicBlock(16))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.pool(self.layer1(self.stem(x))), 1))


class EncoderClassifier(nn.Module):
    def __init__(self, vocab: int = 50, d: int = 32, classes: int = 4) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(d, classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.emb(tokens)).mean(1))


class CausalBlock(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab: int = 40, T: int = 16, d: int = 32) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([CausalBlock(d, 4) for _ in range(2)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


@dataclass
class Arch:
    name: str
    build: Callable[[], nn.Module]
    batch: Callable[[int, torch.device], tuple[tuple[torch.Tensor, ...], torch.Tensor]]
    loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    optimizer: Callable[[Any], torch.optim.Optimizer]


def _gen(step: int) -> torch.Generator:
    return torch.Generator().manual_seed(1000 + step)


def _cls_batch(shape: tuple[int, ...], classes: int, tokens: int | None = None):
    def f(step: int, dev: torch.device):
        g = _gen(step)
        x = torch.randint(0, tokens, shape, generator=g) if tokens else torch.randn(*shape, generator=g)
        y = torch.randint(0, classes, (shape[0],), generator=g)
        return (x.to(dev),), y.to(dev)
    return f


def _lm_batch(step: int, dev: torch.device):
    seq = torch.randint(0, 40, (8, 17), generator=_gen(step)).to(dev)
    return (seq[:, :-1],), seq[:, 1:]


def _ae_batch(step: int, dev: torch.device):
    x = torch.rand(32, 32, generator=_gen(step)).to(dev)
    return (x,), x


ARCHS: dict[str, Arch] = {
    "mlp": Arch("mlp", lambda: nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 64), nn.ReLU(),
                                             nn.Linear(64, 5)),
                _cls_batch((32, 20), 5), F.cross_entropy, lambda p: torch.optim.Adam(p, lr=1e-3)),
    "resnet": Arch("resnet", TinyResNet, _cls_batch((16, 3, 16, 16), 10), F.cross_entropy,
                   lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
    "lstm": Arch("lstm", lambda: SeqClassifier("lstm"), _cls_batch((16, 12), 4, tokens=50), F.cross_entropy,
                 lambda p: torch.optim.Adam(p, lr=1e-3)),
    "gru": Arch("gru", lambda: SeqClassifier("gru"), _cls_batch((16, 12), 4, tokens=50), F.cross_entropy,
                lambda p: torch.optim.AdamW(p, lr=1e-3)),
    "transformer": Arch("transformer", EncoderClassifier, _cls_batch((16, 12), 4, tokens=50), F.cross_entropy,
                        lambda p: torch.optim.AdamW(p, lr=1e-3)),
    "gpt": Arch("gpt", TinyGPT, _lm_batch, lambda o, y: F.cross_entropy(o.reshape(-1, o.size(-1)), y.reshape(-1)),
                lambda p: torch.optim.AdamW(p, lr=3e-4)),
    "autoencoder": Arch("autoencoder", lambda: nn.Sequential(nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 32), nn.Sigmoid()),
                        _ae_batch, F.mse_loss, lambda p: torch.optim.SGD(p, lr=0.1)),
}


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------- training
def _train(arch: Arch, steps: int, dev: torch.device, *, hooks_config: HookConfig | None, with_hooks: bool,
           mutate: Callable[[nn.Module], None] | None = None, batch_fault: Callable[[int, tuple], tuple] | None = None,
           eval_mode: bool = False, optimizer: Callable[[Any], torch.optim.Optimizer] | None = None,
           compile_backend: str | None = None) -> dict[str, Any]:
    torch.manual_seed(0)
    model = arch.build().to(dev)
    if mutate is not None:
        mutate(model)
    opt = (optimizer or arch.optimizer)(model.parameters())
    run_model = torch.compile(model, backend=compile_backend) if compile_backend else model
    hooks = RuntimeHooks(run_model, opt, hooks_config).attach() if with_hooks else None
    params = [p for p in model.parameters() if p.requires_grad]
    manual: list[dict[str, float]] = []
    step_times: list[float] = []
    for step in range(steps):
        t0 = time.perf_counter()
        run_model.train(not eval_mode)
        x, y = arch.batch(step, dev)
        if batch_fault is not None:
            x = batch_fault(step, x)
        opt.zero_grad(set_to_none=True)
        loss = arch.loss(run_model(*x), y)
        loss.backward()
        gsq = sum(float(p.grad.double().pow(2).sum()) for p in params if p.grad is not None)
        before = [p.detach().clone() for p in params]
        opt.step()
        dsq = sum(float((p.detach().double() - b.double()).pow(2).sum()) for p, b in zip(params, before))
        manual.append({"loss": float(loss.detach()), "grad_norm": math.sqrt(gsq), "delta_norm": math.sqrt(dsq)})
        if dev.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
    timed = sorted(step_times[1:] or step_times)
    seconds = timed[len(timed) // 2]  # median step time, first (warm-up) step excluded
    states: list[TrainingState] = []
    detach_clean = True
    n_layers = 0
    if hooks is not None:
        hooks.flush()
        states = hooks.stream.drain()
        n_layers = len(hooks.layers)
        hooks.detach()
        detach_clean = (
            all(len(m._forward_hooks) == 0 for m in model.modules())
            and len(opt._optimizer_step_pre_hooks) == 0 and len(opt._optimizer_step_post_hooks) == 0
            and torch.Tensor.backward is ORIGINAL_BACKWARD
        )
    return {"model": model, "states": states, "manual": manual, "seconds": seconds, "detach_clean": detach_clean,
            "n_layers": n_layers, "params": [p.detach().clone() for p in model.parameters()]}


def _close(a: float | None, b: float, rtol: float) -> bool:
    return a is not None and math.isfinite(a) and abs(a - b) <= rtol * max(1.0, abs(b))


def validate_architecture(name: str, steps: int = 5, device: torch.device | None = None,
                          config: HookConfig | None = None) -> dict[str, Any]:
    arch = ARCHS[name]
    dev = device or default_device()
    cfg = config or HookConfig(histogram_bins=8)
    base = _train(arch, steps, dev, hooks_config=None, with_hooks=False)
    hooked = _train(arch, steps, dev, hooks_config=cfg, with_hooks=True)
    states, manual = hooked["states"], hooked["manual"]
    non_intrusive = (
        [m["loss"] for m in base["manual"]] == [m["loss"] for m in manual]
        and all(torch.equal(a, b) for a, b in zip(base["params"], hooked["params"]))
    )
    one_per_step = len(states) == steps and [s.step for s in states] == list(range(steps))
    loss_ok = one_per_step and all(_close(s.loss, m["loss"], 1e-5) for s, m in zip(states, manual))
    grad_ok = one_per_step and all(_close(s.grad_norm, m["grad_norm"], 1e-3) for s, m in zip(states, manual))
    delta_ok = one_per_step and all(_close(s.delta_norm, m["delta_norm"], 1e-3) for s, m in zip(states, manual))
    act_cov = min((sum(l.act_mean is not None for l in s.layers.values()) / max(1, hooked["n_layers"]) for s in states), default=0.0)
    grad_cov = min((sum(l.grad_norm is not None for l in s.layers.values()) for s in states), default=0)
    hist_ok = all(any(l.histogram for l in s.layers.values()) for s in states) if cfg.histogram_bins else True
    clean = not any(a.code.startswith("nonfinite") for s in states for a in s.anomalies)
    result = {
        "arch": name, "device": str(dev), "steps": steps, "layers": hooked["n_layers"], "snapshots": len(states),
        "non_intrusive": non_intrusive, "loss_matches": loss_ok, "grad_norm_matches": grad_ok,
        "delta_matches": delta_ok, "activation_coverage": round(act_cov, 3), "layers_with_grad_stats": grad_cov,
        "histograms": hist_ok, "no_false_nonfinite": clean, "detach_clean": hooked["detach_clean"],
        "overhead": round(hooked["seconds"] / max(base["seconds"], 1e-9), 2),  # median step-time ratio
    }
    result["passed"] = all([non_intrusive, one_per_step, loss_ok, grad_ok, delta_ok, act_cov >= 0.8, grad_cov >= 1,
                            hist_ok, clean, hooked["detach_clean"]])
    return result


# ------------------------------------------------------------------ faults
def _codes(states: list[TrainingState]) -> set[str]:
    return {a.code for s in states for a in s.anomalies}


def validate_fault_signals(device: torch.device | None = None) -> dict[str, bool]:
    dev = device or default_device()
    mlp, ae = ARCHS["mlp"], ARCHS["autoencoder"]
    cfg = HookConfig()
    out: dict[str, bool] = {}

    def nan_input(step: int, x: tuple) -> tuple:
        if step == 2:
            x = (x[0].clone(),)
            x[0][0, 0] = float("nan")
        return x

    r = _train(mlp, 4, dev, hooks_config=cfg, with_hooks=True, batch_fault=nan_input)
    out["nan_input -> nonfinite_activation"] = {"nonfinite_activation", "nonfinite_loss"} <= _codes(r["states"][2:3])

    def dead(model: nn.Module) -> None:
        with torch.no_grad():
            model[0].bias.fill_(-1e3)

    r = _train(mlp, 3, dev, hooks_config=cfg, with_hooks=True, mutate=dead)
    out["negative bias -> dead_units"] = "dead_units" in _codes(r["states"])

    def saturate(model: nn.Module) -> None:
        with torch.no_grad():
            model[0].weight.mul_(100.0)

    r = _train(ae, 3, dev, hooks_config=cfg, with_hooks=True, mutate=saturate)
    out["huge weights -> saturated_units"] = "saturated_units" in _codes(r["states"])

    r = _train(mlp, 3, dev, hooks_config=cfg, with_hooks=True, optimizer=lambda p: torch.optim.SGD(p, lr=1e4))
    out["lr x1e4 -> update_too_large"] = "update_too_large" in _codes(r["states"])

    r = _train(mlp, 3, dev, hooks_config=cfg, with_hooks=True, eval_mode=True)
    out["model.eval() -> eval_mode_training"] = "eval_mode_training" in _codes(r["states"])

    def spike(step: int, x: tuple) -> tuple:
        return (x[0] * 1e3,) if step == 25 else x

    r = _train(mlp, 28, dev, hooks_config=cfg, with_hooks=True, batch_fault=spike)
    out["scaled batch -> loss_spike"] = "loss_spike" in _codes(r["states"][25:26])
    out["clean run -> no anomalies"] = not _codes(_train(mlp, 28, dev, hooks_config=cfg, with_hooks=True)["states"])
    return out


def validate_stream_consumer(device: torch.device | None = None, steps: int = 12) -> bool:
    """A separate agent thread receives every snapshot while training runs."""
    dev = device or default_device()
    arch = ARCHS["mlp"]
    torch.manual_seed(0)
    model = arch.build().to(dev)
    opt = arch.optimizer(model.parameters())
    hooks = RuntimeHooks(model, opt, HookConfig(max_pending=1)).attach()
    got: list[int] = []

    def agent() -> None:
        for state in hooks.stream:
            got.append(state.step)

    t = threading.Thread(target=agent, daemon=True)
    t.start()
    for step in range(steps):
        x, y = arch.batch(step, dev)
        opt.zero_grad()
        arch.loss(model(*x), y).backward()
        opt.step()
    hooks.detach()
    hooks.stream.close()
    t.join(timeout=10)
    return got == list(range(steps))


def validate_compiled(device: torch.device | None = None, backend: str = "aot_eager") -> bool:
    """A torch.compile-d model trains unchanged and still yields loss/gradient/weight snapshots."""
    dev = device or default_device()
    try:
        r = _train(ARCHS["mlp"], 4, dev, hooks_config=HookConfig(), with_hooks=True, compile_backend=backend)
    except Exception:
        return False
    s = r["states"]
    return len(s) == 4 and all(_close(st.grad_norm, m["grad_norm"], 1e-3) and _close(st.loss, m["loss"], 1e-5)
                               for st, m in zip(s, r["manual"]))


# --------------------------------------------------------------------- cli
def run_all(device: torch.device | None = None, steps: int = 5, archs: list[str] | None = None) -> dict[str, Any]:
    dev = device or default_device()
    arch_results = [validate_architecture(a, steps, dev) for a in (archs or list(ARCHS))]
    faults = validate_fault_signals(dev)
    extra = {"stream consumer thread": validate_stream_consumer(dev), "torch.compile (aot_eager)": validate_compiled(dev)}
    passed = all(r["passed"] for r in arch_results) and all(faults.values()) and all(extra.values())
    return {"device": str(dev), "torch": torch.__version__, "architectures": arch_results, "faults": faults,
            "extra": extra, "passed": passed}


def _print(report: dict[str, Any]) -> None:
    cols = ["arch", "layers", "snapshots", "non_intrusive", "loss_matches", "grad_norm_matches", "delta_matches",
            "activation_coverage", "layers_with_grad_stats", "detach_clean", "overhead", "passed"]
    print(f"device={report['device']} torch={report['torch']}")
    print(" | ".join(cols))
    for r in report["architectures"]:
        print(" | ".join(str(r[c]) for c in cols))
    print("\nfault signals:")
    for k, v in {**report["faults"], **report["extra"]}.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\nOVERALL: {'PASS' if report['passed'] else 'FAIL'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate silent_bug_agent runtime hooks across standard architectures.")
    ap.add_argument("--device", default=None, help="cpu | cuda | cuda:1 (default: cuda if available)")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--arch", nargs="*", choices=list(ARCHS))
    ap.add_argument("--json", help="write the full report here")
    a = ap.parse_args(argv)
    report = run_all(torch.device(a.device) if a.device else None, a.steps, a.arch)
    _print(report)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=2, default=str)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
