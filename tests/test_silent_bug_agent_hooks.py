"""Tests for silent_bug_agent.runtime.hooks (non-intrusive runtime perception)."""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from silent_bug_agent.runtime import HookConfig, RuntimeHooks, SnapshotStream, TensorStatsExtractor, TrainingState
from silent_bug_agent.runtime.validate import (
    ARCHS,
    validate_architecture,
    validate_compiled,
    validate_fault_signals,
    validate_stream_consumer,
)

CPU = torch.device("cpu")


@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_architecture_validation(arch):
    r = validate_architecture(arch, steps=3, device=CPU)
    assert r["passed"], r


def test_fault_signals():
    res = validate_fault_signals(CPU)
    assert all(res.values()), res


def test_stream_consumer_thread_and_compile():
    assert validate_stream_consumer(CPU)
    assert validate_compiled(CPU)


def _train(model, hooks, steps, manual_update=False, lr=0.1):
    opt = None if manual_update else torch.optim.SGD(model.parameters(), lr=lr)
    for step in range(steps):
        x = torch.randn(8, 4, generator=torch.Generator().manual_seed(step))
        loss = model(x).pow(2).mean()
        if opt:
            opt.zero_grad()
        else:
            model.zero_grad()
        loss.backward()
        if opt:
            opt.step()
        else:
            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * p.grad
            hooks.end_step()


def test_every_n_steps():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    hooks = RuntimeHooks(model, opt, HookConfig(every_n_steps=2)).attach()
    for step in range(6):
        opt.zero_grad()
        model(torch.randn(8, 4)).pow(2).mean().backward()
        opt.step()
    hooks.detach()
    assert [s.step for s in hooks.stream.drain()] == [0, 2, 4]


def test_manual_updates_with_end_step():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    hooks = RuntimeHooks(model).attach()
    _train(model, hooks, 3, manual_update=True)
    hooks.detach()
    states = hooks.stream.drain()
    assert len(states) == 3 and all(s.loss is not None and s.grad_norm and s.layers for s in states)
    assert all(isinstance(s, TrainingState) and "step" in s.to_dict() for s in states)
    assert "loss=" in states[0].summary()


def test_stream_drops_oldest_when_full():
    stream = SnapshotStream(capacity=2)
    seen = []
    stream.subscribe(lambda s: seen.append(s.step))
    for i in range(5):
        stream.publish(TrainingState(i, 0.0, True, 1.0, None, None, False, 1, [], None, None, None, None))
    assert stream.dropped == 3 and [s.step for s in stream.drain()] == [3, 4]
    assert seen == [0, 1, 2, 3, 4] and stream.latest.step == 4
    stream.close()
    assert stream.get(timeout=0.1) is None


def test_extractor_sampling_and_dtypes():
    ex = TensorStatsExtractor(max_elems=1000)
    x = torch.randn(100_000) * 3 + 1
    mean, std = ex.activation(x)[:2].tolist()
    assert abs(mean - 1) < 0.3 and abs(std - 3) < 0.3
    stats = ex.activation(torch.tensor([1.0, float("nan"), 0.0], dtype=torch.bfloat16)).tolist()
    assert stats[8] == 1.0 and stats[5] == pytest.approx(0.5)
    assert math.isclose(ex.gradient(torch.ones(4))[0].item(), 2.0)


def test_bfloat16_model_runs():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 2)).to(torch.bfloat16)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    with RuntimeHooks(model, opt) as hooks:
        for _ in range(2):
            opt.zero_grad()
            F.cross_entropy(model(torch.randn(8, 4, dtype=torch.bfloat16)).float(), torch.randint(0, 2, (8,))).backward()
            opt.step()
    states = hooks.stream.drain()
    assert len(states) == 2 and all(s.grad_norm and s.delta_norm for s in states)


def test_nan_outside_sample_is_still_detected():
    ex = TensorStatsExtractor(max_elems=100)
    x = torch.zeros(10_000)
    x[1] = float("nan")  # stride 100 never samples index 1
    assert ex.activation(x)[8].item() >= 1
    assert ex.gradient(x)[3].item() >= 1
