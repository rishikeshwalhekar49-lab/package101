"""Non-intrusive runtime hooks: the sensory layer of the Runtime Monitor Agent.

``RuntimeHooks`` attaches PyTorch hooks to an existing model and optimizer --
no change to the model's source code -- and turns every monitored training
step into one :class:`TrainingState` snapshot:

* **forward hooks** on each selected layer -> activation distributions
  (mean, std, min, max, mean |x|, fraction exactly zero, fraction saturated,
  norm, non-finite count, optional histogram);
* **tensor (backward) hooks** on each layer's output -> gradient flowing back
  through that layer (norm, mean |g|, max |g|, non-finite count);
* **optimizer step pre/post hooks** -> per-parameter gradient norms, weight
  norms and weight deltas (||W_after - W_before||), learning rates;
* a **backward() wrapper** -> loss values, with EMA z-score spike tracking
  (adapted from ``trainmedic.detect.numerics``).

Design for low overhead (adapted from ``trainmedic.detect.tracer``):

* Statistics are computed on the tensor's own device (GPU or CPU) from views of
  the activations -- activations are never copied or moved. Tensors larger than
  ``max_elems`` (default 65,536) are sub-sampled with a strided *view*; one extra
  reduction over the full tensor still catches any NaN/Inf outside the sample.
  Norms of sampled tensors are estimates (scaled by the stride).
* Nothing synchronises with the GPU inside a hook. At the end of a monitored
  step all statistics are packed into one small device vector and copied to the
  host once, asynchronously, into a reusable pinned buffer; the snapshot is
  built when that copy has landed (``max_pending`` steps of lag at most).
* The only extra device copy is a strided sample of each weight tensor taken
  just before ``optimizer.step()`` into a buffer allocated once, so the weight
  delta can be measured afterwards.
* Snapshots are published to a :class:`SnapshotStream`, a bounded ring buffer
  that never blocks training: an agent thread can ``get()`` / iterate /
  ``subscribe()``; when the consumer falls behind, the oldest snapshots are
  dropped and counted.

Hooks inside a ``torch.compile``-d region are skipped while it is being traced
(the optimizer and loss signals keep working); see ``validate.py`` for the
standalone validation across MLP, ResNet, LSTM, GRU, Transformer, GPT-style
and autoencoder architectures.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn

from trainmedic.detect.numerics import EmaZScore

ACT_FIELDS = ("mean", "std", "min", "max", "absmean", "frac_zero", "frac_saturated", "norm", "nonfinite")
GRAD_FIELDS = ("norm", "absmean", "absmax", "nonfinite")
PARAM_FIELDS = ("grad_norm", "grad_absmax", "weight_norm", "delta_norm")

_SKIP_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.Identity, nn.Flatten,
               nn.Unflatten, nn.modules.loss._Loss)
_KIND_TABLE: list[tuple[type, str]] = [
    (nn.MultiheadAttention, "attention"),
    (nn.RNNBase, "rnn"),
    (nn.modules.conv._ConvNd, "conv"),
    (nn.Linear, "linear"),
    (nn.Embedding, "embedding"),
    (nn.modules.batchnorm._BatchNorm, "norm"),
    (nn.LayerNorm, "norm"),
    (nn.GroupNorm, "norm"),
    (nn.ReLU, "relu"),
    (nn.ReLU6, "relu"),
    (nn.Tanh, "tanh"),
    (nn.Hardtanh, "tanh"),
    (nn.Sigmoid, "sigmoid"),
    (nn.GELU, "gelu"),
    (nn.SiLU, "silu"),
    (nn.Softmax, "softmax"),
    (nn.modules.pooling._MaxPoolNd, "pool"),
    (nn.modules.pooling._AdaptiveAvgPoolNd, "pool"),
]
_SATURATING = {"tanh", "sigmoid"}
_DEAD_PRONE = {"relu"}


def _compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


def layer_kind(module: nn.Module) -> str:
    for typ, kind in _KIND_TABLE:
        if isinstance(module, typ):
            return kind
    return "other"


def _first_float_tensor(obj: Any) -> torch.Tensor | None:
    if isinstance(obj, torch.Tensor):
        return obj if obj.is_floating_point() and obj.numel() > 0 else None
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, (list, tuple)):
        for o in obj:
            t = _first_float_tensor(o)
            if t is not None:
                return t
    return None


# =============================================================== statistics
class TensorStatsExtractor:
    """Batched tensor statistics computed on the tensor's own device.

    Every method returns a small device tensor and never calls ``.item()`` /
    ``float()``, so a hook never forces a GPU synchronisation."""

    def __init__(self, max_elems: int = 1 << 16, histogram_bins: int = 0) -> None:
        self.max_elems = max_elems
        self.histogram_bins = histogram_bins

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """A flat strided *view* of at most ``max_elems`` elements, and the stride used."""
        flat = x.detach()
        flat = flat.reshape(-1) if flat.is_contiguous() else flat.flatten()
        stride = max(1, math.ceil(flat.numel() / self.max_elems))
        return (flat[::stride] if stride > 1 else flat), stride

    @staticmethod
    def _f32(t: torch.Tensor) -> torch.Tensor:
        return t if t.dtype == torch.float32 else t.float()

    @staticmethod
    def _full_nonfinite(x: torch.Tensor, sampled: torch.Tensor) -> torch.Tensor:
        """Sampled non-finite count, but at least 1 if *any* element of ``x`` is NaN/Inf.

        One reduction over the full tensor (NaN/Inf propagate through a sum), so a
        single bad value outside the sample is never missed."""
        total = x.detach().sum(dtype=torch.float32)
        return torch.where(torch.isfinite(total), sampled, sampled.clamp_min(1.0))

    def activation(self, x: torch.Tensor, kind: str = "other") -> torch.Tensor:
        f, stride = self.sample(x)
        f = self._f32(f)
        finite = torch.isfinite(f)
        n = finite.sum().to(f.dtype)
        nf = n.clamp_min(1.0)
        z = torch.where(finite, f, torch.zeros((), dtype=f.dtype, device=f.device))
        s2 = (z * z).sum()
        mean = z.sum() / nf
        std = (s2 / nf - mean * mean).clamp_min(0).sqrt()
        inf = torch.full((), float("inf"), dtype=f.dtype, device=f.device)
        mn = torch.where(finite, f, inf).amin()
        mx = torch.where(finite, f, -inf).amax()
        absmean = z.abs().sum() / nf
        frac_zero = ((f == 0) & finite).sum().to(f.dtype) / nf
        if kind == "tanh":
            sat = (z.abs() >= 0.99).sum().to(f.dtype) / nf
        elif kind == "sigmoid":
            sat = (((z <= 0.01) | (z >= 0.99)) & finite).sum().to(f.dtype) / nf
        else:
            sat = torch.zeros((), dtype=f.dtype, device=f.device)
        scale = math.sqrt(stride)
        nonfinite = (f.numel() - n) * stride
        if stride > 1:
            nonfinite = self._full_nonfinite(x, nonfinite)
        return torch.stack([mean, std, mn, mx, absmean, frac_zero, sat, s2.sqrt() * scale, nonfinite])

    def gradient(self, g: torch.Tensor) -> torch.Tensor:
        f, stride = self.sample(g)
        f = self._f32(f)
        finite = torch.isfinite(f)
        n = finite.sum().to(f.dtype)
        z = torch.where(finite, f, torch.zeros((), dtype=f.dtype, device=f.device)).abs()
        norm = (z * z).sum().sqrt() * math.sqrt(stride)
        nonfinite = (f.numel() - n) * stride
        if stride > 1:
            nonfinite = self._full_nonfinite(g, nonfinite)
        return torch.stack([norm, z.sum() / n.clamp_min(1.0), z.amax(), nonfinite])

    def histogram(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised histogram over the tensor's own [min, max] range (edges: use the min/max stats)."""
        f, _ = self.sample(x)
        f = self._f32(f)
        f = torch.where(torch.isfinite(f), f, torch.zeros((), dtype=f.dtype, device=f.device))
        h = torch.histc(f, bins=self.histogram_bins, min=0, max=0)
        return h / h.sum().clamp_min(1.0)


# ============================================================ data model
@dataclass
class LayerStats:
    name: str
    kind: str
    calls: int
    numel: int
    act_mean: float | None = None
    act_std: float | None = None
    act_min: float | None = None
    act_max: float | None = None
    act_absmean: float | None = None
    act_frac_zero: float | None = None
    act_frac_saturated: float | None = None
    act_norm: float | None = None
    act_nonfinite: float | None = None
    grad_norm: float | None = None
    grad_absmean: float | None = None
    grad_absmax: float | None = None
    grad_nonfinite: float | None = None
    histogram: list[float] | None = None


@dataclass
class ParamStats:
    name: str
    shape: tuple[int, ...]
    grad_norm: float | None
    grad_absmax: float | None
    weight_norm: float | None
    delta_norm: float | None
    update_ratio: float | None


@dataclass
class Anomaly:
    code: str
    target: str
    value: float | None
    message: str


@dataclass
class TrainingState:
    """One perception snapshot of a training step."""

    step: int
    wall_time: float
    training: bool
    loss: float | None
    loss_ema: float | None
    loss_z: float | None
    loss_spike: bool
    n_backward: int
    lrs: list[float]
    grad_norm: float | None
    weight_norm: float | None
    delta_norm: float | None
    update_ratio: float | None
    layers: dict[str, LayerStats] = field(default_factory=dict)
    params: dict[str, ParamStats] = field(default_factory=dict)
    anomalies: list[Anomaly] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        def f(v: float | None, spec: str = ".4g") -> str:
            return "-" if v is None else format(v, spec)

        flags = ",".join(sorted({a.code for a in self.anomalies})) or "ok"
        return (f"step {self.step}: loss={f(self.loss)} grad_norm={f(self.grad_norm)} "
                f"update_ratio={f(self.update_ratio, '.3g')} layers={len(self.layers)} [{flags}]")


# ============================================================== streaming
class SnapshotStream:
    """Bounded, thread-safe ring buffer of snapshots that never blocks the producer."""

    def __init__(self, capacity: int = 1024) -> None:
        self._q: deque[TrainingState] = deque(maxlen=capacity)
        self._cv = threading.Condition()
        self._subs: list[Callable[[TrainingState], None]] = []
        self._closed = False
        self.latest: TrainingState | None = None
        self.published = 0
        self.dropped = 0
        self.subscriber_errors = 0

    def publish(self, state: TrainingState) -> None:
        with self._cv:
            if self._q.maxlen is not None and len(self._q) == self._q.maxlen:
                self.dropped += 1
            self._q.append(state)
            self.latest = state
            self.published += 1
            subs = list(self._subs)
            self._cv.notify_all()
        for cb in subs:
            try:
                cb(state)
            except Exception:  # a broken consumer must never break training
                self.subscriber_errors += 1

    def subscribe(self, callback: Callable[[TrainingState], None]) -> Callable[[], None]:
        with self._cv:
            self._subs.append(callback)

        def unsubscribe() -> None:
            with self._cv:
                if callback in self._subs:
                    self._subs.remove(callback)

        return unsubscribe

    def get(self, timeout: float | None = None) -> TrainingState | None:
        """Next unread snapshot (blocks up to ``timeout``); ``None`` on timeout or when closed and empty."""
        with self._cv:
            self._cv.wait_for(lambda: bool(self._q) or self._closed, timeout)
            return self._q.popleft() if self._q else None

    def drain(self) -> list[TrainingState]:
        with self._cv:
            out = list(self._q)
            self._q.clear()
            return out

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def __iter__(self) -> Iterator[TrainingState]:
        while True:
            s = self.get()
            if s is None:
                return
            yield s

    def __len__(self) -> int:
        return len(self._q)


# ============================================================== collectors
class ActivationCollector:
    """Forward hooks (activations) and output-tensor hooks (gradients) on selected layers."""

    def __init__(self, layers: list[tuple[str, nn.Module, str]], extractor: TensorStatsExtractor, *,
                 track_grads: bool = True, record_eval: bool = False) -> None:
        self.layers = layers
        self.extractor = extractor
        self.track_grads = track_grads
        self.record_eval = record_eval
        self.active = True
        self.handles: list[Any] = []
        self.reset()

    def reset(self) -> None:
        self.act: dict[int, torch.Tensor] = {}
        self.grad: dict[int, torch.Tensor] = {}
        self.hist: dict[int, torch.Tensor] = {}
        self.calls: dict[int, int] = {}
        self.numel: dict[int, int] = {}

    def attach(self) -> None:
        for i, (_name, module, _kind) in enumerate(self.layers):
            self.handles.append(module.register_forward_hook(partial(self._on_forward, i)))

    def detach(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _on_forward(self, i: int, module: nn.Module, inputs: Any, output: Any) -> None:
        if not self.active or _compiling():
            return
        if not torch.is_grad_enabled() and not self.record_eval:
            return
        t = _first_float_tensor(output)
        if t is None:
            return
        kind = self.layers[i][2]
        with torch.no_grad():
            self.act[i] = self.extractor.activation(t, kind)
            if self.extractor.histogram_bins:
                self.hist[i] = self.extractor.histogram(t)
        self.calls[i] = self.calls.get(i, 0) + 1
        self.numel[i] = t.numel()
        if self.track_grads and t.requires_grad:
            t.register_hook(partial(self._on_grad, i))

    def _on_grad(self, i: int, g: torch.Tensor) -> None:
        if self.active and g is not None:
            with torch.no_grad():
                self.grad[i] = self.extractor.gradient(g)
        return None


class ParameterCollector:
    """Optimizer-step hooks: gradient norms, weight norms and weight deltas per parameter."""

    def __init__(self, named_params: list[tuple[str, nn.Parameter]], max_elems: int = 1 << 20) -> None:
        self.names = [n for n, _ in named_params]
        self.params = [p for _, p in named_params]
        self.max_elems = max_elems
        self._shadow: list[torch.Tensor | None] = [None] * len(self.params)
        self._strides = [1] * len(self.params)
        self._pre: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]] | None = None
        self.last_has_grad: list[bool] = [False] * len(self.params)
        self.device: torch.device | None = self.params[0].device if self.params else None

    def _to_dev(self, t: torch.Tensor) -> torch.Tensor:
        return t if self.device is None or t.device == self.device else t.to(self.device, non_blocking=True)

    def _sample(self, i: int, p: torch.Tensor) -> torch.Tensor:
        flat = p.detach()
        flat = flat.reshape(-1) if flat.is_contiguous() else flat.flatten()
        s = self._strides[i]
        return flat[::s] if s > 1 else flat

    @torch.no_grad()
    def before_step(self) -> None:
        nan = torch.full((), float("nan"), device=self.device)
        gnorm, gmax, wnorm = [], [], []
        grads = [p.grad for p in self.params if p.grad is not None]
        self.last_has_grad = [p.grad is not None for p in self.params]
        g_norms = iter(torch._foreach_norm(grads)) if grads else iter(())
        w_norms = torch._foreach_norm([p.detach() for p in self.params]) if self.params else []
        for i, p in enumerate(self.params):
            if self._shadow[i] is None:
                self._strides[i] = max(1, math.ceil(p.numel() / self.max_elems))
                self._shadow[i] = torch.empty_like(self._sample(i, p))
            if p.grad is not None:
                gnorm.append(self._to_dev(next(g_norms).float()))  # exact (fused foreach norm)
                gmax.append(self._to_dev(self._sample(i, p.grad).abs().amax().float()))  # on the sample
            else:
                gnorm.append(nan)
                gmax.append(nan)
            wnorm.append(self._to_dev(w_norms[i].float()))
            self._shadow[i].copy_(self._sample(i, p))  # the only weight copy, into a reused buffer
        self._pre = (gnorm, gmax, wnorm)

    @torch.no_grad()
    def after_step(self) -> torch.Tensor:
        if self._pre is None:
            self.before_step()
        gnorm, gmax, wnorm = self._pre  # type: ignore[misc]
        deltas = []
        for i, p in enumerate(self.params):
            buf = self._shadow[i]
            buf.sub_(self._sample(i, p))  # in place: old - new
            deltas.append(buf)
        d_norms = torch._foreach_norm(deltas) if deltas else []
        dnorm = [self._to_dev(d.float()) * math.sqrt(self._strides[i]) for i, d in enumerate(d_norms)]
        self._pre = None
        if not self.params:
            return torch.empty(0, len(PARAM_FIELDS))
        return torch.stack([torch.stack(gnorm), torch.stack(gmax), torch.stack(wnorm), torch.stack(dnorm)], dim=1)


class LossCollector:
    """Captures every scalar passed to ``backward()`` (as a device tensor, no sync)."""

    def __init__(self) -> None:
        self.values: list[torch.Tensor] = []
        self._orig: Any = None
        self._in = False

    def install(self) -> None:
        if self._orig is not None:
            return
        self._orig = torch.Tensor.backward
        orig, coll = self._orig, self

        def backward(t, *a, **kw):  # type: ignore[no-untyped-def]
            if not coll._in and not _compiling() and t.numel() == 1 and t.is_floating_point():
                coll.values.append(t.detach().reshape(()).float())
            coll._in = True
            try:
                return orig(t, *a, **kw)
            finally:
                coll._in = False

        torch.Tensor.backward = backward  # type: ignore[assignment]

    def uninstall(self) -> None:
        if self._orig is not None:
            torch.Tensor.backward = self._orig  # type: ignore[assignment]
            self._orig = None

    def observe(self, loss: torch.Tensor | float) -> None:
        if isinstance(loss, torch.Tensor):
            self.values.append(loss.detach().reshape(()).float())
        else:
            self.values.append(torch.tensor(float(loss)))


# ============================================================ orchestrator
@dataclass
class HookConfig:
    every_n_steps: int = 1
    layers: str | Callable[[str, nn.Module], bool] = "auto"
    max_layers: int = 256
    max_elems: int = 1 << 16  # per activation/gradient: larger tensors are sampled (strided view)
    max_param_elems: int = 1 << 20  # per weight tensor, for the weight-delta sample
    histogram_bins: int = 0
    track_activation_grads: bool = True
    record_eval_forwards: bool = False
    capture_backward_loss: bool = True
    stream_capacity: int = 1024
    max_pending: int = 4
    loss_z: float = 6.0
    loss_warmup: int = 10
    dead_threshold: float = 0.99
    saturation_threshold: float = 0.95
    max_update_ratio: float = 1.0
    grad_z: float = 6.0


def select_layers(model: nn.Module, rule: str | Callable[[str, nn.Module], bool] = "auto",
                  max_layers: int = 256) -> list[tuple[str, nn.Module, str]]:
    out: list[tuple[str, nn.Module, str]] = []
    covered: list[str] = []
    for name, m in model.named_modules():
        name = name.replace("_orig_mod.", "").replace("_orig_mod", "")
        if not name:
            continue
        if callable(rule):
            keep = rule(name, m)
        else:
            if isinstance(m, _SKIP_TYPES):
                continue
            is_leaf = not any(True for _ in m.children())
            special = isinstance(m, (nn.MultiheadAttention, nn.RNNBase))
            inside_special = any(name.startswith(c + ".") for c in covered)
            keep = (is_leaf or special) and not inside_special if rule == "auto" else is_leaf
            if special and keep:
                covered.append(name)
        if keep:
            out.append((name, m, layer_kind(m)))
        if len(out) >= max_layers:
            break
    return out


class RuntimeHooks:
    """Attach non-intrusive hooks to ``model`` (and ``optimizer``) and stream ``TrainingState`` snapshots.

    Usage::

        hooks = RuntimeHooks(model, optimizer).attach()
        for batch in loader:            # the training loop is unchanged
            ...
        states = hooks.stream.drain()   # or hooks.stream.get() from an agent thread
        hooks.detach()

    Without an optimizer (or with manual updates) call ``hooks.end_step()`` after each step."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer | None = None,
                 config: HookConfig | None = None, stream: SnapshotStream | None = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config or HookConfig()
        self.stream = stream or SnapshotStream(self.config.stream_capacity)
        self.extractor = TensorStatsExtractor(self.config.max_elems, self.config.histogram_bins)
        self.layers = select_layers(model, self.config.layers, self.config.max_layers)
        self.acts = ActivationCollector(self.layers, self.extractor, track_grads=self.config.track_activation_grads,
                                        record_eval=self.config.record_eval_forwards)
        names = {id(p): n.replace("_orig_mod.", "") for n, p in model.named_parameters()}
        if optimizer is not None:
            params = [p for g in optimizer.param_groups for p in g["params"]]
        else:
            params = [p for p in model.parameters() if p.requires_grad]
        self.param_coll = ParameterCollector([(names.get(id(p), f"param{i}"), p) for i, p in enumerate(params)],
                                             self.config.max_param_elems)
        self.losses = LossCollector()
        self._handles: list[Any] = []
        self._step = 0
        self._attached = False
        self._pending: deque[tuple[Any, torch.Tensor, dict[str, Any]]] = deque()
        self._free: list[torch.Tensor] = []
        self._loss_det = EmaZScore(alpha=0.05, warmup=self.config.loss_warmup, z=self.config.loss_z)
        self._grad_det = EmaZScore(alpha=0.05, warmup=self.config.loss_warmup, z=self.config.grad_z)
        self._mode_sensitive = any(isinstance(m, (nn.modules.dropout._DropoutNd, nn.modules.batchnorm._BatchNorm))
                                   for m in model.modules())
        self._nan_rows: dict[tuple[int, str], torch.Tensor] = {}

    # ------------------------------------------------------------ lifecycle
    @property
    def step(self) -> int:
        return self._step

    @property
    def monitoring(self) -> bool:
        return self._step % max(1, self.config.every_n_steps) == 0

    def attach(self) -> "RuntimeHooks":
        if self._attached:
            return self
        self.acts.attach()
        self.acts.active = self.monitoring
        if self.config.capture_backward_loss:
            self.losses.install()
        if self.optimizer is not None:
            self._handles.append(self.optimizer.register_step_pre_hook(self._on_opt_pre))
            self._handles.append(self.optimizer.register_step_post_hook(self._on_opt_post))
        self._attached = True
        return self

    def detach(self) -> None:
        if not self._attached:
            return
        self.flush()
        self.acts.detach()
        self.losses.uninstall()
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._attached = False

    def __enter__(self) -> "RuntimeHooks":
        return self.attach()

    def __exit__(self, *exc: Any) -> None:
        self.detach()

    def observe_loss(self, loss: torch.Tensor | float) -> None:
        """Report a loss explicitly (when ``capture_backward_loss`` is off or backward() is not used)."""
        self.losses.observe(loss)

    # --------------------------------------------------------------- steps
    def _on_opt_pre(self, opt: Any, args: Any, kwargs: Any) -> None:
        if self.monitoring:
            self.param_coll.before_step()

    def _on_opt_post(self, opt: Any, args: Any, kwargs: Any) -> None:
        self._complete_step(self.param_coll.after_step() if self.monitoring else None)

    def end_step(self) -> None:
        """Mark the end of a step when no optimizer hook is available."""
        rows = None
        if self.monitoring:
            self.param_coll.before_step()
            rows = self.param_coll.after_step()
        self._complete_step(rows)

    def _complete_step(self, param_rows: torch.Tensor | None) -> None:
        if param_rows is not None:
            self._finalize(param_rows)
        self.acts.reset()
        self.losses.values.clear()
        self._step += 1
        self.acts.active = self.monitoring
        self._poll(block=False)

    def flush(self) -> None:
        """Publish every pending snapshot (waits for in-flight device-to-host copies)."""
        self._poll(block=True)

    @property
    def latest(self) -> TrainingState | None:
        self._poll(block=False)
        return self.stream.latest

    # ----------------------------------------------------------- packing
    def _device(self) -> torch.device:
        if self.param_coll.device is not None:
            return self.param_coll.device
        for t in self.acts.act.values():
            return t.device
        return torch.device("cpu")

    def _nan_row(self, n: int, dev: torch.device) -> torch.Tensor:
        key = (n, str(dev))
        if key not in self._nan_rows:
            self._nan_rows[key] = torch.full((n,), float("nan"), device=dev)
        return self._nan_rows[key]

    def _rows(self, table: dict[int, torch.Tensor], width: int, dev: torch.device) -> torch.Tensor:
        L = len(self.layers)
        if L == 0 or width == 0:
            return torch.empty(0, device=dev)
        rows = [table[i] if i in table else self._nan_row(width, dev) for i in range(L)]
        rows = [r if r.device == dev else r.to(dev, non_blocking=True) for r in rows]
        return torch.stack(rows).reshape(-1)

    def _finalize(self, param_rows: torch.Tensor) -> None:
        dev = self._device()
        with torch.no_grad():
            losses = [v if v.device == dev else v.to(dev, non_blocking=True) for v in self.losses.values]
            loss = torch.stack(losses).mean().reshape(1) if losses else self._nan_row(1, dev)
            bins = self.extractor.histogram_bins
            parts = [
                loss,
                self._rows(self.acts.act, len(ACT_FIELDS), dev),
                self._rows(self.acts.grad, len(GRAD_FIELDS), dev),
                self._rows(self.acts.hist, bins, dev) if bins else torch.empty(0, device=dev),
                param_rows.to(dev).reshape(-1).float(),
            ]
            packed = torch.cat(parts)
        host = self._host_buffer(packed)
        host.copy_(packed, non_blocking=packed.is_cuda)
        event = None
        if packed.is_cuda:
            event = torch.cuda.Event()
            event.record()
        meta = {
            "step": self._step, "wall_time": time.time(), "training": bool(self.model.training),
            "n_backward": len(self.losses.values), "calls": dict(self.acts.calls), "numel": dict(self.acts.numel),
            "has_act": set(self.acts.act), "has_grad": set(self.acts.grad), "has_hist": set(self.acts.hist),
            "lrs": [float(g.get("lr", float("nan"))) for g in self.optimizer.param_groups] if self.optimizer else [],
            "has_param_grad": list(self.param_coll.last_has_grad),
            "n": packed.numel(),
        }
        self._pending.append((event, host, meta))
        while len(self._pending) > self.config.max_pending:
            self._publish(*self._pending.popleft(), block=True)

    def _host_buffer(self, packed: torch.Tensor) -> torch.Tensor:
        n = packed.numel()
        for i, b in enumerate(self._free):
            if b.numel() >= n:
                return self._free.pop(i)[:n]
        pin = packed.is_cuda and torch.cuda.is_available()
        return torch.empty(max(n, 64), dtype=torch.float32, pin_memory=pin)[:n]

    def _poll(self, block: bool) -> None:
        while self._pending:
            event, host, meta = self._pending[0]
            if not block and event is not None and not event.query():
                return
            self._pending.popleft()
            self._publish(event, host, meta, block=True)

    def _publish(self, event: Any, host: torch.Tensor, meta: dict[str, Any], block: bool) -> None:
        if event is not None and block:
            event.synchronize()
        vals = host.tolist()
        self._free.append(host)
        self.stream.publish(self._materialize(vals, meta))

    # ------------------------------------------------------- materialise
    def _materialize(self, vals: list[float], meta: dict[str, Any]) -> TrainingState:
        def num(v: float) -> float | None:
            return v if isinstance(v, float) and math.isfinite(v) else (None if v != v else v)

        L, bins = len(self.layers), self.extractor.histogram_bins
        pos = 0
        loss = vals[pos]
        pos += 1
        act = vals[pos : pos + L * len(ACT_FIELDS)]
        pos += L * len(ACT_FIELDS)
        grd = vals[pos : pos + L * len(GRAD_FIELDS)]
        pos += L * len(GRAD_FIELDS)
        hist = vals[pos : pos + (L * bins if bins else 0)]
        pos += L * bins if bins else 0
        prm = vals[pos:]
        layers: dict[str, LayerStats] = {}
        for i, (name, _m, kind) in enumerate(self.layers):
            if i not in meta["has_act"] and i not in meta["has_grad"]:
                continue
            ls = LayerStats(name, kind, meta["calls"].get(i, 0), meta["numel"].get(i, 0))
            if i in meta["has_act"]:
                row = act[i * len(ACT_FIELDS) : (i + 1) * len(ACT_FIELDS)]
                for f, v in zip(ACT_FIELDS, row):
                    setattr(ls, f"act_{f}", num(v) if f not in ("min", "max") else (v if math.isfinite(v) else None))
            if i in meta["has_grad"]:
                row = grd[i * len(GRAD_FIELDS) : (i + 1) * len(GRAD_FIELDS)]
                for f, v in zip(GRAD_FIELDS, row):
                    setattr(ls, f"grad_{f}", num(v))
            if bins and i in meta["has_hist"]:
                ls.histogram = hist[i * bins : (i + 1) * bins]
            layers[name] = ls
        params: dict[str, ParamStats] = {}
        P = len(self.param_coll.params)
        sq_g = sq_w = sq_d = 0.0
        any_g = False
        nonfinite_grad = []
        for j in range(P):
            gn, gm, wn, dn = prm[j * 4 : j * 4 + 4]
            name = self.param_coll.names[j]
            if meta["has_param_grad"][j]:
                if math.isfinite(gn):
                    sq_g += gn * gn
                    any_g = True
                else:
                    nonfinite_grad.append(name)
            sq_w += wn * wn if math.isfinite(wn) else 0.0
            sq_d += dn * dn if math.isfinite(dn) else 0.0
            params[name] = ParamStats(name, tuple(self.param_coll.params[j].shape), num(gn), num(gm), num(wn), num(dn),
                                      (dn / wn) if math.isfinite(dn) and math.isfinite(wn) and wn > 0 else None)
        grad_norm = math.sqrt(sq_g) if any_g and not nonfinite_grad else (float("nan") if nonfinite_grad else None)
        weight_norm = math.sqrt(sq_w) if P else None
        delta_norm = math.sqrt(sq_d) if P else None
        update_ratio = delta_norm / weight_norm if weight_norm else None
        loss_v = loss if meta["n_backward"] > 0 else None  # NaN here is a real NaN loss, not "missing"
        lz, spike, lema = None, False, None
        if loss_v is not None:
            if math.isfinite(loss_v):
                lz, spike = self._loss_det.update(loss_v)
                lema = self._loss_det.mean
        state = TrainingState(
            step=meta["step"], wall_time=meta["wall_time"], training=meta["training"], loss=loss_v, loss_ema=lema,
            loss_z=lz if lz is not None and math.isfinite(lz) else None, loss_spike=bool(spike),
            n_backward=meta["n_backward"], lrs=meta["lrs"],
            grad_norm=grad_norm if grad_norm is None or math.isfinite(grad_norm) else None,
            weight_norm=weight_norm, delta_norm=delta_norm, update_ratio=update_ratio, layers=layers, params=params,
        )
        state.anomalies = self._anomalies(state, nonfinite_grad, loss_v)
        return state

    def _anomalies(self, s: TrainingState, nonfinite_grad: list[str], loss: float | None) -> list[Anomaly]:
        cfg = self.config
        out: list[Anomaly] = []
        if loss is not None and not math.isfinite(loss):
            out.append(Anomaly("nonfinite_loss", "loss", loss, f"loss is {loss}"))
        if s.loss_spike:
            out.append(Anomaly("loss_spike", "loss", s.loss_z, f"loss z-score {s.loss_z:.1f} above {cfg.loss_z}"))
        for name in nonfinite_grad:
            out.append(Anomaly("nonfinite_gradient", name, None, f"gradient of {name} contains NaN/Inf"))
        for ls in s.layers.values():
            if ls.act_nonfinite:
                out.append(Anomaly("nonfinite_activation", ls.name, ls.act_nonfinite,
                                   f"{ls.name} output has {ls.act_nonfinite:.0f} NaN/Inf values"))
            if ls.grad_nonfinite:
                out.append(Anomaly("nonfinite_activation_grad", ls.name, ls.grad_nonfinite,
                                   f"gradient into {ls.name} has {ls.grad_nonfinite:.0f} NaN/Inf values"))
            if ls.kind in _DEAD_PRONE and ls.act_frac_zero is not None and ls.act_frac_zero >= cfg.dead_threshold:
                out.append(Anomaly("dead_units", ls.name, ls.act_frac_zero,
                                   f"{ls.act_frac_zero:.0%} of {ls.name} outputs are exactly zero"))
            if ls.kind in _SATURATING and ls.act_frac_saturated is not None and ls.act_frac_saturated >= cfg.saturation_threshold:
                out.append(Anomaly("saturated_units", ls.name, ls.act_frac_saturated,
                                   f"{ls.act_frac_saturated:.0%} of {ls.name} outputs are saturated"))
        if s.update_ratio is not None and s.update_ratio > cfg.max_update_ratio:
            out.append(Anomaly("update_too_large", "parameters", s.update_ratio,
                               f"the step changed the weights by {s.update_ratio:.3g}x their norm"))
        if (s.delta_norm == 0.0 and s.grad_norm and s.grad_norm > 0 and any(lr > 0 for lr in s.lrs)):
            out.append(Anomaly("no_update", "parameters", 0.0, "optimizer.step() did not change any weight"))
        if s.grad_norm is not None and s.grad_norm > 0:
            z, spike = self._grad_det.update(math.log(s.grad_norm))
            if spike:
                out.append(Anomaly("grad_norm_spike", "gradients", s.grad_norm, f"gradient norm {s.grad_norm:.3g} spiked (z={z:.1f})"))
        if self.optimizer is not None and self._mode_sensitive and not s.training:
            out.append(Anomaly("eval_mode_training", "model", None,
                               "optimizer stepped while the model was in eval mode (dropout/batch-norm frozen)"))
        return out
