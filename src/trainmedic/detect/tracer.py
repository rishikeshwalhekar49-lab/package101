"""Runtime tracing of PyTorch training (TrainCheck-style API/state tracing).

The tracer installs *global* hooks, so it works on unmodified scripts:

* module forward pre/post hooks (root calls only; skipped while torch.compile traces),
* optimizer step pre/post hooks plus ``Optimizer.__init__``/``zero_grad`` wrappers,
* ``Tensor.backward`` / ``torch.autograd.backward`` wrappers,
* ``nn.Module.train`` (mode changes), LR scheduler ``step``,
* functional loss wrappers (``F.cross_entropy`` etc. -- also used by loss modules),
* ``DataLoader.__iter__`` (which loader feeds which forward; sampler; subset overlap).

On the first trainable root forward of each model it also runs three cheap,
side-effect-free behavioural probes in eval mode (RNG state is restored):
eval-mode determinism, batch independence and causality (for sequence models).

Every observation becomes a JSON event in a :class:`~trainmedic.detect.trace.Trace`;
invariants are checked on the trace afterwards (see :mod:`.invariants`).
"""

from __future__ import annotations

import math
import time
import traceback
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import register_module_forward_hook, register_module_forward_pre_hook
from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

from .trace import Trace


class StopTraining(BaseException):
    """Raised from an optimizer hook once the step/time budget is exhausted.

    It derives from BaseException so that ``except Exception`` blocks in user
    code do not swallow it."""


LOSS_FUNCS = (
    "cross_entropy", "nll_loss", "binary_cross_entropy", "binary_cross_entropy_with_logits", "mse_loss",
    "l1_loss", "smooth_l1_loss", "huber_loss", "kl_div",
)
_DROPOUT_BN = (nn.modules.dropout._DropoutNd, nn.modules.batchnorm._BatchNorm)
_ACTIVE: "Tracer | None" = None


def active_tracer() -> "Tracer | None":
    return _ACTIVE


def _compiling() -> bool:
    try:
        return bool(torch.compiler.is_compiling())
    except Exception:
        return False


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else (float("nan") if v != v else (float("inf") if v > 0 else float("-inf")))


def _first_tensor(obj: Any, batch: int | None = None) -> torch.Tensor | None:
    if isinstance(obj, torch.Tensor):
        return obj if batch is None or (obj.dim() > 0 and obj.size(0) == batch) else None
    if isinstance(obj, dict):
        for k in ("logits", "last_hidden_state", "loss", "out", "output", "pred"):
            if k in obj:
                t = _first_tensor(obj[k], batch)
                if t is not None and (k != "loss" or batch is None):
                    return t
        obj = list(obj.values())
    if hasattr(obj, "to_tuple"):
        obj = obj.to_tuple()
    if isinstance(obj, (list, tuple)):
        for o in obj:
            t = _first_tensor(o, batch)
            if t is not None and t.dim() > 0:
                return t
    return None


def _tensor_summary(t: torch.Tensor | None) -> dict[str, Any]:
    if t is None:
        return {}
    d: dict[str, Any] = {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
    if t.is_floating_point() and t.numel() > 0:
        with torch.no_grad():
            s = t.detach().float().flatten()[:65536]
            d["mean"] = _f(s.mean())
            d["std"] = _f(s.std()) if s.numel() > 1 else 0.0
    return d


class Tracer:
    def __init__(
        self,
        *,
        max_steps: int | None = None,
        max_seconds: float | None = None,
        probes: bool = True,
        param_sample_limit: int = 2_000_000,
        dist_check_every: int = 10,
    ) -> None:
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.probes = probes
        self.param_sample_limit = param_sample_limit
        self.dist_check_every = dist_check_every
        self.events: list[dict[str, Any]] = []
        self.meta: dict[str, Any] = {"torch": torch.__version__}
        self._installed = False
        self._handles: list[Any] = []
        self._patches: list[tuple[Any, str, Any]] = []
        self._labels: dict[int, str] = {}
        self._keep: list[Any] = []
        self._counts: dict[str, int] = {}
        self._depth = 0
        self._pending_root: tuple[Any, tuple] | None = None
        self._in_probe = False
        self._in_backward = False
        self._in_train_call = False
        self._in_sched_step = False
        self._probed: set[str] = set()
        self._optimizers: dict[int, Any] = {}
        self._opt_steps = 0
        self._bw_window = 0
        self._bw_since_opt: dict[str, int] = {}
        self._snap: dict[int, tuple[int, torch.Tensor]] = {}
        self._param_names: dict[int, str] = {}
        self._root_modules: dict[str, Any] = {}
        self._current_loader: str | None = None
        self._loader_info: dict[str, dict[str, Any]] = {}
        self._subset_indices: dict[str, tuple[str, set[int]]] = {}
        self._last_loss: dict[str, Any] | None = None
        self._covered: set[str] = set()
        self._t0 = time.time()

    # --------------------------------------------------------------- basics
    def _label(self, obj: Any, prefix: str) -> str:
        key = id(obj)
        lab = self._labels.get(key)
        if lab is None:
            n = self._counts.get(prefix, 0)
            self._counts[prefix] = n + 1
            lab = f"{prefix}{n}"
            self._labels[key] = lab
            self._keep.append(obj)  # keep alive so ids are not reused
        return lab

    def emit(self, kind: str, **fields: Any) -> dict[str, Any]:
        ev = {"kind": kind, "i": len(self.events), "step": self._opt_steps, "t": round(time.time() - self._t0, 4)}
        ev.update(fields)
        self.events.append(ev)
        return ev

    @property
    def quiet(self) -> bool:
        return self._in_probe or _compiling()

    # -------------------------------------------------------------- install
    def install(self) -> "Tracer":
        global _ACTIVE
        if self._installed:
            return self
        # torch warns that global hooks also fire for torch.compile's wrapper module; we rely on exactly that
        warnings.filterwarnings("ignore", message=r"Using `torch.compile\(module\)` when there are global hooks")
        self._handles.append(register_module_forward_pre_hook(self._fwd_pre))
        self._handles.append(register_module_forward_hook(self._fwd_post))
        self._handles.append(register_optimizer_step_pre_hook(self._opt_pre))
        self._handles.append(register_optimizer_step_post_hook(self._opt_post))
        self._patch(torch.optim.Optimizer, "__init__", self._wrap_opt_init)
        self._patch(torch.optim.Optimizer, "zero_grad", self._wrap_zero_grad("optimizer"))
        self._patch(nn.Module, "zero_grad", self._wrap_zero_grad("module"))
        self._patch(nn.Module, "train", self._wrap_train)
        self._patch(torch.Tensor, "backward", self._wrap_backward)
        self._patch(torch.autograd, "backward", self._wrap_backward_fn)
        for name in LOSS_FUNCS:
            if hasattr(F, name):
                self._patch(F, name, self._wrap_loss(name))
        self._patch(torch.utils.data.DataLoader, "__iter__", self._wrap_loader_iter)
        self._patch_schedulers()
        from .. import reporting as _report

        _report.add_listener(self._on_metric)
        self._installed = True
        _ACTIVE = self
        self._t0 = time.time()
        return self

    def uninstall(self) -> None:
        global _ACTIVE
        if not self._installed:
            return
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for owner, name, orig in reversed(self._patches):
            setattr(owner, name, orig)
        self._patches.clear()
        from .. import reporting as _report

        _report.remove_listener(self._on_metric)
        self._installed = False
        if _ACTIVE is self:
            _ACTIVE = None
        self._finalize()

    def __enter__(self) -> "Tracer":
        return self.install()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is StopTraining:
            self.meta["stopped_early"] = True
            self.uninstall()
            return True
        if exc is not None:
            self.meta["error"] = "".join(traceback.format_exception_only(exc_type, exc)).strip()
        self.uninstall()
        return False

    def _patch(self, owner: Any, name: str, factory) -> None:
        orig = owner.__dict__[name] if isinstance(owner, type) and name in owner.__dict__ else getattr(owner, name)
        self._patches.append((owner, name, orig))
        setattr(owner, name, factory(orig))

    def trace(self) -> Trace:
        return Trace(list(self.events), dict(self.meta))

    def register_optimizer(self, opt: Any) -> str:
        lab = self._label(opt, "O")
        if id(opt) not in self._optimizers:
            self._optimizers[id(opt)] = opt
            self.emit("optimizer", opt=lab, cls=type(opt).__name__, groups=len(opt.param_groups),
                      lrs=[_f(g.get("lr")) for g in opt.param_groups],
                      n_params=sum(len(g["params"]) for g in opt.param_groups))
        return lab

    def register_module(self, module: nn.Module) -> str:
        lab = self._label(module, "M")
        if lab not in self._root_modules:
            self._root_modules[lab] = module
            inner = getattr(module, "_orig_mod", module)
            for n, p in inner.named_parameters():
                self._param_names.setdefault(id(p), f"{lab}.{n}")
        return lab

    # -------------------------------------------------------------- forward
    def _fwd_pre(self, module: nn.Module, args: tuple) -> None:
        if self.quiet or isinstance(module, nn.modules.loss._Loss):
            return
        self._depth += 1
        if self._depth == 1:
            self._pending_root = (module, args)

    def _fwd_post(self, module: nn.Module, args: tuple, output: Any) -> None:
        if self.quiet or isinstance(module, nn.modules.loss._Loss):
            return
        self._depth = max(0, self._depth - 1)
        if self._depth != 0 or self._pending_root is None or self._pending_root[0] is not module:
            return
        self._pending_root = None
        lab = self.register_module(module)
        inner = getattr(module, "_orig_mod", module)
        x = args[0] if args and isinstance(args[0], torch.Tensor) else None
        trainable = any(p.requires_grad for p in inner.parameters())
        self.emit(
            "forward",
            module=lab,
            cls=type(inner).__name__,
            training=bool(module.training),
            mode_sensitive=any(isinstance(m, _DROPOUT_BN) for m in inner.modules()),
            grad_enabled=torch.is_grad_enabled(),
            trainable=trainable,
            loader=self._current_loader,
            compiled=type(module).__name__ == "OptimizedModule",
            input=_tensor_summary(x),
            output=_tensor_summary(_first_tensor(output)),
        )
        if self.probes and lab not in self._probed and trainable and x is not None and torch.is_grad_enabled():
            self._probed.add(lab)
            self._run_probes(lab, module, args, output)

    # --------------------------------------------------------------- probes
    def _run_probes(self, lab: str, module: nn.Module, args: tuple, output: Any) -> None:
        self._in_probe = True
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
        modes = {m: m.training for m in module.modules()}
        try:
            module.eval()
            from . import probes

            target = getattr(module, "_orig_mod", module)  # probe compiled models eagerly (no recompiles)
            for res in probes.run_all(target, args):
                self.emit("probe", module=lab, **res)
        except Exception as exc:  # probes are best-effort
            self.emit("probe", module=lab, probe="error", passed=True, detail=f"{type(exc).__name__}: {exc}"[:300])
        finally:
            for m, t in modes.items():
                m.training = t
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            self._in_probe = False

    # ------------------------------------------------------------- backward
    def _tracked_params(self) -> list[torch.Tensor]:
        ps: list[torch.Tensor] = []
        for opt in self._optimizers.values():
            for g in opt.param_groups:
                ps.extend(g["params"])
        return ps

    def _pre_backward(self, tensor: torch.Tensor) -> None:
        self._depth = 0
        fields: dict[str, Any] = {"grad_enabled": torch.is_grad_enabled(), "requires_grad": bool(tensor.requires_grad)}
        if tensor.numel() == 1:
            fields["value"] = _f(tensor.detach())
        if self._last_loss is not None:
            fields["loss_fn"] = self._last_loss["fn"]
            fields["loss_value"] = self._last_loss.get("value")
        if self._bw_window == 0:
            params = self._tracked_params()
            if params:
                dirty = 0
                for p in params:
                    if p.grad is not None and bool(torch.any(p.grad != 0)):
                        dirty += 1
                fields["grads_clean"] = dirty == 0
                fields["dirty_params"] = dirty
        self._bw_window += 1
        for lab in list(self._bw_since_opt) + [self._labels[k] for k in self._optimizers if self._labels.get(k) not in self._bw_since_opt]:
            self._bw_since_opt[lab] = self._bw_since_opt.get(lab, 0) + 1
        fields["window_index"] = self._bw_window
        self.emit("backward", **fields)
        self._last_loss = None

    def _wrap_backward(self, orig):
        tracer = self

        def backward(t, *a, **kw):
            if tracer._in_backward or tracer.quiet:
                return orig(t, *a, **kw)
            tracer._in_backward = True
            try:
                tracer._pre_backward(t)
                return orig(t, *a, **kw)
            finally:
                tracer._in_backward = False

        return backward

    def _wrap_backward_fn(self, orig):
        tracer = self

        def autograd_backward(tensors, *a, **kw):
            if tracer._in_backward or tracer.quiet:
                return orig(tensors, *a, **kw)
            tracer._in_backward = True
            try:
                t = tensors[0] if isinstance(tensors, (list, tuple)) and tensors else tensors
                if isinstance(t, torch.Tensor):
                    tracer._pre_backward(t)
                return orig(tensors, *a, **kw)
            finally:
                tracer._in_backward = False

        return autograd_backward

    # ------------------------------------------------------------ optimizer
    def _wrap_opt_init(self, orig):
        tracer = self

        def __init__(opt, *a, **kw):
            orig(opt, *a, **kw)
            if tracer._installed:
                tracer.register_optimizer(opt)

        return __init__

    def _wrap_zero_grad(self, owner_kind: str):
        tracer = self

        def make(orig):
            def zero_grad(obj, *a, **kw):
                if not tracer.quiet:
                    lab = tracer._label(obj, "O" if owner_kind == "optimizer" else "M")
                    tracer.emit("zero_grad", owner=lab, owner_kind=owner_kind)
                return orig(obj, *a, **kw)

            return zero_grad

        return make

    def _opt_pre(self, opt: Any, args: Any, kwargs: Any) -> None:
        if self._in_probe:
            return
        lab = self.register_optimizer(opt)
        self._depth = 0
        n_params = n_grad = n_nonzero = 0
        sq = 0.0
        finite = True
        missing: list[str] = []
        budget = self.param_sample_limit
        self._snap = {}
        lrs = []
        with torch.no_grad():
            for g in opt.param_groups:
                lrs.append(_f(g.get("lr")))
                for p in g["params"]:
                    n_params += 1
                    if p.grad is None:
                        if len(missing) < 20:
                            missing.append(self._param_names.get(id(p), f"param{n_params}"))
                        continue
                    n_grad += 1
                    gn = float(p.grad.detach().float().norm())
                    if not math.isfinite(gn):
                        finite = False
                    else:
                        sq += gn * gn
                    if gn != 0.0:
                        n_nonzero += 1
                    elif len(missing) < 20:
                        missing.append(self._param_names.get(id(p), f"param{n_params}") + " (zero grad)")
                    flat = p.detach().flatten()
                    stride = max(1, flat.numel() // max(1, min(flat.numel(), budget)))
                    sample = flat[::stride].clone()
                    budget = max(1024, budget - sample.numel())
                    self._snap[id(p)] = (stride, sample)
        since = self._bw_since_opt.get(lab, 0)
        self.emit(
            "opt_pre", opt=lab, lrs=lrs, n_params=n_params, n_with_grad=n_grad, n_nonzero_grad=n_nonzero,
            grad_norm=math.sqrt(sq) if finite else float("nan"), grads_finite=finite, backwards_since_step=since,
            backwards_in_window=self._bw_window, params_without_grad=missing,
        )
        if lab not in self._covered:
            self._covered.add(lab)
            self._coverage_event(lab)

    def _coverage_event(self, lab: str) -> None:
        in_opt = {id(p) for p in self._tracked_params()}
        not_in_opt: list[str] = []
        for mlab, m in self._root_modules.items():
            inner = getattr(m, "_orig_mod", m)
            for n, p in inner.named_parameters():
                if p.requires_grad and id(p) not in in_opt and len(not_in_opt) < 20:
                    not_in_opt.append(f"{mlab}.{n}")
        self.emit("coverage", opt=lab, trainable_not_in_optimizer=not_in_opt)

    def _opt_post(self, opt: Any, args: Any, kwargs: Any) -> None:
        if self._in_probe:
            return
        lab = self._label(opt, "O")
        upd_sq = par_sq = 0.0
        changed = 0
        with torch.no_grad():
            for g in opt.param_groups:
                for p in g["params"]:
                    snap = self._snap.get(id(p))
                    if snap is None:
                        continue
                    stride, old = snap
                    new = p.detach().flatten()[::stride]
                    if new.shape != old.shape:
                        continue
                    d = (new.float() - old.float())
                    dn = float((d * d).sum())
                    upd_sq += dn
                    par_sq += float((old.float() ** 2).sum())
                    if dn > 0:
                        changed += 1
        self._snap = {}
        self._opt_steps += 1
        state_finite = True
        st = getattr(opt, "state", {})
        for i, s in enumerate(st.values()):
            if i >= 64:
                break
            for v in s.values() if isinstance(s, dict) else []:
                if isinstance(v, torch.Tensor) and v.is_floating_point() and v.numel() and not bool(torch.isfinite(v).all()):
                    state_finite = False
        un, pn = math.sqrt(upd_sq), math.sqrt(par_sq)
        self.emit("opt_step", opt=lab, update_norm=un, param_norm=pn, update_ratio=(un / pn) if pn > 0 else None,
                  n_changed=changed, optimizer_state_finite=state_finite)
        self._bw_since_opt[lab] = 0
        self._bw_window = 0
        self._maybe_dist_check()
        if self.max_steps is not None and self._opt_steps >= self.max_steps:
            self.meta["stopped_early"] = True
            raise StopTraining(f"trainmedic: reached max_steps={self.max_steps}")
        if self.max_seconds is not None and time.time() - self._t0 > self.max_seconds:
            self.meta["stopped_early"] = True
            raise StopTraining(f"trainmedic: reached max_seconds={self.max_seconds}")

    def _maybe_dist_check(self) -> None:
        try:
            import torch.distributed as dist
        except Exception:
            return
        if not (dist.is_available() and dist.is_initialized()) or self._opt_steps % self.dist_check_every:
            return
        world = dist.get_world_size()
        if world < 2:
            return
        from .distributed import param_checksum

        sums = []
        for m in self._root_modules.values():
            sums.append(param_checksum(getattr(m, "_orig_mod", m)))
        local = torch.tensor(sums or [0.0], dtype=torch.float64)
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            local = local.cuda()
        gathered = [torch.zeros_like(local) for _ in range(world)]
        dist.all_gather(gathered, local)
        self.emit("rank_checksums", rank=dist.get_rank(), checksums=[g.cpu().tolist() for g in gathered])

    # ----------------------------------------------------------- mode / lr
    def _wrap_train(self, orig):
        tracer = self

        def train(module, mode: bool = True):
            outer = not tracer._in_train_call
            tracer._in_train_call = True
            try:
                return orig(module, mode)
            finally:
                if outer:
                    tracer._in_train_call = False
                    if not tracer.quiet:
                        tracer.emit("mode", module=tracer._label(module, "M"), training=bool(mode))

        return train

    def _patch_schedulers(self) -> None:
        from torch.optim import lr_scheduler as ls

        base = getattr(ls, "LRScheduler", None) or getattr(ls, "_LRScheduler", None)
        classes = [base] if base is not None else []
        for obj in vars(ls).values():
            if isinstance(obj, type) and obj not in classes and "step" in obj.__dict__ and (
                base is not None and issubclass(obj, base) or obj.__name__ == "ReduceLROnPlateau"
            ):
                classes.append(obj)
        for cls in classes:
            self._patch(cls, "step", self._wrap_sched_step)

    def _wrap_sched_step(self, orig):
        tracer = self

        def step(sched, *a, **kw):
            if tracer._in_sched_step:
                return orig(sched, *a, **kw)
            tracer._in_sched_step = True
            try:
                out = orig(sched, *a, **kw)
            finally:
                tracer._in_sched_step = False
            if not tracer.quiet and getattr(sched, "last_epoch", 1) >= 1:
                try:
                    lrs = [_f(g["lr"]) for g in sched.optimizer.param_groups]
                except Exception:
                    lrs = []
                tracer.emit("sched_step", sched=tracer._label(sched, "S"), cls=type(sched).__name__, lrs=lrs,
                            opt_steps=tracer._opt_steps)
            return out

        return step

    # ----------------------------------------------------------------- loss
    def _wrap_loss(self, name: str):
        tracer = self

        def make(orig):
            def loss_fn(input, target, *a, **kw):
                out = orig(input, target, *a, **kw)
                if not tracer.quiet:
                    try:
                        tracer._record_loss(name, input, target, out, kw)
                    except Exception:
                        pass
                return out

            loss_fn.__wrapped__ = orig  # type: ignore[attr-defined]
            loss_fn.__name__ = name
            return loss_fn

        return make

    def _record_loss(self, name: str, inp: Any, target: Any, out: Any, kw: dict) -> None:
        fields: dict[str, Any] = {"fn": name, "reduction": kw.get("reduction", "mean")}
        if isinstance(out, torch.Tensor) and out.numel() == 1:
            fields["value"] = _f(out.detach())
        if isinstance(inp, torch.Tensor) and inp.is_floating_point() and inp.numel():
            with torch.no_grad():
                x = inp.detach()
                if x.dim() >= 2 and name in ("cross_entropy", "nll_loss"):
                    x = x[:256] if x.dim() == 2 else x[:16]
                    cls_dim = 1
                    fields["n_classes"] = int(x.size(cls_dim))
                    xf = x.float()
                    fields["in01"] = bool((xf >= 0).all() and (xf <= 1).all())
                    fields["rowsum_err"] = _f((xf.sum(cls_dim) - 1).abs().mean())
                    fields["all_nonpos"] = bool((xf <= 1e-6).all())
                    fields["lse_err"] = _f(torch.logsumexp(xf, dim=cls_dim).abs().mean())
                else:
                    xf = x.flatten()[:65536].float()
                    fields["in01"] = bool((xf >= 0).all() and (xf <= 1).all())
                    fields["frac_extreme"] = _f(((xf < 1e-4) | (xf > 1 - 1e-4)).float().mean())
                fields["input_mean"] = _f(x.float().mean())
        if isinstance(target, torch.Tensor) and target.numel():
            t = target.detach().flatten()[:65536]
            if not t.is_floating_point():
                u = torch.unique(t[t >= 0])
                fields["target_unique"] = int(u.numel())
                fields["target_max"] = int(t.max())
        self._last_loss = fields
        self.emit("loss", **fields)

    # -------------------------------------------------------------- loaders
    def _wrap_loader_iter(self, orig):
        tracer = self

        def __iter__(loader):
            it = orig(loader)
            if tracer.quiet:
                return it
            lab = tracer._label(loader, "L")
            if lab not in tracer._loader_info:
                tracer._register_loader(lab, loader)
            return _LoaderIter(it, lab, tracer)

        return __iter__

    def _register_loader(self, lab: str, loader: Any) -> None:
        ds = loader.dataset
        base = ds
        indices = None
        while isinstance(base, torch.utils.data.Subset):
            idx = list(base.indices)
            indices = idx if indices is None else [idx[i] for i in indices]
            base = base.dataset
        info = {
            "dataset": self._label(ds, "D"),
            "base_dataset": self._label(base, "D"),
            "dataset_cls": type(ds).__name__,
            "sampler": type(loader.sampler).__name__ if getattr(loader, "sampler", None) is not None else None,
            "batch_size": getattr(loader, "batch_size", None),
            "drop_last": getattr(loader, "drop_last", None),
        }
        try:
            info["size"] = len(ds)
        except Exception:
            info["size"] = None
        info["batches"] = 0
        info["unique_per_batch"] = []
        info["labels_seen"] = set()
        self._loader_info[lab] = info
        if indices is not None:
            self._subset_indices[lab] = (info["base_dataset"], set(indices))
        self.emit("loader", loader=lab, **{k: v for k, v in info.items() if k not in ("unique_per_batch", "labels_seen", "batches")})

    def _on_batch(self, lab: str, batch: Any) -> None:
        self._current_loader = lab
        info = self._loader_info.get(lab)
        if info is None:
            return
        info["batches"] += 1
        if info["batches"] > 200:
            return
        y = None
        if isinstance(batch, (list, tuple)) and len(batch) >= 2 and isinstance(batch[-1], torch.Tensor):
            y = batch[-1]
        elif isinstance(batch, dict):
            for k in ("labels", "label", "y", "targets", "target"):
                if isinstance(batch.get(k), torch.Tensor):
                    y = batch[k]
                    break
        if y is not None and not y.is_floating_point() and y.dim() == 1 and y.numel() > 1:
            u = torch.unique(y)
            info["unique_per_batch"].append(int(u.numel()))
            if len(info["labels_seen"]) < 10000:
                info["labels_seen"].update(int(v) for v in u[:1000])

    # -------------------------------------------------------------- metrics
    def _on_metric(self, metrics: dict[str, Any]) -> None:
        self.emit("metric", values={k: _f(v) if not isinstance(v, str) else v for k, v in metrics.items()})

    # ------------------------------------------------------------- finalize
    def _finalize(self) -> None:
        for lab, info in self._loader_info.items():
            upb = info["unique_per_batch"]
            self.emit(
                "loader_stats", loader=lab, batches=info["batches"],
                mean_unique_per_batch=(sum(upb) / len(upb)) if upb else None,
                labels_total=len(info["labels_seen"]) or None,
            )
        labs = list(self._subset_indices)
        for i, a in enumerate(labs):
            for b in labs[i + 1 :]:
                ba, ia = self._subset_indices[a]
                bb, ib = self._subset_indices[b]
                if ba == bb:
                    self.emit("dataset_overlap", a=a, b=b, overlap=len(ia & ib), size_a=len(ia), size_b=len(ib))
        self.emit("end", steps=self._opt_steps, seconds=round(time.time() - self._t0, 3))
        self.meta.setdefault("stopped_early", False)


class _LoaderIter:
    def __init__(self, it: Any, lab: str, tracer: Tracer) -> None:
        self._it = it
        self._lab = lab
        self._tracer = tracer

    def __iter__(self) -> "_LoaderIter":
        return self

    def __next__(self) -> Any:
        b = next(self._it)
        try:
            self._tracer._on_batch(self._lab, b)
        except Exception:
            pass
        return b

    def __len__(self) -> int:
        return len(self._it)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._it, name)
