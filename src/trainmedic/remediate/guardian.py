"""Guardian: online, invariant-triggered and root-cause-conditioned remediation.

Prior work rolls back on *loss/gradient statistics* (PaLM restarts ~100 steps
earlier and skips 200-500 batches; ZClip clips on gradient-norm z-scores;
production systems restart from checkpoints). The Guardian additionally
triggers on *semantic* invariants (replica divergence, non-finite optimizer
state, model left in eval mode, parameters not updating) and picks the action
from the suspected cause instead of always rolling back:

=========================  =============================  ===================================
trigger                    suspected cause                action
=========================  =============================  ===================================
non-finite gradient        transient overflow             skip the step (escalate to rollback
                                                          + LR backoff if it keeps happening)
loss spike / NaN loss      a bad batch                    roll back to the last snapshot and
                                                          skip only that batch
repeated spikes            optimisation instability       roll back *and* back off the LR
non-finite optimizer state corrupted state                restore optimizer state only
replica divergence         missed all-reduce / SDC        resync replicas (no rollback)
eval mode / no update      a code bug                     alert -- rolling back cannot fix code
=========================  =============================  ===================================
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import torch
import torch.nn as nn

from ..detect.distributed import param_checksum, replicas_consistent
from ..detect.numerics import EmaZScore


class Action(str, Enum):
    CONTINUE = "continue"
    SKIP_STEP = "skip_step"
    ROLLBACK = "rollback"
    RESTORE_OPTIMIZER = "restore_optimizer"
    RESYNC = "resync"
    ALERT = "alert"
    HALT = "halt"


@dataclass
class Decision:
    action: Action = Action.CONTINUE
    trigger: str | None = None
    cause: str | None = None
    reason: str = ""
    resume_step: int | None = None
    resume_batch: int | None = None
    skip_batches: list[Any] = field(default_factory=list)
    lr_scale: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action.value, "trigger": self.trigger, "cause": self.cause, "reason": self.reason,
                "resume_step": self.resume_step, "resume_batch": self.resume_batch, "skip_batches": self.skip_batches,
                "lr_scale": self.lr_scale}


@dataclass
class _Snapshot:
    resume_step: int
    next_batch: Any
    model: dict
    optimizer: dict
    scheduler: dict | None
    loss_det: dict
    grad_det: dict
    lr_multiplier: float
    replicas: list[dict] | None = None
    replica_opts: list[dict] | None = None


class RemediationPolicy:
    def __init__(self, *, max_rollbacks: int = 10, repeat_spikes_for_lr: int = 2, spike_window: int = 200,
                 lr_backoff: float = 0.5, max_nonfinite_skips: int = 3) -> None:
        self.max_rollbacks = max_rollbacks
        self.repeat_spikes_for_lr = repeat_spikes_for_lr
        self.spike_window = spike_window
        self.lr_backoff = lr_backoff
        self.max_nonfinite_skips = max_nonfinite_skips

    def decide(self, trigger: str, g: "Guardian", ctx: dict[str, Any]) -> Decision:
        step, batch = ctx.get("step"), ctx.get("batch")
        if trigger == "nonfinite_grad":
            recent = [s for s in g.nonfinite_steps if step is not None and step - s <= self.spike_window]
            if len(recent) > self.max_nonfinite_skips:
                return self._rollback(g, trigger, "lr_scale", "repeated non-finite gradients", [batch], self.lr_backoff)
            return Decision(Action.SKIP_STEP, trigger, "nonfinite_grad", "non-finite gradient: step skipped", skip_batches=[])
        if trigger in ("loss_spike", "nonfinite_loss"):
            recent = [s for s in g.spike_steps if step is not None and step - s <= self.spike_window]
            if len(recent) >= self.repeat_spikes_for_lr:
                return self._rollback(g, trigger, "lr_scale", f"{len(recent)} spikes within {self.spike_window} steps: "
                                      "optimisation instability, backing off the LR", [batch], self.lr_backoff)
            return self._rollback(g, trigger, "loss_spike", "isolated spike: rolling back and skipping the batch", [batch], 1.0)
        if trigger == "optimizer_state_nonfinite":
            return Decision(Action.RESTORE_OPTIMIZER, trigger, "optimizer_state_corrupt",
                            "optimizer state became non-finite: restoring it from the last snapshot", skip_batches=[batch])
        if trigger == "replica_divergence":
            return Decision(Action.RESYNC, trigger, "rank_divergence", "replicas diverged: resyncing from the majority")
        if trigger in ("train_mode_off", "params_not_updating"):
            cause = {"train_mode_off": "missing_train_mode", "params_not_updating": "params_not_updated"}[trigger]
            return Decision(Action.ALERT, trigger, cause, ctx.get("message", "code-level invariant violated; rollback cannot fix it"))
        return Decision()

    def _rollback(self, g, trigger, cause, reason, skip, lr_scale) -> Decision:
        if g.rollbacks >= self.max_rollbacks:
            return Decision(Action.HALT, trigger, cause, f"{reason}; rollback budget exhausted")
        return Decision(Action.ROLLBACK, trigger, cause, reason, skip_batches=[b for b in skip if b is not None], lr_scale=lr_scale)


class Guardian:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        *,
        snapshot_every: int = 50,
        keep: int = 3,
        loss_z: float = 6.0,
        warmup: int = 20,
        ema_alpha: float = 0.05,
        adaptive_clip: bool = False,
        clip_z: float = 3.0,
        policy: RemediationPolicy | None = None,
        replicas: list[nn.Module] | None = None,
        replica_optimizers: list[torch.optim.Optimizer] | None = None,
        check_code_invariants: bool = True,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.snapshot_every = snapshot_every
        self.snapshots: deque[_Snapshot] = deque(maxlen=keep)
        self.loss_det = EmaZScore(alpha=ema_alpha, warmup=warmup, z=loss_z)
        self.grad_det = EmaZScore(alpha=ema_alpha, warmup=warmup, z=clip_z)
        self.adaptive_clip = adaptive_clip
        self.policy = policy or RemediationPolicy()
        self.replicas = replicas
        self.replica_optimizers = replica_optimizers
        self.check_code = check_code_invariants
        self.on_event = on_event
        self.skip: set[Any] = set()
        self.events: list[dict[str, Any]] = []
        self.rollbacks = 0
        self.lr_multiplier = 1.0
        self.spike_steps: list[int] = []
        self.nonfinite_steps: list[int] = []
        self._alerted: set[str] = set()
        self._pre_checksum: float | None = None
        self._pending: Decision | None = None
        self._last_grad_norm = 0.0
        self._mode_sensitive = any(isinstance(m, (nn.modules.dropout._DropoutNd, nn.modules.batchnorm._BatchNorm))
                                   for m in model.modules())
        self.snapshot(resume_step=0, next_batch=0)

    # ------------------------------------------------------------ snapshots
    def snapshot(self, resume_step: int, next_batch: Any) -> None:
        def cpu(sd):
            return {k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)) for k, v in sd.items()}

        self.snapshots.append(_Snapshot(
            resume_step=resume_step, next_batch=next_batch, model=cpu(self.model.state_dict()),
            optimizer=copy.deepcopy(self.optimizer.state_dict()),
            scheduler=copy.deepcopy(self.scheduler.state_dict()) if self.scheduler is not None else None,
            loss_det=self.loss_det.state(), grad_det=self.grad_det.state(), lr_multiplier=self.lr_multiplier,
            replicas=[cpu(r.state_dict()) for r in self.replicas] if self.replicas else None,
            replica_opts=[copy.deepcopy(o.state_dict()) for o in self.replica_optimizers] if self.replica_optimizers else None,
        ))

    def _restore(self, snap: _Snapshot, optimizer_only: bool = False) -> None:
        if not optimizer_only:
            self.model.load_state_dict(snap.model)
            if self.scheduler is not None and snap.scheduler is not None:
                self.scheduler.load_state_dict(snap.scheduler)
            self.loss_det.load(snap.loss_det)
            self.grad_det.load(snap.grad_det)
            if self.replicas and snap.replicas:
                for r, sd in zip(self.replicas, snap.replicas):
                    r.load_state_dict(sd)
            if self.replica_optimizers and snap.replica_opts:
                for o, sd in zip(self.replica_optimizers, snap.replica_opts):
                    o.load_state_dict(sd)
        self.optimizer.load_state_dict(snap.optimizer)
        factor = self.lr_multiplier / snap.lr_multiplier
        if factor != 1.0:
            self._scale_lr(factor)

    def _scale_lr(self, factor: float) -> None:
        opts = [self.optimizer] + list(self.replica_optimizers or [])
        for opt in opts:
            for g in opt.param_groups:
                g["lr"] *= factor
        if self.scheduler is not None and hasattr(self.scheduler, "base_lrs"):
            self.scheduler.base_lrs = [lr * factor for lr in self.scheduler.base_lrs]

    # ------------------------------------------------------------------ api
    def should_skip(self, batch_id: Any) -> bool:
        return batch_id in self.skip

    def _grad_norm(self) -> float:
        sq = 0.0
        for g in self.optimizer.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    n = float(p.grad.detach().float().norm())
                    if not math.isfinite(n):
                        return float("nan")
                    sq += n * n
        return math.sqrt(sq)

    def pre_step(self, step: int | None = None, batch_id: Any = None) -> bool:
        """Call after backward(), before optimizer.step(). Returns False if the step must be skipped."""
        gn = self._grad_norm()
        self._last_grad_norm = gn
        self._pre_checksum = None
        if not math.isfinite(gn):
            self.nonfinite_steps.append(step or 0)
            d = self.policy.decide("nonfinite_grad", self, {"step": step, "batch": batch_id})
            self._record(step, d, grad_norm=gn)
            if d.action == Action.SKIP_STEP:
                self.optimizer.zero_grad(set_to_none=True)
                return False
            self._pending = d
            self.optimizer.zero_grad(set_to_none=True)
            return False
        if self.adaptive_clip:
            z, spike = self.grad_det.update(gn)
            if spike:
                std = math.sqrt(max(self.grad_det.var, 0.0))
                limit = self.grad_det.mean + self.grad_det.z * std
                coef = limit / (gn + 1e-12)
                for g in self.optimizer.param_groups:
                    for p in g["params"]:
                        if p.grad is not None:
                            p.grad.mul_(coef)
                self.events.append({"step": step, "action": "clip", "grad_norm": gn, "clipped_to": limit, "z": z})
        if self.check_code:
            self._pre_checksum = param_checksum(self.model)
        return True

    def post_step(self, step: int, loss: float | torch.Tensor, batch_id: Any = None) -> Decision:
        """Call after optimizer.step() (or after a skipped step). Returns the remediation decision."""
        loss = float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss)
        pending, self._pending = self._pending, None
        triggers: list[tuple[str, dict[str, Any]]] = []
        if pending is not None:
            d = pending
        else:
            if self.replicas is not None and not replicas_consistent(self.replicas)[0]:
                triggers.append(("replica_divergence", {}))
            if not self._optimizer_state_finite():
                triggers.append(("optimizer_state_nonfinite", {}))
            if not math.isfinite(loss):
                triggers.append(("nonfinite_loss", {}))
            else:
                z, spike = self.loss_det.update(loss)
                if spike:
                    triggers.append(("loss_spike", {"z": z}))
            if self.check_code:
                if self._mode_sensitive and not self.model.training and "train_mode_off" not in self._alerted:
                    triggers.append(("train_mode_off", {"message": "model is in eval mode during training steps (missing model.train())"}))
                if (self._pre_checksum is not None and self._last_grad_norm > 0 and "params_not_updating" not in self._alerted
                        and any(g["lr"] > 0 for g in self.optimizer.param_groups)
                        and param_checksum(self.model) == self._pre_checksum):
                    triggers.append(("params_not_updating", {"message": "optimizer.step() did not change any parameter"}))
            d = Decision()
            if triggers:
                trig, ctx = triggers[0]
                if trig in ("loss_spike", "nonfinite_loss"):
                    self.spike_steps.append(step)
                ctx.update(step=step, batch=batch_id)
                d = self.policy.decide(trig, self, ctx)
        self._apply(d, step)
        if d.action == Action.CONTINUE and self.snapshot_every and (step + 1) % self.snapshot_every == 0:
            nb = batch_id + 1 if isinstance(batch_id, int) else None
            self.snapshot(resume_step=step + 1, next_batch=nb)
        if d.action != Action.CONTINUE:
            self._record(step, d, loss=loss, batch=batch_id)
        return d

    def _apply(self, d: Decision, step: int) -> None:
        self.skip.update(d.skip_batches)
        if d.action == Action.ROLLBACK:
            snap = self.snapshots[-1]
            self.rollbacks += 1
            if d.lr_scale != 1.0:
                self.lr_multiplier *= d.lr_scale
            self._restore(snap)
            d.resume_step, d.resume_batch = snap.resume_step, snap.next_batch
        elif d.action == Action.RESTORE_OPTIMIZER:
            self._restore(self.snapshots[-1], optimizer_only=True)
        elif d.action == Action.RESYNC:
            self._resync()
        elif d.action == Action.ALERT and d.trigger:
            self._alerted.add(d.trigger)

    def _resync(self) -> None:
        if self.replicas:
            _, sums = replicas_consistent(self.replicas)
            votes: dict[float, list[int]] = {}
            for r, s in enumerate(sums):
                votes.setdefault(round(s, 6), []).append(r)
            src = max(votes.values(), key=len)[0]
            sd = copy.deepcopy(self.replicas[src].state_dict())
            osd = copy.deepcopy(self.replica_optimizers[src].state_dict()) if self.replica_optimizers else None
            for r, m in enumerate(self.replicas):
                if r != src:
                    m.load_state_dict(sd)
                    if osd is not None:
                        self.replica_optimizers[r].load_state_dict(osd)
            return
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            with torch.no_grad():
                for p in self.model.parameters():
                    dist.broadcast(p.data, src=0)

    def _optimizer_state_finite(self) -> bool:
        for i, st in enumerate(self.optimizer.state.values()):
            if i >= 256:
                break
            for v in st.values():
                if isinstance(v, torch.Tensor) and v.is_floating_point() and v.numel() and not bool(torch.isfinite(v).all()):
                    return False
        return True

    def _record(self, step: int | None, d: Decision, **extra: Any) -> None:
        ev = {"step": step, **d.to_dict(), **extra}
        self.events.append(ev)
        if self.on_event is not None:
            self.on_event(ev)

    def summary(self) -> dict[str, Any]:
        acts: dict[str, int] = {}
        for e in self.events:
            acts[e["action"]] = acts.get(e["action"], 0) + 1
        return {"rollbacks": self.rollbacks, "lr_multiplier": self.lr_multiplier, "skipped_batches": sorted(self.skip, key=str),
                "actions": acts, "alerts": sorted(self._alerted)}


def run_guarded(
    guardian: Guardian,
    get_batch: Callable[[int], Any],
    loss_fn: Callable[[Any], torch.Tensor],
    total_steps: int,
    *,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Reference training loop with rollback support (requires a deterministic,
    index-addressable data pipeline, as PaLM-style rollback-and-skip does)."""
    opt, sched = guardian.optimizer, guardian.scheduler
    step, pos = 0, 0
    history: list[dict[str, Any]] = []
    limit = max_batches or total_steps * 10
    while step < total_steps and pos < limit:
        bid = pos
        pos += 1
        if guardian.should_skip(bid):
            continue
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(get_batch(bid))
        loss.backward()
        if guardian.pre_step(step, bid):
            opt.step()
            if sched is not None:
                sched.step()
        d = guardian.post_step(step, loss, bid)
        history.append({"step": step, "batch": bid, "loss": float(loss.detach()), "action": d.action.value})
        if d.action == Action.HALT:
            break
        if d.action == Action.ROLLBACK:
            step = d.resume_step or 0
            pos = d.resume_batch if d.resume_batch is not None else pos
            continue
        step += 1
    return {"history": history, "steps": step, **guardian.summary()}
