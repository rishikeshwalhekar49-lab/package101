"""Training invariants checked over a :class:`~trainmedic.detect.trace.Trace`.

Two kinds of invariants exist:

* **universal** invariants hold for (almost) any correct PyTorch training loop
  (gradients are cleared between steps, the loss receives logits, evaluation
  runs in eval mode, ...). They are usable without any reference run.
* **inferred** invariants are learned from traces of *healthy* reference runs, in
  the spirit of TrainCheck (OSDI'25): an invariant is kept only if it holds on
  every reference trace, and numeric invariants fit their thresholds to the
  references. Universal invariants that a healthy reference violates are
  dropped, which is what keeps false alarms (and thus damage) low on unusual
  but correct pipelines.

The violated invariant later serves two more purposes: it is the context given
to the diagnoser, and it is the acceptance oracle for candidate patches (a patch
must make the violated invariant hold without breaking the others).
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, ClassVar

from ..types import Evidence, Severity
from .trace import Trace

REGISTRY: dict[str, type["Invariant"]] = {}


def register(cls: type["Invariant"]) -> type["Invariant"]:
    REGISTRY[cls.name] = cls
    return cls


def classify_forwards(trace: Trace) -> dict[int, str]:
    """Label each root forward as 'train' (feeds a backward), 'eval' (no grad) or 'other'."""
    roles: dict[int, str] = {}
    pending: list[dict[str, Any]] = []
    for e in trace.events:
        k = e["kind"]
        if k == "forward":
            if not e.get("grad_enabled", True):
                roles[e["i"]] = "eval"
            else:
                pending.append(e)
        elif k == "backward":
            if pending:
                loader = pending[-1].get("loader")
                for f in pending:
                    roles[f["i"]] = "train" if f.get("loader") == loader else roles.get(f["i"], "other")
            pending = []
        elif k in ("opt_step", "mode"):
            for f in pending:
                roles.setdefault(f["i"], "other")
            pending = []
    for f in pending:
        roles.setdefault(f["i"], "other")
    return roles


def _forwards(trace: Trace, role: str) -> list[dict[str, Any]]:
    roles = classify_forwards(trace)
    return [e for e in trace.of("forward") if roles.get(e["i"]) == role]


def _loaders_by_role(trace: Trace) -> tuple[set[str], set[str]]:
    roles = classify_forwards(trace)
    train, ev = set(), set()
    for e in trace.of("forward"):
        if e.get("loader") is None:
            continue
        r = roles.get(e["i"])
        if r == "train":
            train.add(e["loader"])
        elif r == "eval":
            ev.add(e["loader"])
    return train, ev - train


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


class Invariant:
    name: ClassVar[str] = "invariant"
    category: ClassVar[str] = "general"
    cause: ClassVar[str] = "unknown"
    description: ClassVar[str] = ""
    universal: ClassVar[bool] = True
    base_confidence: ClassVar[float] = 0.6
    severity: ClassVar[Severity] = Severity.WARNING

    def __init__(self, params: dict[str, Any] | None = None, inferred: bool = False, confidence: float | None = None) -> None:
        self.params = params or {}
        self.inferred = inferred
        self.confidence = confidence if confidence is not None else self.base_confidence

    # override -------------------------------------------------------------
    def violations(self, trace: Trace) -> list[tuple[str, dict[str, Any]]]:
        """Return (message, data) pairs. ``data`` may include ``step``."""
        raise NotImplementedError

    @classmethod
    def infer(cls, traces: list[Trace]) -> "Invariant | None":
        inst = cls()
        if all(not inst.violations(t) for t in traces):
            inst.inferred = True
            inst.confidence = max(inst.confidence, min(0.95, inst.base_confidence + 0.2))
            return inst
        return None

    # common ---------------------------------------------------------------
    def check(self, trace: Trace) -> list[Evidence]:
        out = []
        for msg, data in self.violations(trace):
            conf = data.pop("confidence", self.confidence)
            out.append(
                Evidence(
                    id="", source="invariant", rule=self.name, message=msg,
                    severity=data.pop("severity", self.severity), confidence=conf,
                    step=data.get("step"), category=self.category, cause=data.pop("cause", self.cause),
                    data={**data, "inferred": self.inferred},
                )
            )
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params, "inferred": self.inferred, "confidence": self.confidence}

    def __repr__(self) -> str:
        return f"<{self.name}{' (inferred)' if self.inferred else ''} {self.params}>"


# ============================================================ gradient lifecycle
@register
class GradsClearedBetweenSteps(Invariant):
    name = "grads_cleared_between_steps"
    category = "optimization"
    cause = "missing_zero_grad"
    description = "Each optimizer's gradients are cleared between consecutive steps (zero_grad or set to None)."
    base_confidence = 0.9
    severity = Severity.ERROR

    def violations(self, trace):
        by_opt: dict[str, list[int]] = {}
        for e in trace.of("opt_step"):
            by_opt.setdefault(e["opt"], []).append(e["i"])
        zero_events = trace.of("zero_grad")
        clean_bw = [e for e in trace.of("backward") if e.get("grads_clean")]
        out = []
        for opt, idxs in by_opt.items():
            bad = []
            for a, b in zip(idxs, idxs[1:]):
                cleared = any(a < z["i"] < b and (z["owner"] == opt or z["owner_kind"] == "module") for z in zero_events)
                cleared = cleared or any(a < w["i"] < b for w in clean_bw)
                if not cleared:
                    bad.append(a)
            n = len(idxs) - 1
            if n >= 2 and len(bad) >= 2 and len(bad) / n > 0.5:
                step = next(e["step"] for e in trace.events if e["i"] == bad[0])
                out.append((f"optimizer {opt}: gradients were never cleared between {len(bad)}/{n} consecutive steps, "
                            f"so each update applies the sum of all previous gradients", {"step": step, "opt": opt, "fraction": len(bad) / n}))
        return out


@register
class GradientsPresentAtStep(Invariant):
    name = "gradients_present_at_step"
    category = "optimization"
    cause = "grads_cleared_before_step"
    description = "When backward() ran since the last step, the optimizer sees non-zero gradients."
    base_confidence = 0.9
    severity = Severity.ERROR

    def violations(self, trace):
        pres = [e for e in trace.of("opt_pre") if e.get("backwards_since_step", 0) > 0 and e.get("n_params", 0) > 0]
        bad = [e for e in pres if e.get("n_nonzero_grad", 0) == 0]
        if len(pres) >= 2 and len(bad) / len(pres) > 0.5:
            return [(f"{len(bad)}/{len(pres)} optimizer steps saw only empty gradients even though backward() ran; "
                     "gradients are cleared between backward() and step()", {"step": bad[0]["step"]})]
        return []


@register
class BackwardBeforeStep(Invariant):
    name = "backward_before_step"
    category = "optimization"
    cause = "step_without_backward"
    description = "Every optimizer step is preceded by at least one backward pass since the previous step."
    base_confidence = 0.7

    def violations(self, trace):
        lbfgs = {e["opt"] for e in trace.of("optimizer") if e.get("cls") == "LBFGS"}
        pres = [e for e in trace.of("opt_pre") if e["opt"] not in lbfgs]
        seen: set[str] = set()
        bad = []
        for e in pres:
            if e["opt"] in seen and e.get("backwards_since_step", 1) == 0:
                bad.append(e)
            seen.add(e["opt"])
        if len(bad) >= 2:
            return [(f"{len(bad)} optimizer steps had no new backward() since the previous step (stale gradients reapplied)",
                     {"step": bad[0]["step"]})]
        return []


@register
class ParamsUpdated(Invariant):
    name = "params_updated_by_step"
    category = "optimization"
    cause = "params_not_updated"
    description = "A step with non-zero gradients and positive LR changes the parameters."
    base_confidence = 0.85
    severity = Severity.ERROR

    def violations(self, trace):
        bad = total = 0
        first = None
        last_pre = None
        for e in trace.events:
            if e["kind"] == "opt_pre":
                last_pre = e
            elif e["kind"] == "opt_step" and last_pre is not None:
                if last_pre.get("n_nonzero_grad", 0) > 0 and any((lr or 0) > 0 for lr in last_pre.get("lrs", [])):
                    total += 1
                    if e.get("n_changed", 1) == 0:
                        bad += 1
                        first = first or e
                last_pre = None
        if total >= 2 and bad / total > 0.5:
            return [(f"{bad}/{total} steps did not change any parameter despite non-zero gradients and LR > 0",
                     {"step": first["step"] if first else None})]
        return []


@register
class TrainableParamsReceiveGrad(Invariant):
    name = "trainable_params_receive_grad"
    category = "graph"
    cause = "disconnected_params"
    description = "Every parameter handed to the optimizer receives a non-zero gradient at least once."
    base_confidence = 0.55

    @staticmethod
    def _never(trace: Trace) -> set[str]:
        pres = trace.of("opt_pre")
        if not pres:
            return set()
        sets = [set(p.replace(" (zero grad)", "") for p in e.get("params_without_grad", [])) for e in pres]
        never = set.intersection(*sets) if sets else set()
        return never

    def violations(self, trace):
        if trace.n_steps < 3:
            return []
        allowed = set(self.params.get("allowed", []))
        never = self._never(trace) - allowed
        if never:
            names = sorted(never)
            return [(f"{len(names)} optimized parameter tensor(s) never received a gradient: {', '.join(names[:5])}"
                     f"{' ...' if len(names) > 5 else ''} (graph is disconnected: detach, no_grad or unused layer)",
                     {"step": 0, "params": names})]
        return []

    @classmethod
    def infer(cls, traces):
        allowed: set[str] = set()
        for t in traces:
            allowed |= cls._never(t)
        return cls({"allowed": sorted(allowed)}, inferred=True, confidence=0.85)


@register
class OptimizerCoversTrainableParams(Invariant):
    name = "optimizer_covers_trainable_params"
    category = "optimization"
    cause = "optimizer_coverage"
    description = "Parameters of trained models that require grad are handed to an optimizer."
    base_confidence = 0.5

    @staticmethod
    def _missing(trace: Trace) -> set[str]:
        trained = {e["module"] for e in _forwards(trace, "train")}
        out: set[str] = set()
        for e in trace.of("coverage"):
            for p in e.get("trainable_not_in_optimizer", []):
                if p.split(".", 1)[0] in trained:
                    out.add(p)
        return out

    def violations(self, trace):
        miss = self._missing(trace) - set(self.params.get("allowed", []))
        if miss:
            names = sorted(miss)
            return [(f"{len(names)} trainable parameter tensor(s) of a trained model are not in any optimizer: "
                     f"{', '.join(names[:5])}{' ...' if len(names) > 5 else ''}", {"step": 0, "params": names})]
        return []

    @classmethod
    def infer(cls, traces):
        allowed: set[str] = set()
        for t in traces:
            allowed |= cls._missing(t)
        return cls({"allowed": sorted(allowed)}, inferred=True, confidence=0.8)


@register
class GradAccumulationScaled(Invariant):
    name = "grad_accumulation_scaled"
    category = "optimization"
    cause = "grad_accum_unscaled"
    description = "With k backward passes per step, each backpropagated loss is divided by k."
    base_confidence = 0.7
    severity = Severity.ERROR

    def violations(self, trace):
        windows = [e for e in trace.of("opt_pre") if e.get("backwards_in_window", 1) >= 2]
        if len(windows) < 2:
            return []
        ks = [e["backwards_in_window"] for e in windows]
        k = statistics.median(ks)
        ratios = [e["value"] / e["loss_value"] for e in trace.of("backward")
                  if _finite(e.get("value")) and _finite(e.get("loss_value")) and abs(e["loss_value"]) > 1e-12]
        if len(ratios) < 4:
            return []
        r = statistics.median(ratios)
        if abs(r - 1.0) < 0.05 and k >= 2:
            return [(f"gradients are accumulated over ~{k:g} backward passes per step but the loss is not divided by {k:g} "
                     f"(backpropagated value / loss value = {r:.2f}); the effective step is {k:g}x too large",
                     {"step": windows[0]["step"], "k": k})]
        return []


# ================================================================ model mode
@register
class TrainModeDuringTraining(Invariant):
    name = "train_mode_during_training"
    category = "model_mode"
    cause = "missing_train_mode"
    description = "Models with dropout/batch-norm are in train() mode when their output is backpropagated."
    base_confidence = 0.85
    severity = Severity.ERROR

    def violations(self, trace):
        fw = [e for e in _forwards(trace, "train") if e.get("mode_sensitive")]
        bad = [e for e in fw if not e.get("training", True)]
        if fw and len(bad) >= max(1, 0.1 * len(fw)):
            return [(f"{len(bad)}/{len(fw)} training forward passes of {bad[0]['cls']} ran in eval mode "
                     "(dropout off, batch-norm statistics frozen); model.train() is missing after evaluation",
                     {"step": bad[0]["step"], "module": bad[0]["module"]})]
        return []


@register
class EvalModeDuringEvaluation(Invariant):
    name = "eval_mode_during_evaluation"
    category = "model_mode"
    cause = "missing_eval_mode"
    description = "Models with dropout/batch-norm are in eval() mode when evaluated under no_grad()."
    base_confidence = 0.75

    def violations(self, trace):
        fw = [e for e in _forwards(trace, "eval") if e.get("mode_sensitive") and e.get("trainable", True)]
        bad = [e for e in fw if e.get("training")]
        if fw and len(bad) / len(fw) > 0.5:
            return [(f"{len(bad)}/{len(fw)} no-grad (evaluation) forward passes of {bad[0]['cls']} ran in train mode; "
                     "model.eval() is missing, so dropout is active and batch-norm uses batch statistics",
                     {"step": bad[0]["step"], "module": bad[0]["module"]})]
        return []


# ======================================================================= loss
@register
class CrossEntropyGetsLogits(Invariant):
    name = "cross_entropy_gets_logits"
    category = "loss"
    cause = "double_softmax"
    description = "Inputs to cross_entropy are unnormalised logits, not probabilities."
    base_confidence = 0.9
    severity = Severity.ERROR

    def violations(self, trace):
        ev = [e for e in trace.of("loss") if e["fn"] == "cross_entropy" and e.get("n_classes", 0) >= 2]
        bad = [e for e in ev if e.get("in01") and _finite(e.get("rowsum_err")) and e["rowsum_err"] < 1e-3]
        if ev and len(bad) / len(ev) > 0.5:
            return [(f"cross_entropy received probabilities (rows sum to 1) in {len(bad)}/{len(ev)} calls; "
                     "softmax is applied twice", {"step": bad[0]["step"]})]
        return []


@register
class NLLGetsLogProbs(Invariant):
    name = "nll_gets_log_probs"
    category = "loss"
    cause = "nll_without_log_softmax"
    description = "Inputs to nll_loss are log-probabilities (non-positive, logsumexp = 0)."
    base_confidence = 0.85
    severity = Severity.ERROR

    def violations(self, trace):
        ev = [e for e in trace.of("loss") if e["fn"] == "nll_loss" and e.get("n_classes", 0) >= 2]
        bad = [e for e in ev if not e.get("all_nonpos") or (_finite(e.get("lse_err")) and e["lse_err"] > 1e-2)]
        if ev and len(bad) / len(ev) > 0.5:
            return [(f"nll_loss received values that are not log-probabilities in {len(bad)}/{len(ev)} calls; "
                     "apply log_softmax first", {"step": bad[0]["step"]})]
        return []


@register
class BCEWithLogitsGetsLogits(Invariant):
    name = "bce_with_logits_gets_logits"
    category = "loss"
    cause = "double_sigmoid"
    description = "Inputs to binary_cross_entropy_with_logits are logits, not sigmoid probabilities."
    base_confidence = 0.75
    severity = Severity.ERROR

    def violations(self, trace):
        ev = [e for e in trace.of("loss") if e["fn"] == "binary_cross_entropy_with_logits"]
        bad = [e for e in ev if e.get("in01")]
        if len(ev) >= 3 and len(bad) / len(ev) > 0.9:
            return [(f"binary_cross_entropy_with_logits received values in [0, 1] in {len(bad)}/{len(ev)} calls; "
                     "sigmoid is applied twice", {"step": bad[0]["step"]})]
        return []


# ================================================================== numerics
@register
class FiniteLoss(Invariant):
    name = "finite_loss"
    category = "numerics"
    cause = "nonfinite_loss"
    description = "Backpropagated losses are finite."
    base_confidence = 0.95
    severity = Severity.ERROR

    def violations(self, trace):
        bad = [e for e in trace.of("backward") if "value" in e and not _finite(e["value"])]
        if bad:
            return [(f"loss became non-finite ({bad[0]['value']}) at step {bad[0]['step']} ({len(bad)} backward passes affected)",
                     {"step": bad[0]["step"]})]
        return []


@register
class FiniteGradients(Invariant):
    name = "finite_gradients"
    category = "numerics"
    cause = "nonfinite_grad"
    description = "Gradients are finite at optimizer steps (occasional overflow is tolerated, as with GradScaler)."
    base_confidence = 0.85
    severity = Severity.ERROR

    def violations(self, trace):
        pres = trace.of("opt_pre")
        bad = [e for e in pres if e.get("grads_finite") is False]
        if pres and bad and len(bad) / len(pres) > 0.2:
            return [(f"{len(bad)}/{len(pres)} optimizer steps had non-finite gradients", {"step": bad[0]["step"]})]
        return []


@register
class OptimizerStateFinite(Invariant):
    name = "optimizer_state_finite"
    category = "numerics"
    cause = "optimizer_state_corrupt"
    description = "Optimizer state (momentum, second moments) stays finite."
    base_confidence = 0.9
    severity = Severity.ERROR

    def violations(self, trace):
        bad = [e for e in trace.of("opt_step") if e.get("optimizer_state_finite") is False]
        if bad:
            return [(f"optimizer state contains NaN/Inf from step {bad[0]['step']}", {"step": bad[0]["step"]})]
        return []


@register
class UpdateRatioSane(Invariant):
    name = "update_ratio_sane"
    category = "optimization"
    cause = "lr_scale"
    description = "The relative update size ||dW||/||W|| stays in a plausible range."
    base_confidence = 0.45

    def violations(self, trace):
        rs = trace.series("opt_step", "update_ratio")
        if len(rs) < 3:
            return []
        med = statistics.median(rs)
        lo, hi = self.params.get("lo", 1e-8), self.params.get("hi", 0.2)
        mx, max_hi = max(rs), self.params.get("max_hi", 1.0)
        if mx > max_hi:
            step = next(i for i, r in enumerate(rs) if r == mx)
            return [(f"a single optimizer step changed the parameters by {mx:.3g}x their norm (limit {max_hi:.3g}); "
                     "the learning rate is far too high", {"step": step, "max": mx, "confidence": max(self.confidence, 0.65)})]
        if med > hi:
            return [(f"median relative update ||dW||/||W|| = {med:.3g} exceeds {hi:.3g}; the learning rate is likely too high",
                     {"step": 0, "median": med})]
        if 0 < med < lo:
            return [(f"median relative update ||dW||/||W|| = {med:.3g} is below {lo:.3g}; the model barely changes "
                     "(learning rate too low or gradients vanish)", {"step": 0, "median": med})]
        return []

    @classmethod
    def infer(cls, traces):
        series = [t.series("opt_step", "update_ratio") for t in traces]
        series = [s for s in series if len(s) >= 3]
        if not series:
            return None
        meds = [statistics.median(s) for s in series]
        maxes = [max(s) for s in series]
        return cls({"lo": min(meds) / 10, "hi": max(meds) * 10, "max_hi": max(1.0, max(maxes) * 10)}, inferred=True, confidence=0.75)


@register
class LossDoesNotCollapse(Invariant):
    name = "loss_not_trivially_solved"
    category = "data"
    cause = "target_leakage"
    description = "Classification/LM loss does not collapse to ~0 within the first few steps (target leakage)."
    base_confidence = 0.5

    def violations(self, trace):
        ce = [e for e in trace.of("loss") if e["fn"] in ("cross_entropy", "nll_loss") and e.get("n_classes", 0) >= 4]
        ls = trace.step_losses()
        if not ce or len(ls) < 10:
            return []
        c = ce[0]["n_classes"]
        init, expected = ls[0], math.log(c)
        if not (0.5 * expected <= init <= 2.0 * expected):
            return []
        early = ls[: min(len(ls), 30)]
        if min(early) < 0.01 * expected:
            i = next(j for j, v in enumerate(early) if v < 0.01 * expected)
            return [(f"cross-entropy over {c} classes fell from {init:.2f} to {min(early):.4f} within {i + 1} steps; "
                     "the objective is suspiciously easy (targets may leak into inputs, e.g. unshifted LM labels)",
                     {"step": i})]
        return []


@register
class InitialLossPlausible(Invariant):
    name = "initial_loss_plausible"
    category = "numerics"
    cause = "init_scale"
    description = "Cross-entropy at initialisation is close to ln(num_classes)."
    base_confidence = 0.35
    severity = Severity.INFO

    def violations(self, trace):
        ce = [e for e in trace.of("loss") if e["fn"] == "cross_entropy" and e.get("n_classes", 0) >= 2 and _finite(e.get("value"))]
        if not ce:
            return []
        v, c = ce[0]["value"], ce[0]["n_classes"]
        hi = self.params.get("max_ratio", 3.0)
        if v > hi * math.log(c):
            return [(f"initial cross-entropy {v:.2f} is {v / math.log(c):.1f}x ln({c}) = {math.log(c):.2f}; "
                     "logits are badly scaled at initialisation", {"step": 0})]
        return []


@register
class NoLossSpikes(Invariant):
    name = "no_loss_spikes"
    category = "numerics"
    cause = "loss_spike"
    description = "The training loss has no sudden spikes (EMA z-score above the threshold)."
    base_confidence = 0.4

    def violations(self, trace):
        from .numerics import spike_indices

        ls = trace.step_losses()
        z = self.params.get("z", 6.0)
        spikes = spike_indices(ls, z=z)
        if spikes:
            return [(f"{len(spikes)} loss spike(s) (z > {z:g}) starting at step {spikes[0]}; "
                     "consider gradient clipping, a lower LR, or Guardian rollback", {"step": spikes[0], "spikes": spikes[:20]})]
        return []


# ======================================================================= data
@register
class TrainingDataShuffled(Invariant):
    name = "training_data_shuffled"
    category = "data"
    cause = "no_shuffle"
    description = "Training batches mix classes (data is shuffled)."
    base_confidence = 0.8

    def violations(self, trace):
        train, _ = _loaders_by_role(trace)
        info = {e["loader"]: e for e in trace.of("loader")}
        stats = {e["loader"]: e for e in trace.of("loader_stats")}
        out = []
        for lab in train:
            st = stats.get(lab, {})
            mu, tot = st.get("mean_unique_per_batch"), st.get("labels_total")
            bs = info.get(lab, {}).get("batch_size") or 0
            if mu is None or not tot or tot < 3 or st.get("batches", 0) < 3:
                continue
            expected = min(tot, bs) if bs else tot
            if mu < 0.35 * expected:
                out.append((f"training loader {lab} ({info.get(lab, {}).get('sampler')}) yields batches with only "
                            f"{mu:.1f} distinct labels on average out of {tot}; the data is ordered and not shuffled",
                            {"step": 0, "loader": lab}))
        return out


@register
class NoTrainEvalOverlap(Invariant):
    name = "no_train_eval_overlap"
    category = "data"
    cause = "train_eval_leakage"
    description = "Evaluation data is disjoint from training data."
    base_confidence = 0.85
    severity = Severity.ERROR

    def violations(self, trace):
        train, ev = _loaders_by_role(trace)
        info = {e["loader"]: e for e in trace.of("loader")}
        out = []
        for e in ev:
            for t in train:
                if info.get(e, {}).get("dataset") and info.get(e, {}).get("dataset") == info.get(t, {}).get("dataset"):
                    out.append((f"evaluation loader {e} iterates the same dataset object as training loader {t}",
                                {"step": 0, "loaders": [e, t]}))
        for o in trace.of("dataset_overlap"):
            pair = {o["a"], o["b"]}
            if pair & train and pair & ev and o["overlap"] > 0:
                frac = o["overlap"] / max(1, min(o["size_a"], o["size_b"]))
                out.append((f"{o['overlap']} samples ({frac:.0%}) appear in both a training and an evaluation subset",
                            {"step": 0, "overlap": o["overlap"]}))
        return out


@register
class InputDistributionConsistent(Invariant):
    name = "input_distribution_consistent"
    category = "data"
    cause = "preprocessing_mismatch"
    description = "Float inputs seen in evaluation have a similar scale to those seen in training."
    base_confidence = 0.65

    def violations(self, trace):
        tr = [e["input"] for e in _forwards(trace, "train") if _finite(e.get("input", {}).get("mean"))]
        ev = [e["input"] for e in _forwards(trace, "eval") if _finite(e.get("input", {}).get("mean"))]
        if len(tr) < 2 or len(ev) < 2:
            return []
        mt = statistics.mean(x["mean"] for x in tr)
        st = statistics.mean(x.get("std", 0) or 0 for x in tr)
        me = statistics.mean(x["mean"] for x in ev)
        se = statistics.mean(x.get("std", 0) or 0 for x in ev)
        k = self.params.get("k", 2.0)
        shift = abs(me - mt) > 0.5 * max(st, 1e-6) and abs(me - mt) > 1e-3
        scale = st > 0 and se > 0 and (se / st > k or st / se > k)
        if shift or scale:
            return [(f"evaluation inputs (mean {me:.3g}, std {se:.3g}) differ from training inputs (mean {mt:.3g}, std {st:.3g}); "
                     "preprocessing/normalisation differs between training and evaluation", {"step": 0})]
        return []


# ===================================================================== probes
@register
class EvalModeDeterministic(Invariant):
    name = "eval_mode_deterministic"
    category = "model_mode"
    cause = "dropout_in_eval"
    description = "In eval mode a model's output does not depend on the random state."
    base_confidence = 0.85
    severity = Severity.ERROR

    def violations(self, trace):
        bad = [e for e in trace.of("probe") if e.get("probe") == "eval_determinism" and not e.get("passed", True)]
        return [(f"{e['module']}: {e.get('detail')} (relative difference {e.get('score', 0):.3g}); "
                 "dropout is active in eval mode (e.g. F.dropout without training=self.training)", {"step": e["step"], "module": e["module"]})
                for e in bad]


@register
class BatchIndependence(Invariant):
    name = "batch_independence"
    category = "model_structure"
    cause = "batch_mixing"
    description = "In eval mode, the output for one sample does not depend on other samples in the batch."
    base_confidence = 0.8
    severity = Severity.ERROR

    def violations(self, trace):
        bad = [e for e in trace.of("probe") if e.get("probe") == "batch_independence" and not e.get("passed", True)]
        return [(f"{e['module']}: {e.get('detail')} (leak ratio {e.get('score', 0):.3g}); a reshape/view/permute mixes the batch dimension",
                 {"step": e["step"], "module": e["module"]}) for e in bad]


@register
class Causality(Invariant):
    name = "causality"
    category = "attention"
    cause = "future_leakage"
    description = "Sequence-model outputs at position t do not depend on inputs after t."
    base_confidence = 0.65
    severity = Severity.ERROR

    def violations(self, trace):
        out = []
        for e in trace.of("probe"):
            if e.get("probe") == "causality" and not e.get("passed", True):
                conf = self.confidence if (self.inferred or e.get("input_kind") == "token_sequence") else 0.35
                out.append((f"{e['module']}: {e.get('detail')} (relative change {e.get('score', 0):.3g}); the causal attention "
                            "mask is missing or inverted (expected for bidirectional encoders)",
                            {"step": e["step"], "module": e["module"], "confidence": conf}))
        return out


# =============================================================== distributed
@register
class RanksConsistent(Invariant):
    name = "ranks_consistent"
    category = "distributed"
    cause = "rank_divergence"
    description = "Data-parallel replicas hold identical parameters after each step."
    base_confidence = 0.9
    severity = Severity.ERROR

    def violations(self, trace):
        for e in trace.of("rank_checksums"):
            cols = list(zip(*e["checksums"]))
            for col in cols:
                ref = col[0]
                if any(abs(v - ref) > 1e-6 * max(1.0, abs(ref)) for v in col[1:]):
                    return [(f"parameters diverged across data-parallel ranks at step {e['step']} (checksums {list(col)})",
                             {"step": e["step"]})]
        return []


# ========================================================== reference-only
@register
class SchedulerCadence(Invariant):
    name = "scheduler_cadence"
    category = "optimization"
    cause = "lr_schedule_misuse"
    description = "LR schedulers are stepped at the same cadence (per batch / per epoch) as in healthy runs."
    universal = False
    base_confidence = 0.75

    @staticmethod
    def _stats(trace: Trace) -> dict[str, Any]:
        ev = trace.of("sched_step")
        steps = max(1, trace.n_steps)
        return {"ratio": len(ev) / steps, "first_at": ev[0]["opt_steps"] if ev else None, "n": len(ev)}

    def violations(self, trace):
        ref_ratio, first_at = self.params.get("ratio"), self.params.get("first_at")
        if ref_ratio is None or first_at is None:
            return []
        st = self._stats(trace)
        if trace.n_steps < first_at:
            return []
        if st["n"] == 0:
            return [("the LR scheduler is never stepped, but healthy runs step it "
                     f"{ref_ratio:.3g} times per optimizer step", {"step": first_at})]
        r = st["ratio"] / ref_ratio
        if r > 3 or r < 1 / 3:
            return [(f"the LR scheduler is stepped {st['ratio']:.3g} times per optimizer step vs {ref_ratio:.3g} in healthy runs "
                     "(per-batch vs per-epoch mix-up)", {"step": st["first_at"] or 0})]
        return []

    @classmethod
    def infer(cls, traces):
        stats = [cls._stats(t) for t in traces]
        if not stats or any(s["n"] == 0 for s in stats):
            return None
        return cls({"ratio": statistics.median(s["ratio"] for s in stats),
                    "first_at": max(s["first_at"] for s in stats)}, inferred=True, confidence=0.75)


@register
class GradNormRange(Invariant):
    name = "grad_norm_range"
    category = "numerics"
    cause = "grad_scale"
    description = "The median gradient norm is within the range seen in healthy runs."
    universal = False
    base_confidence = 0.6

    def violations(self, trace):
        gn = trace.series("opt_pre", "grad_norm")
        if len(gn) < 3 or "lo" not in self.params:
            return []
        med = statistics.median(gn)
        if med < self.params["lo"] or med > self.params["hi"]:
            return [(f"median gradient norm {med:.3g} is outside the healthy range [{self.params['lo']:.3g}, {self.params['hi']:.3g}]",
                     {"step": 0, "median": med})]
        return []

    @classmethod
    def infer(cls, traces):
        meds = [statistics.median(t.series("opt_pre", "grad_norm")) for t in traces if len(t.series("opt_pre", "grad_norm")) >= 3]
        if not meds:
            return None
        return cls({"lo": min(meds) / 30, "hi": max(meds) * 30}, inferred=True)


@register
class LossProgress(Invariant):
    name = "loss_progress"
    category = "optimization"
    cause = "no_learning"
    description = "Training loss falls by a similar factor as in healthy runs over the same number of steps."
    universal = False
    base_confidence = 0.55

    @staticmethod
    def _ratio(trace: Trace, n: int | None = None) -> float | None:
        ls = trace.step_losses()
        if n is not None:
            ls = ls[:n]
        if len(ls) < 6:
            return None
        k = max(1, len(ls) // 5)
        a, b = statistics.mean(ls[:k]), statistics.mean(ls[-k:])
        return b / a if a > 0 else None

    def violations(self, trace):
        ref, n = self.params.get("ratio"), self.params.get("steps")
        r = self._ratio(trace, n)
        if ref is None or r is None or not math.isfinite(r):
            return []
        if r > max(ref * 3, ref + 0.3):
            return [(f"training loss fell to {r:.0%} of its initial value vs {ref:.0%} in healthy runs over {n} steps; "
                     "training makes far less progress", {"step": 0})]
        return []

    @classmethod
    def infer(cls, traces):
        n = min((len(t.step_losses()) for t in traces), default=0)
        rs = [cls._ratio(t, n) for t in traces]
        rs = [r for r in rs if r is not None]
        if not rs:
            return None
        return cls({"ratio": max(rs), "steps": n}, inferred=True)


# ================================================================== sets
class InvariantSet:
    def __init__(self, invariants: list[Invariant]) -> None:
        self.invariants = invariants

    @classmethod
    def default(cls) -> "InvariantSet":
        return cls([c() for c in REGISTRY.values() if c.universal])

    @classmethod
    def infer(cls, traces: list[Trace], keep_universal_if_untested: bool = True) -> "InvariantSet":
        """Learn invariants from healthy traces (all must satisfy each kept invariant)."""
        out: list[Invariant] = []
        for c in REGISTRY.values():
            inst = c.infer(traces)
            if inst is not None:
                out.append(inst)
        return cls(out)

    def check(self, trace: Trace) -> list[Evidence]:
        ev: list[Evidence] = []
        for inv in self.invariants:
            try:
                ev.extend(inv.check(trace))
            except Exception as exc:  # never let one invariant break checking
                ev.append(Evidence(id="", source="invariant", rule=inv.name, message=f"internal error: {exc}",
                                   severity=Severity.INFO, confidence=0.0, category="internal"))
        if trace.meta.get("error"):
            ev.append(Evidence(id="", source="runtime_error", rule="crash", message=str(trace.meta["error"])[:500],
                               severity=Severity.ERROR, confidence=1.0, category="crash", cause="crash"))
        return ev

    def names(self) -> list[str]:
        return [i.name for i in self.invariants]

    def to_list(self) -> list[dict[str, Any]]:
        return [i.to_dict() for i in self.invariants]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_list(), indent=2))

    @classmethod
    def from_list(cls, items: list[dict[str, Any]]) -> "InvariantSet":
        out = []
        for d in items:
            c = REGISTRY.get(d["name"])
            if c is not None:
                out.append(c(d.get("params"), d.get("inferred", False), d.get("confidence")))
        return cls(out)

    @classmethod
    def load(cls, path: str | Path) -> "InvariantSet":
        return cls.from_list(json.loads(Path(path).read_text()))

    def __len__(self) -> int:
        return len(self.invariants)


def check_trace(trace: Trace, invariants: InvariantSet | None = None) -> list[Evidence]:
    return (invariants or InvariantSet.default()).check(trace)
