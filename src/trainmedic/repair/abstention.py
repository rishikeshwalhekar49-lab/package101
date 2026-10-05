"""Abstention: when *not* to touch the code.

LLM repair agents readily "find" bugs in bug-free programs and their edits
damage correct code more often than they repair broken code ("If It's Not
Buggy, Don't Fix It", 2026). In DL training the damage is silent (a changed LR
schedule, removed dropout, altered loss reduction), so the policy here gates a
repair at three points:

1. before diagnosis: is there *actionable* evidence at all?
2. before validation: is each candidate patch grounded in that evidence, and
   free of collateral risky edits unrelated to the cited cause?
3. after validation: did the patch measurably help without breaking invariants?
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..types import Diagnosis, Evidence, Severity
from .patch import Patch

HYGIENE_RULES = {"TM016", "TM029", "TM030", "TM034", "TM004"}
HYGIENE_CAUSES = {"unseeded", "graph_retention", "hooks_bypassed", "init_scale"}
RISKY = {
    "removes_dropout", "removes_zero_grad", "removes_normalization", "removes_regularization", "changes_seed",
    "changes_data_split", "changes_metric_computation", "deletes_code", "changes_loss_reduction", "changes_lr",
    "removes_mode_switch",
}
ALLOWED_RISKS: dict[str, set[str]] = {
    "lr_scale": {"changes_lr"},
    "lr_schedule_misuse": {"changes_lr"},
    "no_learning": {"changes_lr"},
    "train_eval_leakage": {"changes_data_split"},
    "grad_accum_unscaled": {"changes_loss_reduction"},
    "grads_cleared_before_step": {"removes_zero_grad"},
    "eval_augmentation": {"deletes_code"},
    "fp16_no_loss_scaling": {"changes_loss_reduction"},
    "missing_eval_mode": {"removes_mode_switch"},
    "missing_train_mode": {"removes_mode_switch"},
    "crash": RISKY,
}


@dataclass
class AbstentionPolicy:
    min_evidence_confidence: float = 0.5
    min_cause_confidence: float = 0.5
    require_grounding: bool = True
    reject_collateral_risk: bool = True
    ignore_rules: set[str] = field(default_factory=lambda: set(HYGIENE_RULES))

    def actionable(self, evidence: list[Evidence]) -> list[Evidence]:
        return [
            e for e in evidence
            if e.confidence >= self.min_evidence_confidence
            and e.severity != Severity.INFO
            and e.rule not in self.ignore_rules
            and (e.cause or "") not in HYGIENE_CAUSES
        ]

    def gate_before_repair(self, evidence: list[Evidence], diagnosis: Diagnosis | None) -> tuple[bool, str]:
        act = self.actionable(evidence)
        if not act:
            return True, "no actionable evidence (all signals below the confidence threshold or hygiene-only)"
        if diagnosis is not None and not diagnosis.bug_found:
            return True, "diagnosis does not establish a defect"
        return False, f"{len(act)} actionable piece(s) of evidence"

    def gate_patch(self, patch: Patch, evidence: list[Evidence]) -> tuple[bool, str]:
        """Return (ok, reason) for a candidate patch before spending compute on it."""
        if not patch.edits:
            return False, "empty patch"
        act = {e.id: e for e in self.actionable(evidence)}
        cited = [act[i] for i in patch.evidence_ids if i in act]
        if self.require_grounding and not cited:
            return False, "ungrounded: the patch cites no actionable evidence"
        if self.reject_collateral_risk:
            allowed: set[str] = set()
            for e in cited:
                allowed |= ALLOWED_RISKS.get(e.cause or "", set())
            risky = (set(patch.risks()) & RISKY) - allowed
            if risky:
                return False, f"collateral risky edits unrelated to the cited cause: {', '.join(sorted(risky))}"
        return True, "grounded"
