"""Rules for RL / preference post-training code (PPO, GRPO, DPO): KL-penalty
sign errors, trainable reference models and un-exponentiated policy ratios."""

from __future__ import annotations

import ast
import re

from ...types import Severity
from .analysis import call_name, const_value, dotted, last_part, method, node_key, receiver
from .engine import LintContext, fixer, rule

KL_RE = re.compile(r"(^|[_.])kl($|[_.\d])|kl_div|kl_coef|kl_penalty|kl_loss|kl_ctl|^kl", re.I)
REDUCERS = {"mean", "sum", "masked_mean", "masked_sum"}


def _mentions_kl(ctx: LintContext, node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and KL_RE.search(n.id):
            return True
        if isinstance(n, ast.Attribute) and KL_RE.search(n.attr):
            return True
        if isinstance(n, ast.Call) and KL_RE.search(last_part(call_name(n))):
            return True
    return False


def signed_terms(expr: ast.AST, sign: int = 1, parent: ast.AST | None = None):
    """Flatten an additive expression into (term, sign, parent BinOp) leaves."""
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Add, ast.Sub)):
        yield from signed_terms(expr.left, sign, expr)
        yield from signed_terms(expr.right, sign if isinstance(expr.op, ast.Add) else -sign, expr)
        return
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.USub):
        yield from signed_terms(expr.operand, -sign, None)
        return
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.UAdd):
        yield from signed_terms(expr.operand, sign, None)
        return
    if isinstance(expr, ast.Call) and (method(expr) in REDUCERS) and isinstance(expr.func, ast.Attribute):
        yield from signed_terms(expr.func.value, sign, None)
        return
    if isinstance(expr, ast.Call) and last_part(call_name(expr)) in REDUCERS and expr.args and not isinstance(expr.func, ast.Attribute) or (
        isinstance(expr, ast.Call) and call_name(expr) in ("torch.mean", "torch.sum") and expr.args
    ):
        yield from signed_terms(expr.args[0], sign, None)
        return
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, (ast.Mult, ast.Div)):
        for side, other in ((expr.left, expr.right), (expr.right, expr.left)):
            c = const_value(side)
            if isinstance(c, (int, float)) and c < 0:
                yield from signed_terms(other, -sign, None)
                return
    yield expr, sign, parent


@rule(
    "TM023", "kl-penalty-wrong-sign", category="rl_post_training", cause="kl_sign", severity=Severity.ERROR,
    confidence=0.8,
    description="The KL penalty enters the loss with a negative sign (or the reward with a positive sign), rewarding divergence from the reference policy.",
)
def kl_sign(ctx: LintContext):
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = dotted(node.targets[0]).lower()
        is_loss = "loss" in target and not KL_RE.search(target)
        is_reward = ("reward" in target or "return" in target) and not is_loss
        if not (is_loss or is_reward):
            continue
        leaves = list(signed_terms(node.value))
        if len(leaves) < 2:
            continue
        kl = [(t, s, p) for t, s, p in leaves if _mentions_kl(ctx, t)]
        others = [(t, s, p) for t, s, p in leaves if not _mentions_kl(ctx, t)]
        if not kl or not others:
            continue
        bad = all(s < 0 for _, s, _ in kl) if is_loss else all(s > 0 for _, s, _ in kl)
        if not bad:
            continue
        term, sign, parent = kl[0]
        what = "loss" if is_loss else "reward"
        data = {}
        if isinstance(parent, ast.BinOp) and parent.right is term:
            data = {"binop": node_key(parent), "left": node_key(parent.left), "right": node_key(parent.right),
                    "op": "+" if isinstance(parent.op, ast.Sub) else "-"}
        yield ctx.finding(
            "TM023", node,
            f"KL term `{ctx.a.text(term)}` enters the {what} with sign {'-' if sign < 0 else '+'}; a KL penalty must "
            f"{'increase the loss' if is_loss else 'decrease the reward'}.",
            data=data,
        )


@fixer("TM023")
def _fix_kl(ctx: LintContext, ev):
    if "binop" not in ev.data:
        return
    b = ctx.a.find_node(ev.data["binop"])
    left = ctx.a.text(ctx.a.find_node(ev.data["left"]))
    right = ctx.a.text(ctx.a.find_node(ev.data["right"]))
    ctx.editor.replace_node(b, f"{left} {ev.data['op']} {right}")


REF_RE = re.compile(r"(^|_|\.)ref(erence)?(_|$)|^ref|ref_model|reference", re.I)


@rule(
    "TM024", "trainable-reference-model", category="rl_post_training", cause="reference_model_trainable",
    severity=Severity.WARNING, confidence=0.55,
    description="The frozen reference policy is trainable (in the optimizer, or not frozen/eval), so the KL anchor drifts.",
)
def trainable_reference(ctx: LintContext):
    a = ctx.a
    refs = [m for m in a.models if REF_RE.search(m) and not m.startswith("self.") or REF_RE.search(m.split(".")[-1])]
    for ref in sorted(set(refs)):
        if ref not in a.assignments:
            continue
        in_opt = False
        for opt in a.optimizers:
            for _s, v in a.assignments.get(opt, []):
                if isinstance(v, ast.Call):
                    for n in ast.walk(v):
                        if isinstance(n, ast.Call) and method(n) in ("parameters", "named_parameters") and receiver(n) == ref:
                            in_opt = True
        if in_opt:
            yield ctx.finding("TM024", a.assignments[ref][0][0], f"reference model `{ref}` is passed to the optimizer and will be updated.",
                              confidence=0.9, severity=Severity.ERROR)
            continue
        frozen = re.search(rf"{re.escape(ref)}\.(eval\(\)|requires_grad_\(\s*False)|{re.escape(ref)}\.parameters\(\)[^\n]*\n[^\n]*requires_grad\s*=\s*False", ctx.source)
        if frozen:
            continue
        calls = [c for c in a.calls if dotted(c.func) in (ref, f"{ref}.forward", f"{ref}.generate")]
        from .rules_core import in_no_grad
        if calls and all(in_no_grad(ctx, c) for c in calls):
            continue
        stmt = a.assignments[ref][0][0]
        yield ctx.finding(
            "TM024", stmt,
            f"reference model `{ref}` is never frozen (no `.eval()` / `.requires_grad_(False)`) and is called with gradients enabled.",
            data={"ref": ref, "stmt": node_key(stmt)},
        )


@fixer("TM024")
def _fix_ref(ctx: LintContext, ev):
    if "ref" not in ev.data:
        return
    stmt = ctx.a.find_node(ev.data["stmt"], ast.stmt)
    ind = ctx.editor.line_indent(stmt.lineno)
    ref = ev.data["ref"]
    ctx.editor.insert_after_statement(stmt, f"{ind}{ref}.eval()\n{ind}{ref}.requires_grad_(False)")


LOGP_RE = re.compile(r"log_?p|logprob|log_prob|logits?_?p", re.I)


@rule(
    "TM025", "policy-ratio-not-exponentiated", category="rl_post_training", cause="ratio_not_exp",
    severity=Severity.ERROR, confidence=0.8,
    description="The PPO/GRPO importance ratio is computed as a log-prob difference without exp().",
)
def ratio_not_exp(ctx: LintContext):
    for node in ast.walk(ctx.tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = dotted(node.targets[0]).lower()
        if "ratio" not in target or "log" in target:
            continue
        v = node.value
        if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Sub):
            if LOGP_RE.search(ctx.a.text(v.left)) and LOGP_RE.search(ctx.a.text(v.right)):
                yield ctx.finding("TM025", node, f"`{ctx.a.text(node)}` is a log-ratio; the importance ratio is its exp().",
                                  data={"value": node_key(v)})


@fixer("TM025")
def _fix_ratio(ctx: LintContext, ev):
    v = ctx.a.find_node(ev.data["value"])
    ctx.editor.replace_node(v, f"torch.exp({ctx.a.text(v)})")
