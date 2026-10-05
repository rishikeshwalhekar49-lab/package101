"""Source-level bug injection for DL training code.

Operators are AST-located but edit the original text in place, so the buggy twin
differs from the clean program by a minimal, readable diff (which doubles as the
ground-truth repair). They work on any PyTorch script, not just the bundled
pipelines -- ``trainmedic bench mutate my_train.py`` builds a contamination-free
benchmark from a team's own fresh code.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..detect.lint.analysis import CE_CLASSES, CE_NAMES, call_name, const_value, dotted, kwarg, last_part, method, receiver
from ..detect.lint.engine import LintContext
from ..sourceedit import parses


@dataclass
class Mutation:
    operator: str
    description: str
    line: int
    source: str
    category: str
    cause: str
    kind: str = "code"
    expected_rules: list[str] = field(default_factory=list)
    expected_invariants: list[str] = field(default_factory=list)


@dataclass
class Operator:
    name: str
    description: str
    category: str
    cause: str
    find: Callable[[LintContext], list]
    apply: Callable[[LintContext, object], None]
    kind: str = "code"
    expected_rules: list[str] = field(default_factory=list)
    expected_invariants: list[str] = field(default_factory=list)


OPERATORS: dict[str, Operator] = {}


def operator(name, description, category, cause, kind="code", rules=(), invariants=()):
    def deco(fn_pair):
        find, apply = fn_pair()
        OPERATORS[name] = Operator(name, description, category, cause, find, apply, kind, list(rules), list(invariants))
        return fn_pair

    return deco


def _expr_stmts(ctx: LintContext, pred) -> list[ast.stmt]:
    out = []
    for n in ast.walk(ctx.tree):
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and pred(n.value):
            out.append(n)
    return sorted(out, key=lambda n: n.lineno)


def _delete(ctx: LintContext, stmt) -> None:
    ctx.editor.delete_statement(stmt, ctx.a.siblings(stmt))


# ------------------------------------------------------------------ operators
@operator("remove_zero_grad", "delete optimizer.zero_grad()", "optimization", "missing_zero_grad",
          rules=["TM001"], invariants=["grads_cleared_between_steps"])
def _():
    return (lambda ctx: _expr_stmts(ctx, lambda c: method(c) == "zero_grad"), _delete)


@operator("zero_grad_after_backward", "move zero_grad() between backward() and step()", "optimization",
          "grads_cleared_before_step", rules=["TM002"], invariants=["gradients_present_at_step"])
def _():
    def find(ctx):
        sites = []
        for tl in ctx.a.training_loops():
            if tl.zero_grad_calls and tl.backward_calls and tl.zero_grad_calls[0].lineno < tl.backward_calls[0].lineno:
                sites.append((ctx.a.enclosing_stmt(tl.zero_grad_calls[0]), ctx.a.enclosing_stmt(tl.backward_calls[-1])))
        return sites

    def apply(ctx, site):
        z, b = site
        ctx.editor.move_statement_after(z, b, ctx.a.siblings(z))

    return find, apply


@operator("remove_eval", "delete model.eval() before evaluation", "model_mode", "missing_eval_mode",
          rules=["TM006"], invariants=["eval_mode_during_evaluation"])
def _():
    return (lambda ctx: _expr_stmts(ctx, lambda c: method(c) == "eval" and not c.args), _delete)


@operator("remove_train", "delete model.train() before training", "model_mode", "missing_train_mode",
          rules=["TM007"], invariants=["train_mode_during_training"])
def _():
    return (lambda ctx: _expr_stmts(ctx, lambda c: method(c) == "train" and not c.args), _delete)


@operator("softmax_before_ce", "apply softmax to the logits passed to cross-entropy", "loss", "double_softmax",
          rules=["TM008"], invariants=["cross_entropy_gets_logits"])
def _():
    def find(ctx):
        return [c for c in ctx.a.calls if ctx.a.is_loss_call(c, CE_NAMES, CE_CLASSES) and c.args]

    def apply(ctx, call):
        arg = call.args[0]
        ctx.editor.replace_node(arg, f"torch.softmax({ctx.a.text(arg)}, dim=-1)")

    return find, apply


@operator("unscale_grad_accum", "drop the 1/accum scaling of the accumulated loss", "optimization",
          "grad_accum_unscaled", rules=["TM011"], invariants=["grad_accumulation_scaled"])
def _():
    def find(ctx):
        sites = []
        for c in ctx.a.calls:
            if method(c) == "backward" and isinstance(c.func.value, ast.BinOp) and isinstance(c.func.value.op, ast.Div):
                sites.append(("div", c.func.value))
        for n in ast.walk(ctx.tree):
            if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Div) and "loss" in dotted(n.target):
                sites.append(("stmt", n))
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Div) and \
                    any("loss" in dotted(t) for t in n.targets) and "accum" in ctx.a.text(n.value.right):
                sites.append(("stmt", n))
        return sites

    def apply(ctx, site):
        kind, node = site
        if kind == "div":
            left = node.left
            ctx.editor.replace_node(node, ctx.a.text(left))
        else:
            _delete(ctx, node)

    return find, apply


@operator("disable_shuffle", "set shuffle=False on the training DataLoader", "data", "no_shuffle",
          rules=["TM012"], invariants=["training_data_shuffled"])
def _():
    def find(ctx):
        out = []
        for c in ctx.a.calls:
            if last_part(call_name(c)) == "DataLoader":
                kw = kwarg(c, "shuffle")
                if kw is not None and const_value(kw.value) is True:
                    out.append(kw)
        return out

    return find, lambda ctx, kw: ctx.editor.replace_node(kw.value, "False")


@operator("drop_dropout_training_flag", "remove training=self.training from F.dropout", "model_mode",
          "dropout_in_eval", rules=["TM014"], invariants=["eval_mode_deterministic"])
def _():
    def find(ctx):
        return [c for c in ctx.a.calls if last_part(call_name(c)).startswith("dropout") and kwarg(c, "training") is not None]

    def apply(ctx, call):
        kw = kwarg(call, "training")
        s, e = ctx.editor.span(kw.value)
        kw_start = ctx.editor.source.rfind("training", 0, s)
        ends = [ctx.editor.span(n)[1] for n in list(call.args) + [k.value for k in call.keywords if k is not kw]]
        before = [x for x in ends if x <= kw_start]
        start = max(before) if before else kw_start
        ctx.editor.replace_span(start, e, "")

    return find, apply


def _mask_fills(ctx):
    out = []
    for c in ctx.a.calls:
        if method(c) in ("masked_fill", "masked_fill_") and c.args and isinstance(c.args[0], ast.Compare):
            cmp = c.args[0]
            if len(cmp.ops) == 1 and isinstance(cmp.ops[0], ast.Eq) and const_value(cmp.comparators[0]) in (0, 1):
                out.append(cmp)
    return out


@operator("invert_causal_mask", "flip the causal-mask comparison (== 0 <-> == 1)", "attention",
          "causal_mask_inverted", rules=["TM019"], invariants=["causality"])
def _():
    def apply(ctx, cmp):
        v = const_value(cmp.comparators[0])
        ctx.editor.replace_node(cmp.comparators[0], "1" if v == 0 else "0")

    return _mask_fills, apply


@operator("remove_causal_mask", "delete the masked_fill that applies the causal mask", "attention",
          "future_leakage", invariants=["causality"])
def _():
    def find(ctx):
        return [ctx.a.enclosing_stmt(c) for c in ctx.a.calls if method(c) in ("masked_fill", "masked_fill_")]

    return find, _delete


def _score_products(ctx):
    out = []
    for n in ast.walk(ctx.tree):
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Div)):
            for side in (n.left, n.right):
                if isinstance(side, ast.BinOp) and isinstance(side.op, ast.MatMult) and re.search(r"transpose|\.mT|\.T\b", ctx.a.text(side.right)):
                    out.append((n, side))
    return out


@operator("remove_attention_scale", "drop the 1/sqrt(d_head) scaling of attention scores", "attention",
          "attention_scale_missing", rules=["TM017"])
def _():
    return _score_products, lambda ctx, site: ctx.editor.replace_node(site[0], ctx.a.text(site[1]))


@operator("softmax_wrong_dim", "normalise attention over the query axis (dim=-2)", "attention",
          "attention_softmax_dim", rules=["TM018"])
def _():
    def find(ctx):
        out = []
        for c in ctx.a.calls:
            if last_part(call_name(c)) == "softmax":
                fn = ctx.a.enclosing_function(c)
                if fn is not None and _score_products_in(ctx, fn):
                    kw = kwarg(c, "dim")
                    if kw is not None and const_value(kw.value) == -1:
                        out.append(kw.value)
        return out

    return find, lambda ctx, node: ctx.editor.replace_node(node, "-2")


def _score_products_in(ctx, fn) -> bool:
    return any(ctx.a.contains(fn, n) for n, _ in _score_products(ctx)) or bool(
        re.search(r"@\s*\w+\.transpose\(", ctx.a.text(fn))
    )


@operator("unshift_lm_targets", "use the inputs themselves as next-token targets", "attention",
          "lm_target_not_shifted", rules=["TM020"], invariants=["loss_not_trivially_solved"])
def _():
    def find(ctx):
        out = []
        for c in ctx.a.calls:
            if ctx.a.is_loss_call(c, CE_NAMES, CE_CLASSES) and len(c.args) >= 2:
                for n in ast.walk(c.args[1]):
                    if isinstance(n, ast.Subscript) and re.search(r":\s*,\s*1\s*:", ctx.a.text(n)):
                        out.append(n)
                        break
        return out

    def apply(ctx, sub):
        text = ctx.a.text(sub)
        ctx.editor.replace_node(sub, re.sub(r"(:\s*,\s*)1\s*:", r"\1:-1", text, count=1))

    return find, apply


@operator("flip_kl_sign", "subtract (instead of add) the KL penalty in the loss", "rl_post_training", "kl_sign",
          rules=["TM023"])
def _():
    def find(ctx):
        out = []
        for n in ast.walk(ctx.tree):
            if isinstance(n, ast.Assign) and any("loss" in dotted(t) for t in n.targets):
                v = n.value
                if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add) and re.search(r"\bkl", ctx.a.text(v.right)):
                    out.append(v)
        return out

    def apply(ctx, v):
        ctx.editor.replace_node(v, f"{ctx.a.text(v.left)} - {ctx.a.text(v.right)}")

    return find, apply


@operator("swap_preference_pair", "swap chosen/rejected in a preference log-ratio", "rl_post_training",
          "preference_swap")
def _():
    def find(ctx):
        out = []
        for n in ast.walk(ctx.tree):
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub):
                lt, rt = ctx.a.text(n.left), ctx.a.text(n.right)
                if "chosen" in lt and "rejected" in rt and "chosen" not in rt:
                    out.append(n)
        return out

    def apply(ctx, n):
        ctx.editor.replace_node(n, f"{ctx.a.text(n.right)} - {ctx.a.text(n.left)}")

    return find, apply


@operator("lr_x100", "multiply the optimizer learning rate by 100", "optimization", "lr_scale", kind="hyperparameter",
          invariants=["update_ratio_sane"])
def _():
    def find(ctx):
        out = []
        for name in ctx.a.optimizers:
            for _s, v in ctx.a.assignments.get(name, []):
                if isinstance(v, ast.Call):
                    kw = kwarg(v, "lr")
                    if kw is not None and isinstance(const_value(kw.value), (int, float)):
                        out.append(kw.value)
        return out

    def apply(ctx, node):
        ctx.editor.replace_node(node, repr(float(const_value(node)) * 100))

    return find, apply


@operator("clip_after_step", "move gradient clipping after optimizer.step()", "optimization", "grad_clip_misplaced",
          rules=["TM003"])
def _():
    def find(ctx):
        return [(ctx.a.enclosing_stmt(tl.clip_calls[0]), ctx.a.enclosing_stmt(tl.step_calls[0]))
                for tl in ctx.a.training_loops() if tl.clip_calls]

    def apply(ctx, site):
        c, s = site
        ctx.editor.move_statement_after(c, s, ctx.a.siblings(c))

    return find, apply


@operator("batch_mixing_flatten", "flatten with a transpose that mixes the batch dimension", "model_structure",
          "batch_mixing", invariants=["batch_independence"])
def _():
    def find(ctx):
        out = []
        for c in ctx.a.calls:
            if method(c) in ("view", "reshape") and len(c.args) == 2 and const_value(c.args[1]) == -1:
                a0 = c.args[0]
                if isinstance(a0, ast.Call) and method(a0) == "size" and a0.args and const_value(a0.args[0]) == 0:
                    out.append(c)
        return out

    def apply(ctx, c):
        x = ctx.a.text(c.func.value)
        ctx.editor.replace_node(c, f"{x}.transpose(0, 1).reshape({x}.size(0), -1)")

    return find, apply


@operator("skip_eval_normalization", "forget to normalise the evaluation inputs", "data", "preprocessing_mismatch",
          invariants=["input_distribution_consistent"])
def _():
    def find(ctx):
        out = []
        for n in ast.walk(ctx.tree):
            if isinstance(n, ast.Assign) and len(n.targets) == 1:
                t = dotted(n.targets[0])
                if re.search(r"val|test|eval", t) and isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Div) \
                        and re.search(r"std|mean|norm", ctx.a.text(n.value)):
                    out.append(n)
        return out

    return find, _delete


@operator("remove_scheduler_step", "never step the LR scheduler", "optimization", "lr_schedule_misuse",
          rules=["TM040"], invariants=["scheduler_cadence"])
def _():
    return (lambda ctx: _expr_stmts(ctx, lambda c: ctx.a.is_scheduler_step(c)), _delete)


# ----------------------------------------------------------------- driver
def mutate(source: str, filename: str = "<string>", operators: Iterable[str] | None = None, max_sites: int = 1) -> list[Mutation]:
    names = list(operators) if operators else list(OPERATORS)
    out: list[Mutation] = []
    for name in names:
        op = OPERATORS[name]
        ctx = LintContext(source, filename)
        try:
            sites = op.find(ctx)
        except Exception:
            continue
        for i in range(min(max_sites, len(sites))):
            ctx = LintContext(source, filename)
            site = op.find(ctx)[i]
            try:
                op.apply(ctx, site)
                new = ctx.editor.apply()
            except Exception:
                continue
            if new == source or not parses(new):
                continue
            line = getattr(site, "lineno", None) or getattr(site[0] if isinstance(site, tuple) else site, "lineno", 0)
            out.append(Mutation(op.name, op.description, int(line or 0), new, op.category, op.cause, op.kind,
                                list(op.expected_rules), list(op.expected_invariants)))
    return out
