"""Attention / transformer-specific rules (cf. "Why Attention Fails", ICSE 2026):
missing score scaling, softmax over the wrong axis, inverted causal masks and
language-model targets that are not shifted."""

from __future__ import annotations

import ast
import re

from ...types import Severity
from .analysis import CE_CLASSES, CE_NAMES, call_name, const_value, dotted, iter_nodes, kwarg, last_part, method, node_key, root_name
from .engine import LintContext, fixer, rule

TRANSPOSE_RE = re.compile(r"\.transpose\(\s*-?\d+\s*,\s*-?\d+\s*\)|\.mT\b|\.permute\(|\.T\b")
SCALE_RE = re.compile(r"sqrt|\*\*\s*\(?\s*-\s*0?\.5|\*\*\s*0?\.5|scale|rsqrt|temperature|softmax_scale|inv_freq_scale")


def _score_matmuls(ctx: LintContext, fn: ast.AST) -> list[tuple[ast.AST, ast.AST]]:
    """(matmul node, query operand) pairs that look like Q @ K^T."""
    out = []
    for n in iter_nodes(fn):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
            if TRANSPOSE_RE.search(ctx.a.text(n.right)):
                out.append((n, n.left))
        elif isinstance(n, ast.Call) and last_part(call_name(n)) in ("matmul", "bmm") and len(n.args) >= 2:
            if TRANSPOSE_RE.search(ctx.a.text(n.args[1])):
                out.append((n, n.args[0]))
        elif isinstance(n, ast.Call) and last_part(call_name(n)) == "einsum" and n.args:
            spec = const_value(n.args[0])
            if isinstance(spec, str) and re.fullmatch(r"\s*\w*(\w)(\w),\s*\w*(\w)\2->\w*\1\3\s*", spec.replace("...", "")) is None:
                continue
            if isinstance(spec, str) and len(n.args) >= 3:
                out.append((n, n.args[1]))
    return out


def _softmax_calls(fn: ast.AST) -> list[ast.Call]:
    return [n for n in iter_nodes(fn) if isinstance(n, ast.Call) and last_part(call_name(n)) == "softmax"]


def _functions(ctx: LintContext):
    for n in ast.walk(ctx.tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield n


@rule(
    "TM017", "attention-missing-scale", category="attention", cause="attention_scale_missing",
    severity=Severity.WARNING, confidence=0.6,
    description="Attention scores Q @ K^T are not scaled by 1/sqrt(d_head), saturating the softmax.",
)
def attention_missing_scale(ctx: LintContext):
    for fn in _functions(ctx):
        mm = _score_matmuls(ctx, fn)
        if not mm or not _softmax_calls(fn):
            continue
        if SCALE_RE.search(ctx.a.text(fn)):
            continue
        node, q = mm[0]
        yield ctx.finding(
            "TM017", node, f"attention scores `{ctx.a.text(node)}` are not divided by sqrt(head_dim).",
            data={"mm": node_key(node), "q": node_key(q)},
        )


@fixer("TM017")
def _fix_scale(ctx: LintContext, ev):
    mm = ctx.a.find_node(ev.data["mm"])
    q = ctx.a.find_node(ev.data["q"])
    ctx.editor.replace_node(mm, f"({ctx.a.text(mm)}) * ({ctx.a.text(q)}.size(-1) ** -0.5)")


@rule(
    "TM018", "attention-softmax-wrong-dim", category="attention", cause="attention_softmax_dim",
    severity=Severity.ERROR, confidence=0.7,
    description="The attention softmax normalises over the query axis instead of the key axis (dim=-1).",
)
def attention_softmax_dim(ctx: LintContext):
    for fn in _functions(ctx):
        if not _score_matmuls(ctx, fn):
            continue
        for sm in _softmax_calls(fn):
            dim_node = kwarg(sm, "dim")
            dnode = dim_node.value if dim_node else None
            if dnode is None:
                if method(sm) == "softmax" and sm.args and not call_name(sm).startswith(("F.", "torch.")):
                    dnode = sm.args[0]
                elif len(sm.args) >= 2:
                    dnode = sm.args[1]
            dim = const_value(dnode)
            if isinstance(dim, int) and dim in (0, 1, -2):
                yield ctx.finding(
                    "TM018", sm, f"attention softmax uses dim={dim}; attention weights must sum to 1 over keys (dim=-1).",
                    data={"dim": node_key(dnode)},
                )


@fixer("TM018")
def _fix_dim(ctx: LintContext, ev):
    ctx.editor.replace_node(ctx.a.find_node(ev.data["dim"]), "-1")


NEG_INF_RE = re.compile(r"-\s*inf|-\s*float\(\s*['\"]inf|float\(\s*['\"]-inf['\"]\s*\)|-\s*1e\d+|finfo\(.*\)\.min|-\s*\d{4,}|-math\.inf|-torch\.inf|-np\.inf")


def _mask_polarity(cond: ast.AST) -> tuple[ast.AST, int] | None:
    """Return (mask base expression, value of base that gets masked)."""
    if isinstance(cond, ast.Compare) and len(cond.ops) == 1:
        v = const_value(cond.comparators[0])
        if v is True:
            v = 1
        if v is False:
            v = 0
        if v not in (0, 1):
            return None
        op = cond.ops[0]
        if isinstance(op, ast.Eq):
            return cond.left, int(v)
        if isinstance(op, ast.NotEq):
            return cond.left, 1 - int(v)
        return None
    if isinstance(cond, ast.UnaryOp) and isinstance(cond.op, (ast.Invert, ast.Not)):
        inner = _mask_polarity(cond.operand)
        return (inner[0], 1 - inner[1]) if inner else None
    if isinstance(cond, ast.Call) and method(cond) in ("bool", "to") and isinstance(cond.func, ast.Attribute):
        return cond.func.value, 1
    if isinstance(cond, (ast.Name, ast.Attribute, ast.Subscript)):
        return cond, 1
    return None


def _mask_origin(ctx: LintContext, base: ast.AST, at: ast.AST) -> str | None:
    """'tril' if the mask's ones are the *visible* positions, 'triu' if ones are the *masked* positions."""
    name = dotted(base)
    while name.endswith("[]"):
        name = name[:-2]
    candidates: list[str] = []
    if name.startswith("self."):
        attr = name.split(".", 1)[1]
        for c in ctx.a.calls:
            if method(c) == "register_buffer" and c.args and const_value(c.args[0]) == attr and len(c.args) > 1:
                candidates.append(ctx.a.text(c.args[1]))
        for _stmt, v in ctx.a.assignments.get(name, []):
            candidates.append(ctx.a.text(v))
    else:
        root = root_name(base)
        v = ctx.a.last_assignment_before(root, at)
        if v is not None:
            candidates.append(ctx.a.text(v))
    for text in candidates:
        if "tril" in text:
            return "tril"
        if "triu" in text and re.search(r"diagonal\s*=\s*1|,\s*1\s*\)", text):
            return "triu"
    return None


@rule(
    "TM019", "causal-mask-inverted", category="attention", cause="causal_mask_inverted",
    severity=Severity.ERROR, confidence=0.85,
    description="masked_fill hides the positions the causal mask marks as visible, so the model attends to the future.",
)
def causal_mask_inverted(ctx: LintContext):
    for c in ctx.a.calls:
        if method(c) not in ("masked_fill", "masked_fill_") or len(c.args) < 2:
            continue
        if not NEG_INF_RE.search(ctx.a.text(c.args[1]).replace(" ", "")) and not NEG_INF_RE.search(ctx.a.text(c.args[1])):
            continue
        pol = _mask_polarity(c.args[0])
        if pol is None:
            continue
        base, masked_value = pol
        origin = _mask_origin(ctx, base, c)
        if origin == "tril" and masked_value == 1 or origin == "triu" and masked_value == 0:
            want = 0 if origin == "tril" else 1
            yield ctx.finding(
                "TM019", c,
                f"the mask comes from torch.{origin} but masked_fill hides positions where it equals {masked_value}; "
                f"this blocks the past and exposes the future.",
                data={"cond": node_key(c.args[0]), "base": node_key(base), "want": want},
            )


@fixer("TM019")
def _fix_mask(ctx: LintContext, ev):
    cond = ctx.a.find_node(ev.data["cond"])
    base = ctx.a.find_node(ev.data["base"])
    ctx.editor.replace_node(cond, f"{ctx.a.text(base)} == {ev.data['want']}")


@rule(
    "TM020", "lm-targets-not-shifted", category="attention", cause="lm_target_not_shifted",
    severity=Severity.ERROR, confidence=0.75,
    description="Language-model targets are the inputs themselves (not shifted by one), so the model learns to copy.",
)
def lm_targets_not_shifted(ctx: LintContext):
    a = ctx.a
    for c in a.calls:
        if not a.is_loss_call(c, CE_NAMES, CE_CLASSES) or len(c.args) < 2:
            continue
        logits, target = c.args[0], c.args[1]
        lname, tname = root_name(logits), root_name(target)
        if not lname or not tname:
            continue
        val = a.last_assignment_before(lname, c)
        if not isinstance(val, ast.Call):
            continue
        model_call = val
        if not (a.is_model_call(model_call) or call_name(model_call) == "self"):
            continue
        inputs = {root_name(x) for x in model_call.args}
        ttext, ltext = a.text(target), a.text(logits)
        if tname in inputs and "1:" not in ttext and ":-1" not in ltext and "shift" not in ttext:
            data = {"logits": node_key(logits), "target": node_key(target), "lname": lname, "tname": tname}
            input_sliced = any(root_name(x) == tname and re.search(r":\s*-1\s*\]", a.text(x)) for x in model_call.args)
            tsub = next((n for n in ast.walk(target) if isinstance(n, ast.Subscript) and re.search(r":\s*-1\s*\]$", a.text(n))), None)
            if input_sliced and tsub is not None:
                data = {"mode": "retarget", "sub": node_key(tsub)}  # model saw x[:, :-1]; targets must be x[:, 1:]
            yield ctx.finding(
                "TM020", c,
                f"cross-entropy target `{ttext}` is the same tensor that was fed to the model; next-token targets must be shifted (`{tname}[:, 1:]`).",
                data=data,
            )


@fixer("TM020")
def _fix_shift(ctx: LintContext, ev):
    a = ctx.a
    if ev.data.get("mode") == "retarget":
        sub = a.find_node(ev.data["sub"], ast.Subscript)
        text = a.text(sub)
        ctx.editor.replace_node(sub, re.sub(r":\s*-1\s*\]$", "1:]", text))
        return
    for key, name, sl in ((ev.data["logits"], ev.data["lname"], "[:, :-1]"), (ev.data["target"], ev.data["tname"], "[:, 1:]")):
        expr = a.find_node(key)
        names = sorted(
            (n for n in ast.walk(expr) if isinstance(n, ast.Name) and n.id == name), key=lambda n: (n.lineno, n.col_offset)
        )
        if not names:
            raise ValueError("name not found")
        ctx.editor.replace_node(names[0], f"{name}{sl}")
        for n in ast.walk(expr):
            if isinstance(n, ast.Attribute) and n.attr == "view" and root_name(n) == name:
                end = ctx.editor.span(n)[1]
                ctx.editor.replace_span(end - len("view"), end, "reshape")
