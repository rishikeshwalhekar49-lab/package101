"""Rules beyond plain PyTorch loops: JAX PRNG reuse, Keras from_logits
mismatches, mixed precision without loss scaling, and train/eval transform
inconsistencies in torchvision pipelines."""

from __future__ import annotations

import ast
import re

from ...types import Severity
from .analysis import call_name, const_value, dotted, iter_nodes, kwarg, last_part, node_key
from .engine import LintContext, fixer, rule

# ----------------------------------------------------------------------- TM022
NON_CONSUMING = {"split", "fold_in", "PRNGKey", "key", "key_data", "wrap_key_data", "clone", "key_impl"}


def _jax_random_aliases(ctx: LintContext) -> set[str]:
    aliases = {"jax.random"}
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.ImportFrom) and node.module == "jax":
            for a in node.names:
                if a.name == "random":
                    aliases.add(a.asname or "random")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "jax.random" and a.asname:
                    aliases.add(a.asname)
    return aliases


@rule(
    "TM022", "jax-prng-key-reuse", category="framework_jax", cause="prng_key_reuse", severity=Severity.ERROR,
    confidence=0.85,
    description="The same JAX PRNG key is consumed twice, so 'independent' random draws are identical.",
)
def prng_reuse(ctx: LintContext):
    if not ctx.a.imports_module("jax"):
        return
    aliases = _jax_random_aliases(ctx)
    for fn in [n for n in ast.walk(ctx.tree) if isinstance(n, (ast.FunctionDef, ast.Module))]:
        events: list[tuple[tuple[int, int], str, str, ast.AST]] = []
        for n in iter_nodes(fn):
            if isinstance(n, ast.Call):
                cn = call_name(n)
                mod, _, func = cn.rpartition(".")
                if mod in aliases and func not in NON_CONSUMING and n.args and isinstance(n.args[0], ast.Name):
                    events.append(((n.lineno, n.col_offset), "use", n.args[0].id, n))
            elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    for e in ast.walk(t):
                        if isinstance(e, ast.Name):
                            events.append(((n.end_lineno, n.end_col_offset), "def", e.id, n))
            elif isinstance(n, (ast.For, ast.comprehension)):
                for e in ast.walk(n.target):
                    if isinstance(e, ast.Name):
                        events.append(((n.target.lineno, n.target.col_offset), "def", e.id, n))
        for _pos, kind, name, node in events:
            loop = ctx.a.enclosing_loop(node) if kind == "use" else None
            if loop is not None and (fn is ctx.tree or ctx.a.contains(fn, loop)):
                redefined = any(k == "def" and nm == name and ctx.a.contains(loop, nd) for _p, k, nm, nd in events)
                if not redefined:
                    stmt = ctx.a.enclosing_stmt(node)
                    yield ctx.finding(
                        "TM022", node, f"PRNG key `{name}` is consumed on every iteration of the loop at line {loop.lineno} without being split.",
                        data={"call": node_key(node), "key": name, "stmt": node_key(stmt)},
                    )
        used: dict[str, ast.AST] = {}
        for _pos, kind, name, node in sorted(events, key=lambda e: e[0]):
            if kind == "def":
                used.pop(name, None)
            elif name in used:
                stmt = ctx.a.enclosing_stmt(node)
                yield ctx.finding(
                    "TM022", node, f"PRNG key `{name}` was already consumed at line {used[name].lineno}; split it first.",
                    data={"call": node_key(node), "key": name, "stmt": node_key(stmt)},
                )
            else:
                used[name] = node


@fixer("TM022")
def _fix_prng(ctx: LintContext, ev):
    if not ctx.a.imports_module("jax"):
        return
    stmt = ctx.a.find_node(ev.data["stmt"], ast.stmt)
    call = ctx.a.find_node(ev.data["call"], ast.Call)
    key = ev.data["key"]
    ind = ctx.editor.line_indent(stmt.lineno)
    ctx.editor.insert_before_line(stmt.lineno, f"{ind}{key}, _tm_subkey = jax.random.split({key})")
    ctx.editor.replace_node(call.args[0], "_tm_subkey")


# ----------------------------------------------------------------------- TM021
@rule(
    "TM021", "keras-from-logits-mismatch", category="framework_keras", cause="from_logits_mismatch",
    severity=Severity.ERROR, confidence=0.75,
    description="Keras loss `from_logits` disagrees with the model's output activation (softmax applied twice or never).",
)
def keras_from_logits(ctx: LintContext):
    a = ctx.a
    if not (a.imports_module("keras") or a.imports_module("tensorflow")):
        return
    has_softmax = bool(re.search(r"activation\s*=\s*['\"](softmax|sigmoid)['\"]|Softmax\(|Activation\(\s*['\"](softmax|sigmoid)", ctx.source))
    for c in a.calls:
        lp = last_part(call_name(c))
        if lp in ("SparseCategoricalCrossentropy", "CategoricalCrossentropy", "BinaryCrossentropy"):
            fl = kwarg(c, "from_logits")
            if fl is not None and const_value(fl.value) is True and has_softmax:
                yield ctx.finding("TM021", c, f"{lp}(from_logits=True) but the model already ends in softmax/sigmoid.",
                                  data={"kw": node_key(fl.value), "set": "False"})
            elif (fl is None or const_value(fl.value) is False) and not has_softmax:
                yield ctx.finding("TM021", c, f"{lp} treats outputs as probabilities but no softmax/sigmoid output activation exists.",
                                  confidence=0.6, data={"call": node_key(c), "kw": node_key(fl.value) if fl else None, "set": "True"})
    for c in a.calls:
        if last_part(call_name(c)) == "compile":
            loss = kwarg(c, "loss")
            if loss is not None and const_value(loss.value) in ("sparse_categorical_crossentropy", "categorical_crossentropy") and not has_softmax:
                yield ctx.finding("TM021", c, f"loss='{const_value(loss.value)}' expects probabilities, but the model outputs logits.",
                                  confidence=0.6, data={"string_loss": node_key(loss.value), "name": const_value(loss.value)})


@fixer("TM021")
def _fix_keras(ctx: LintContext, ev):
    if "string_loss" in ev.data:
        node = ctx.a.find_node(ev.data["string_loss"])
        cls = "SparseCategoricalCrossentropy" if ev.data["name"].startswith("sparse") else "CategoricalCrossentropy"
        prefix = "tf.keras.losses" if ctx.a.imports_module("tensorflow") and "tf" in ctx.source else "keras.losses"
        ctx.editor.replace_node(node, f"{prefix}.{cls}(from_logits=True)")
        return
    if ev.data.get("kw"):
        ctx.editor.replace_node(ctx.a.find_node(ev.data["kw"]), ev.data["set"])
        return
    call = ctx.a.find_node(ev.data["call"], ast.Call)
    end = ctx.editor.span(call)[1] - 1
    sep = ", " if (call.args or call.keywords) else ""
    ctx.editor.replace_span(end, end, f"{sep}from_logits=True")


# ----------------------------------------------------------------------- TM031
@rule(
    "TM031", "fp16-without-grad-scaler", category="numerics", cause="fp16_no_loss_scaling", severity=Severity.WARNING,
    confidence=0.6,
    description="Training in float16 without a GradScaler lets small gradients underflow to zero.",
)
def fp16_without_scaler(ctx: LintContext):
    a = ctx.a
    if "GradScaler" in ctx.source or "bfloat16" in ctx.source or "bf16" in ctx.source:
        return
    if not any(a.is_backward(c) for c in a.calls):
        return
    for c in a.calls:
        cn = call_name(c)
        lp = last_part(cn)
        half_model = lp == "half" and dotted(c.func.value) in a.models if isinstance(c.func, ast.Attribute) else False
        fp16_autocast = lp == "autocast" and ("float16" in a.text(c) or "cuda.amp" in cn or (c.args and const_value(c.args[0]) == "cuda" and "dtype" not in a.text(c)))
        if half_model or fp16_autocast:
            yield ctx.finding("TM031", c, "float16 training without torch.amp.GradScaler; small gradients underflow to zero.")
            return


# ------------------------------------------------------------- TM032 / TM033
RANDOM_AUG = re.compile(r"^(Random\w+|ColorJitter|AutoAugment|RandAugment|TrivialAugmentWide|AugMix|GaussianBlur|ElasticTransform)$")
TRAIN_RE = re.compile(r"train", re.I)
EVAL_RE = re.compile(r"val|test|eval|dev", re.I)


def _composes(ctx: LintContext):
    for name, items in ctx.a.assignments.items():
        for _s, v in items:
            if isinstance(v, ast.Call) and last_part(call_name(v)) == "Compose" and v.args and isinstance(v.args[0], (ast.List, ast.Tuple)):
                yield name, v, v.args[0]


@rule(
    "TM032", "augmentation-at-eval", category="data", cause="eval_augmentation", severity=Severity.WARNING,
    confidence=0.7,
    description="Random augmentations are applied to validation/test data, making evaluation noisy and biased.",
)
def eval_augmentation(ctx: LintContext):
    for name, call, lst in _composes(ctx):
        if not EVAL_RE.search(name) or TRAIN_RE.search(name):
            continue
        for i, el in enumerate(lst.elts):
            if isinstance(el, ast.Call) and RANDOM_AUG.match(last_part(call_name(el))):
                yield ctx.finding("TM032", el, f"`{name}` applies random augmentation `{last_part(call_name(el))}` at evaluation time.",
                                  data={"list": node_key(lst), "index": i})


@fixer("TM032")
def _fix_eval_aug(ctx: LintContext, ev):
    lst = ctx.a.find_node(ev.data["list"])
    i = ev.data["index"]
    el = lst.elts[i]
    if i + 1 < len(lst.elts):
        s, e = ctx.editor.span(el)[0], ctx.editor.span(lst.elts[i + 1])[0]
    elif i > 0:
        s, e = ctx.editor.span(lst.elts[i - 1])[1], ctx.editor.span(el)[1]
    else:
        s, e = ctx.editor.span(el)
    ctx.editor.replace_span(s, e, "")


@rule(
    "TM033", "normalization-mismatch", category="data", cause="preprocessing_mismatch", severity=Severity.ERROR,
    confidence=0.75,
    description="Training and evaluation pipelines normalise inputs differently, shifting the evaluation input distribution.",
)
def normalization_mismatch(ctx: LintContext):
    comps = list(_composes(ctx))
    def has_norm(lst):
        return any(isinstance(e, ast.Call) and last_part(call_name(e)) == "Normalize" for e in lst.elts)
    train = [(n, c, l) for n, c, l in comps if TRAIN_RE.search(n) and not EVAL_RE.search(n)]
    evals = [(n, c, l) for n, c, l in comps if EVAL_RE.search(n) and not TRAIN_RE.search(n)]
    for tn, _tc, tl in train:
        for en, ec, el in evals:
            if has_norm(tl) != has_norm(el):
                which = en if has_norm(tl) else tn
                yield ctx.finding("TM033", ec, f"`{tn}` and `{en}` disagree on Normalize(); `{which}` is missing it.")
