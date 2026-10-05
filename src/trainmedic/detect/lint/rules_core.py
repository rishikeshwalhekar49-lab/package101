"""Core PyTorch training-loop rules: gradient lifecycle, train/eval mode,
loss/activation pairing, gradient accumulation, schedulers and data loading."""

from __future__ import annotations

import ast
import re

from ...types import Severity
from .analysis import (
    BCE_LOGITS_CLASSES,
    BCE_LOGITS_NAMES,
    CE_CLASSES,
    CE_NAMES,
    EPOCH_SCHEDULERS,
    HF_SCHEDULER_FUNCS,
    NLL_CLASSES,
    NLL_NAMES,
    PER_STEP_SCHEDULERS,
    call_name,
    dotted,
    iter_nodes,
    kwarg,
    last_part,
    method,
    node_key,
    receiver,
    root_name,
)
from .engine import LintContext, fixer, rule

DROPOUT_OR_NORM = (r"Dropout|BatchNorm|dropout|batch_norm|from_pretrained|LayerDrop|InstanceNorm\w*\(.*track_running_stats=True")


def cyclic_between(a: int, b: int, c: int) -> bool:
    """Is ``c`` strictly between ``a`` and ``b`` walking forward around a loop body?"""
    if a < b:
        return a < c < b
    return c > a or c < b


def _pos(node: ast.AST) -> tuple[int, int]:
    return (node.lineno, node.col_offset)  # type: ignore[attr-defined]


def in_no_grad(ctx: LintContext, node: ast.AST) -> bool:
    cur = ctx.a.parent(node)
    while cur is not None:
        if isinstance(cur, (ast.With, ast.AsyncWith)):
            for item in cur.items:
                expr = item.context_expr
                name = call_name(expr) if isinstance(expr, ast.Call) else dotted(expr)
                if last_part(name) in ("no_grad", "inference_mode"):
                    return True
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in cur.decorator_list:
                name = call_name(dec) if isinstance(dec, ast.Call) else dotted(dec)
                if last_part(name) in ("no_grad", "inference_mode"):
                    return True
        cur = ctx.a.parent(cur)
    return False


def has_dropout_or_norm(ctx: LintContext) -> bool:
    return ctx.a.file_mentions(DROPOUT_OR_NORM)


# --------------------------------------------------------------------------- TM001
@rule(
    "TM001", "missing-zero-grad", category="optimization", cause="missing_zero_grad", severity=Severity.ERROR,
    confidence=0.9,
    description="Gradients are never cleared inside the training loop, so they accumulate across optimizer steps.",
)
def missing_zero_grad(ctx: LintContext):
    a = ctx.a
    for tl in a.training_loops():
        if tl.zero_grad_calls or tl.grad_none_assigns:
            continue
        step = tl.step_calls[0]
        opt = receiver(step)
        elsewhere = any(a.is_zero_grad(c) for c in a.calls)
        yield ctx.finding(
            "TM001", step,
            f"`{opt}.step()` is called in a loop that never clears gradients; each step applies the sum of all previous gradients.",
            confidence=0.75 if elsewhere else 0.9,
            data={"optimizer": opt, "step": node_key(step)},
        )


@fixer("TM001")
def _fix_missing_zero_grad(ctx: LintContext, ev):
    step = ctx.a.find_node(ev.data["step"], ast.Call)
    stmt = ctx.a.enclosing_stmt(step)
    indent = ctx.editor.line_indent(stmt.lineno)
    ctx.editor.insert_after_statement(stmt, f"{indent}{ev.data['optimizer']}.zero_grad()")


# --------------------------------------------------------------------------- TM002
@rule(
    "TM002", "zero-grad-before-step", category="optimization", cause="grads_cleared_before_step",
    severity=Severity.ERROR, confidence=0.9,
    description="zero_grad() runs between backward() and step(), so the optimizer steps with empty gradients.",
)
def zero_grad_between_backward_and_step(ctx: LintContext):
    for tl in ctx.a.training_loops():
        if not tl.zero_grad_calls:
            continue
        b, s, z = tl.backward_calls[0], tl.step_calls[0], tl.zero_grad_calls[0]
        if cyclic_between(b.lineno, s.lineno, z.lineno):
            yield ctx.finding(
                "TM002", z,
                f"`{dotted(z.func)}()` clears gradients after `backward()` but before `{dotted(s.func)}()`; the step sees no gradient.",
                data={"zero_grad": node_key(z), "step": node_key(s)},
            )


@fixer("TM002")
def _fix_zero_grad_order(ctx: LintContext, ev):
    z = ctx.a.find_node(ev.data["zero_grad"], ast.Call)
    s = ctx.a.find_node(ev.data["step"], ast.Call)
    zs, ss = ctx.a.enclosing_stmt(z), ctx.a.enclosing_stmt(s)
    ctx.editor.move_statement_after(zs, ss, ctx.a.siblings(zs))


# --------------------------------------------------------------------------- TM003
@rule(
    "TM003", "clip-grad-misplaced", category="optimization", cause="grad_clip_misplaced", severity=Severity.WARNING,
    confidence=0.8,
    description="Gradient clipping must run after backward() and before step(); elsewhere it has no effect.",
)
def clip_misplaced(ctx: LintContext):
    for tl in ctx.a.training_loops():
        if not tl.clip_calls:
            continue
        b, s, c = tl.backward_calls[0], tl.step_calls[0], tl.clip_calls[0]
        if not cyclic_between(b.lineno, s.lineno, c.lineno):
            yield ctx.finding(
                "TM003", c,
                "gradient clipping is not between backward() and step(), so the update is never clipped.",
                data={"clip": node_key(c), "step": node_key(s)},
            )


@fixer("TM003")
def _fix_clip(ctx: LintContext, ev):
    c = ctx.a.find_node(ev.data["clip"], ast.Call)
    s = ctx.a.find_node(ev.data["step"], ast.Call)
    cs, ss = ctx.a.enclosing_stmt(c), ctx.a.enclosing_stmt(s)
    ctx.editor.move_statement_before(cs, ss, ctx.a.siblings(cs))


# --------------------------------------------------------------------------- TM004
@rule(
    "TM004", "scheduler-before-optimizer", category="optimization", cause="lr_schedule_misuse", severity=Severity.INFO,
    confidence=0.5,
    description="lr_scheduler.step() is called before optimizer.step() in the same loop; the first LR value is skipped.",
)
def scheduler_before_optimizer(ctx: LintContext):
    for tl in ctx.a.training_loops():
        if tl.sched_calls and tl.sched_calls[0].lineno < tl.step_calls[0].lineno:
            sc, st = tl.sched_calls[0], tl.step_calls[0]
            yield ctx.finding(
                "TM004", sc, f"`{dotted(sc.func)}()` runs before `{dotted(st.func)}()`; the schedule is shifted by one step.",
                data={"sched": node_key(sc), "step": node_key(st)},
            )


@fixer("TM004")
def _fix_sched_order(ctx: LintContext, ev):
    sc = ctx.a.find_node(ev.data["sched"], ast.Call)
    st = ctx.a.find_node(ev.data["step"], ast.Call)
    scs = ctx.a.enclosing_stmt(sc)
    ctx.editor.move_statement_after(scs, ctx.a.enclosing_stmt(st), ctx.a.siblings(scs))


# --------------------------------------------------------------------------- TM005
@rule(
    "TM005", "epoch-scheduler-per-batch", category="optimization", cause="lr_schedule_misuse", severity=Severity.WARNING,
    confidence=0.45,
    description="An epoch-based scheduler (StepLR, MultiStepLR, ...) is stepped every batch, decaying the LR far too fast.",
)
def epoch_scheduler_per_batch(ctx: LintContext):
    a = ctx.a
    for tl in a.training_loops():
        outer = a.enclosing_loop(tl.loop)
        if outer is None:
            continue
        for sc in tl.sched_calls:
            kind = a.schedulers.get(receiver(sc), "")
            if kind in EPOCH_SCHEDULERS:
                yield ctx.finding(
                    "TM005", sc,
                    f"`{receiver(sc)}` is a {kind} (an epoch-based scheduler) but it is stepped once per batch.",
                    data={"sched": node_key(sc), "loop": node_key(tl.loop)},
                )


@fixer("TM005")
def _fix_epoch_sched(ctx: LintContext, ev):
    sc = ctx.a.find_node(ev.data["sched"], ast.Call)
    loop = ctx.a.find_node(ev.data["loop"], (ast.For, ast.While))
    scs = ctx.a.enclosing_stmt(sc)
    ctx.editor.move_statement_after(scs, loop, ctx.a.siblings(scs))


# --------------------------------------------------------------------------- TM040
@rule(
    "TM040", "scheduler-never-stepped", category="optimization", cause="lr_schedule_misuse", severity=Severity.WARNING,
    confidence=0.7,
    description="An LR scheduler is created but scheduler.step() is never called, so the schedule is never applied.",
)
def scheduler_never_stepped(ctx: LintContext):
    a = ctx.a
    loops = a.training_loops()
    if not loops:
        return
    stepped = {receiver(c) for c in a.calls if method(c) == "step"}
    for name, kind in a.schedulers.items():
        if name in stepped or kind == "unknown":
            continue
        stmt = a.assignments[name][0][0]
        yield ctx.finding(
            "TM040", stmt, f"scheduler `{name}` ({kind}) is never stepped; the learning rate stays constant.",
            data={"sched": name, "kind": kind, "step": node_key(loops[0].step_calls[0]), "loop": node_key(loops[0].loop)},
        )


@fixer("TM040")
def _fix_sched_never(ctx: LintContext, ev):
    a = ctx.a
    kind = ev.data["kind"]
    if kind == "ReduceLROnPlateau":
        return
    step = a.find_node(ev.data["step"], ast.Call)
    loop = a.find_node(ev.data["loop"], (ast.For, ast.While))
    per_step = kind in PER_STEP_SCHEDULERS or kind == "LambdaLR" or bool(HF_SCHEDULER_FUNCS.fullmatch(kind))
    if per_step or a.enclosing_loop(loop) is None:
        st = a.enclosing_stmt(step)
        ctx.editor.insert_after_statement(st, f"{ctx.editor.line_indent(st.lineno)}{ev.data['sched']}.step()")
    else:
        ctx.editor.insert_after_statement(loop, f"{ctx.editor.line_indent(loop.lineno)}{ev.data['sched']}.step()")


# --------------------------------------------------------------------------- TM006
def _no_grad_blocks(ctx: LintContext):
    for node in ast.walk(ctx.tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                expr = item.context_expr
                name = call_name(expr) if isinstance(expr, ast.Call) else dotted(expr)
                if last_part(name) in ("no_grad", "inference_mode"):
                    yield node, node.body
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                name = call_name(dec) if isinstance(dec, ast.Call) else dotted(dec)
                if last_part(name) in ("no_grad", "inference_mode"):
                    yield node, node.body


def _model_in_training_loop(ctx: LintContext, tl) -> str | None:
    for c in ctx.a.calls_in(tl.loop):
        if ctx.a.is_model_call(c) and dotted(c.func) != "self":
            return dotted(c.func)
    return None


def _calls_eval_mode(call: ast.Call) -> bool:
    if method(call) == "eval" and not call.args:
        return True
    if method(call) == "train" and call.args and isinstance(call.args[0], ast.Constant) and call.args[0].value is False:
        return True
    return False


def _calls_train_mode(call: ast.Call) -> bool:
    if method(call) != "train":
        return False
    if not call.args:
        return True
    return not (isinstance(call.args[0], ast.Constant) and call.args[0].value is False)


@rule(
    "TM006", "missing-eval-mode", category="model_mode", cause="missing_eval_mode", severity=Severity.WARNING,
    confidence=0.85,
    description="The model is evaluated under no_grad() but never switched to eval(); dropout/batch-norm stay in training mode.",
)
def missing_eval(ctx: LintContext):
    a = ctx.a
    if any(_calls_eval_mode(c) for c in a.calls) or not has_dropout_or_norm(ctx):
        return
    for block, body in _no_grad_blocks(ctx):
        for c in a.calls_in(block, skip_defs=False):
            if a.is_model_call(c) and dotted(c.func) != "self":
                loops = a.training_loops()
                yield ctx.finding(
                    "TM006", block,
                    f"`{dotted(c.func)}` is evaluated without `{dotted(c.func)}.eval()`; dropout and batch-norm statistics are still in training mode.",
                    data={
                        "model": dotted(c.func),
                        "block": node_key(block),
                        "train_loop": node_key(loops[0].loop) if loops else None,
                        "needs_train": not any(_calls_train_mode(x) for x in a.calls),
                        "train_model": (_model_in_training_loop(ctx, loops[0]) if loops else None) or dotted(c.func),
                    },
                )
                break


@fixer("TM006")
def _fix_missing_eval(ctx: LintContext, ev):
    a = ctx.a
    block = a.find_node(ev.data["block"], (ast.With, ast.AsyncWith, ast.FunctionDef, ast.AsyncFunctionDef))
    model = ev.data["model"]
    if isinstance(block, (ast.FunctionDef, ast.AsyncFunctionDef)):
        first = block.body[0]
        ctx.editor.insert_before_line(first.lineno, f"{ctx.editor.line_indent(first.lineno)}{model}.eval()")
    else:
        ctx.editor.insert_before_line(block.lineno, f"{ctx.editor.line_indent(block.lineno)}{model}.eval()")
    if ev.data.get("needs_train") and ev.data.get("train_loop"):
        loop = a.find_node(ev.data["train_loop"], (ast.For, ast.While))
        first = loop.body[0]
        if first.lineno != block.lineno:
            ctx.editor.insert_before_line(first.lineno, f"{ctx.editor.line_indent(first.lineno)}{ev.data['train_model']}.train()")


# --------------------------------------------------------------------------- TM007
@rule(
    "TM007", "missing-train-mode", category="model_mode", cause="missing_train_mode", severity=Severity.ERROR,
    confidence=0.85,
    description="model.eval() is called during training but model.train() never is, so training continues in eval mode.",
)
def missing_train(ctx: LintContext):
    a = ctx.a
    if any(_calls_train_mode(c) for c in a.calls):
        return
    eval_calls = [c for c in a.calls if _calls_eval_mode(c)]
    loops = a.training_loops()
    if not eval_calls or not loops:
        return
    tl = loops[0]
    ancestors = [tl.loop]
    cur = a.enclosing_loop(tl.loop)
    while cur is not None:
        ancestors.append(cur)
        cur = a.enclosing_loop(cur)
    called_in_loops = {
        dotted(c.func) for loop in ancestors for c in a.calls_in(loop)
    }
    for ec in eval_calls:
        affects = any(a.contains(l, ec) for l in ancestors)
        fn = a.enclosing_function(ec)
        if not affects and fn is not None and fn.name in {last_part(n) for n in called_in_loops}:
            affects = True
        if not affects and fn is a.enclosing_function(tl.loop) and ec.lineno < tl.loop.lineno:
            affects = True
        if affects:
            model = _model_in_training_loop(ctx, tl) or receiver(ec)
            yield ctx.finding(
                "TM007", ec,
                f"`{receiver(ec)}.eval()` runs before/between training steps but `.train()` is never called; later steps train in eval mode.",
                confidence=0.85 if has_dropout_or_norm(ctx) else 0.3,
                severity=Severity.ERROR if has_dropout_or_norm(ctx) else Severity.INFO,
                data={"model": model, "loop": node_key(tl.loop)},
            )
            return


@fixer("TM007")
def _fix_missing_train(ctx: LintContext, ev):
    loop = ctx.a.find_node(ev.data["loop"], (ast.For, ast.While))
    first = loop.body[0]
    ctx.editor.insert_before_line(first.lineno, f"{ctx.editor.line_indent(first.lineno)}{ev.data['model']}.train()")


# ------------------------------------------------------------- TM008 / TM009
def _act_input(call: ast.Call) -> ast.AST | None:
    """For ``F.softmax(x, dim)`` / ``torch.sigmoid(x)`` / ``self.softmax(x)`` return ``x``;
    for the method form ``x.softmax(1)`` return ``x``."""
    cn = call_name(call)
    head = cn.split(".")[0]
    if isinstance(call.func, ast.Attribute) and head not in ("F", "torch", "nn", "functional", "nnf", "self"):
        return call.func.value
    return call.args[0] if call.args else None


def _is_activation(call: ast.AST, names: set[str], self_acts: set[str]) -> bool:
    if not isinstance(call, ast.Call):
        return False
    cn = call_name(call)
    if last_part(cn) in {n.lower() for n in names} and not cn.startswith("self."):
        return True
    return cn in self_acts


def _self_activation_attrs(ctx: LintContext, cls_names: set[str]) -> set[str]:
    out = set()
    for name, items in ctx.a.assignments.items():
        for _s, v in items:
            if name.startswith("self.") and isinstance(v, ast.Call) and last_part(call_name(v)) in cls_names:
                out.add(name)
    return out


def _activation_before_loss(ctx: LintContext, rule_id: str, act: str, act_cls: str, loss_names: set[str], loss_classes: set[str], loss_label: str):
    a = ctx.a
    self_acts = _self_activation_attrs(ctx, {act_cls})
    names = {act}
    uses_loss = any(a.is_loss_call(c, loss_names, loss_classes) for c in a.calls) or any(
        k in loss_classes for k in a.criteria.values()
    )
    if not uses_loss:
        return
    # (B) direct data flow into the loss call
    for c in a.calls:
        if not a.is_loss_call(c, loss_names, loss_classes) or not c.args:
            continue
        arg0 = c.args[0]
        if _is_activation(arg0, names, self_acts):
            inp = _act_input(arg0)  # type: ignore[arg-type]
            yield ctx.finding(
                rule_id, c, f"{loss_label} receives `{a.text(arg0)}`, but it applies {act} internally; {act} is applied twice.",
                data={"mode": "inline", "act": node_key(arg0), "input": node_key(inp) if inp is not None else None},
            )
        elif isinstance(arg0, ast.Name):
            val = a.last_assignment_before(arg0.id, c)
            if _is_activation(val, names, self_acts):
                inp = _act_input(val)  # type: ignore[arg-type]
                yield ctx.finding(
                    rule_id, c, f"`{arg0.id}` is the output of {act} and is passed to {loss_label}, which applies {act} again.",
                    data={"mode": "inline", "act": node_key(val), "input": node_key(inp) if inp is not None else None},
                )
    # (A) a model's forward() returns the activation
    for cls in a.module_classes.values():
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "forward":
                for n in iter_nodes(fn):
                    if isinstance(n, ast.Return) and _is_activation(n.value, names, self_acts):
                        inp = _act_input(n.value)  # type: ignore[arg-type]
                        yield ctx.finding(
                            rule_id, n,
                            f"`{cls.name}.forward` returns {act} probabilities, but the training loss is {loss_label}, which expects logits.",
                            confidence=0.8,
                            data={"mode": "inline", "act": node_key(n.value), "input": node_key(inp) if inp is not None else None},
                        )
    # (C) nn.Sequential(..., nn.Softmax())
    for c in a.calls:
        if last_part(call_name(c)) == "Sequential" and c.args:
            last = c.args[-1]
            if isinstance(last, ast.Call) and last_part(call_name(last)) == act_cls and len(c.args) > 1:
                yield ctx.finding(
                    rule_id, last,
                    f"nn.Sequential ends with {act_cls} but the loss is {loss_label}, which expects raw logits.",
                    confidence=0.8,
                    data={"mode": "sequential", "seq": node_key(c), "index": len(c.args) - 1},
                )


def _fix_activation(ctx: LintContext, ev):
    a = ctx.a
    if ev.data["mode"] == "inline":
        act = a.find_node(ev.data["act"], ast.Call)
        if ev.data.get("input") is None:
            return
        inp = a.find_node(ev.data["input"])
        ctx.editor.replace_node(act, a.text(inp))
    else:
        seq = a.find_node(ev.data["seq"], ast.Call)
        i = ev.data["index"]
        prev, last = seq.args[i - 1], seq.args[i]
        s = ctx.editor.span(prev)[1]
        e = ctx.editor.span(last)[1]
        ctx.editor.replace_span(s, e, "")


@rule(
    "TM008", "softmax-before-cross-entropy", category="loss", cause="double_softmax", severity=Severity.ERROR,
    confidence=0.9,
    description="CrossEntropyLoss already applies log-softmax; feeding it softmax probabilities squashes gradients.",
)
def softmax_before_ce(ctx: LintContext):
    yield from _activation_before_loss(ctx, "TM008", "softmax", "Softmax", CE_NAMES, CE_CLASSES, "cross-entropy")


@rule(
    "TM009", "sigmoid-before-bce-with-logits", category="loss", cause="double_sigmoid", severity=Severity.ERROR,
    confidence=0.9,
    description="BCEWithLogitsLoss already applies a sigmoid; feeding it probabilities applies the sigmoid twice.",
)
def sigmoid_before_bce(ctx: LintContext):
    yield from _activation_before_loss(ctx, "TM009", "sigmoid", "Sigmoid", BCE_LOGITS_NAMES, BCE_LOGITS_CLASSES, "BCE-with-logits")


fixer("TM008")(_fix_activation)
fixer("TM009")(_fix_activation)


# --------------------------------------------------------------------------- TM010
@rule(
    "TM010", "nll-on-raw-logits", category="loss", cause="nll_without_log_softmax", severity=Severity.ERROR,
    confidence=0.7,
    description="NLLLoss expects log-probabilities, but no log_softmax is applied anywhere in this file.",
)
def nll_without_log_softmax(ctx: LintContext):
    a = ctx.a
    if a.file_mentions(r"log_softmax|LogSoftmax|\.log\(|torch\.log\b|logsigmoid"):
        return
    for c in a.calls:
        if a.is_loss_call(c, NLL_NAMES, NLL_CLASSES) and c.args:
            yield ctx.finding(
                "TM010", c, "NLL loss is applied to raw scores; it expects log-probabilities (apply log_softmax first).",
                data={"arg": node_key(c.args[0])},
            )


@fixer("TM010")
def _fix_nll(ctx: LintContext, ev):
    arg = ctx.a.find_node(ev.data["arg"])
    fn = "torch.log_softmax" if ctx.a.imports_module("torch") else "F.log_softmax"
    ctx.editor.replace_node(arg, f"{fn}({ctx.a.text(arg)}, dim=-1)")


# --------------------------------------------------------------------------- TM011
def _backward_loss_expr(call: ast.Call) -> ast.AST | None:
    if method(call) != "backward":
        return None
    r = call.func.value  # type: ignore[attr-defined]
    rname = dotted(r)
    if rname in ("accelerator", "self.accelerator", "fabric", "self.fabric", "torch.autograd", "autograd"):
        return call.args[0] if call.args else None
    if isinstance(r, ast.Call) and method(r) == "scale" and r.args:  # scaler.scale(loss).backward()
        return r.args[0]
    return r


def _mod_divisor(test: ast.AST) -> ast.AST | None:
    for n in ast.walk(test):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod):
            return n.right
    return None


@rule(
    "TM011", "grad-accumulation-unscaled", category="optimization", cause="grad_accum_unscaled",
    severity=Severity.ERROR, confidence=0.8,
    description="Gradients are accumulated over N micro-batches but the loss is not divided by N, multiplying the effective LR by N.",
)
def grad_accum_unscaled(ctx: LintContext):
    a = ctx.a
    for tl in a.training_loops():
        step = tl.step_calls[0]
        cond = a.parent(step)
        guard = None
        while cond is not None and cond is not tl.loop:
            if isinstance(cond, ast.If) and _mod_divisor(cond.test) is not None:
                guard = cond
                break
            cond = a.parent(cond)
        if guard is None:
            continue
        acc = _mod_divisor(guard.test)
        acc_text = a.text(acc)
        for bw in tl.backward_calls:
            if a.contains(guard, bw):
                continue
            loss = _backward_loss_expr(bw)
            if loss is None:
                continue
            if isinstance(loss, ast.BinOp) and isinstance(loss.op, ast.Div):
                continue
            name = dotted(loss)
            scaled = False
            for stmt, value in a.assignments.get(name, []):
                if not a.contains(tl.loop, stmt) and a.enclosing_function(stmt) is not a.enclosing_function(tl.loop):
                    continue
                txt = a.text(stmt)
                if isinstance(stmt, ast.AugAssign) and isinstance(stmt.op, ast.Div):
                    scaled = True
                elif "/" in txt and (acc_text in txt or "accum" in txt or "grad_acc" in txt):
                    scaled = True
            if not scaled:
                yield ctx.finding(
                    "TM011", bw,
                    f"the optimizer steps every `{acc_text}` backward passes but the loss is not divided by `{acc_text}`; the update is {acc_text}x too large.",
                    data={"loss": node_key(loss), "acc": acc_text},
                )
                break


@fixer("TM011")
def _fix_grad_accum(ctx: LintContext, ev):
    loss = ctx.a.find_node(ev.data["loss"])
    ctx.editor.replace_node(loss, f"({ctx.a.text(loss)} / {ev.data['acc']})")


# ------------------------------------------------------------- TM012 / TM013
def _loader_calls(ctx: LintContext):
    for name, items in ctx.a.assignments.items():
        for stmt, value in items:
            if isinstance(value, ast.Call) and last_part(call_name(value)) == "DataLoader":
                yield name, value


def _loader_dataset(ctx: LintContext, call: ast.Call) -> str:
    if call.args:
        return ctx.a.text(call.args[0])
    kw = kwarg(call, "dataset")
    return ctx.a.text(kw.value) if kw else ""


TRAIN_RE = re.compile(r"train", re.I)
EVAL_RE = re.compile(r"val|test|eval|dev|holdout", re.I)


@rule(
    "TM012", "train-loader-not-shuffled", category="data", cause="no_shuffle", severity=Severity.WARNING,
    confidence=0.55,
    description="The training DataLoader does not shuffle; ordered data produces biased, oscillating updates.",
)
def train_loader_no_shuffle(ctx: LintContext):
    for name, call in _loader_calls(ctx):
        ds = _loader_dataset(ctx, call)
        if not (TRAIN_RE.search(name) or TRAIN_RE.search(ds)) or EVAL_RE.search(name):
            continue
        if kwarg(call, "sampler") or kwarg(call, "batch_sampler"):
            continue
        sh = kwarg(call, "shuffle")
        if sh is None or (isinstance(sh.value, ast.Constant) and sh.value.value is False):
            yield ctx.finding(
                "TM012", call, f"training loader `{name}` is created without shuffle=True.",
                confidence=0.6 if sh is not None else 0.55,
                data={"call": node_key(call)},
            )


@fixer("TM012")
def _fix_shuffle(ctx: LintContext, ev):
    call = ctx.a.find_node(ev.data["call"], ast.Call)
    sh = kwarg(call, "shuffle")
    if sh is not None:
        ctx.editor.replace_node(sh.value, "True")
        return
    last = max(list(call.args) + [k.value for k in call.keywords], key=lambda n: (n.end_lineno, n.end_col_offset))
    pos = ctx.editor.span(last)[1]
    ctx.editor.replace_span(pos, pos, ", shuffle=True")


@rule(
    "TM013", "eval-on-training-data", category="data", cause="train_eval_leakage", severity=Severity.ERROR,
    confidence=0.75,
    description="An evaluation DataLoader is built from the training dataset, so reported metrics are inflated.",
)
def eval_on_train_data(ctx: LintContext):
    loaders = list(_loader_calls(ctx))
    train_ds = {_loader_dataset(ctx, c) for n, c in loaders if TRAIN_RE.search(n) and not EVAL_RE.search(n)}
    for name, call in loaders:
        if not EVAL_RE.search(name) or TRAIN_RE.search(name):
            continue
        ds = _loader_dataset(ctx, call)
        if ds and ds in train_ds:
            yield ctx.finding("TM013", call, f"evaluation loader `{name}` uses the same dataset (`{ds}`) as the training loader.")
        elif ds and TRAIN_RE.search(ds) and not EVAL_RE.search(ds):
            yield ctx.finding(
                "TM013", call, f"evaluation loader `{name}` is built from `{ds}`, which looks like training data.", confidence=0.65
            )


# --------------------------------------------------------------------------- TM014
DROPOUT_FUNCS = {"dropout", "dropout1d", "dropout2d", "dropout3d", "alpha_dropout", "feature_alpha_dropout"}


@rule(
    "TM014", "functional-dropout-always-on", category="model_mode", cause="dropout_in_eval", severity=Severity.ERROR,
    confidence=0.85,
    description="F.dropout defaults to training=True, so it stays active in eval mode unless passed training=self.training.",
)
def functional_dropout(ctx: LintContext):
    a = ctx.a
    for c in a.calls:
        cn = call_name(c)
        if last_part(cn) not in DROPOUT_FUNCS or not (cn.startswith(("F.", "nnf.", "functional.")) or "nn.functional" in cn):
            continue
        if kwarg(c, "training") is not None or len(c.args) >= 3:
            continue
        fn = a.enclosing_function(c)
        if fn is None or not fn.args.args or fn.args.args[0].arg != "self":
            continue
        yield ctx.finding("TM014", c, f"`{cn}(...)` is missing `training=self.training`; dropout stays active during evaluation.",
                          data={"call": node_key(c)})


@fixer("TM014")
def _fix_dropout(ctx: LintContext, ev):
    call = ctx.a.find_node(ev.data["call"], ast.Call)
    last = max(list(call.args) + [k.value for k in call.keywords], key=lambda n: (n.end_lineno, n.end_col_offset))
    pos = ctx.editor.span(last)[1]
    ctx.editor.replace_span(pos, pos, ", training=self.training")


# --------------------------------------------------------------------------- TM015
def _is_detached(ctx: LintContext, node: ast.AST) -> ast.AST | None:
    """Return the detaching node if ``node`` is ``x.detach()`` or ``x.data``."""
    if isinstance(node, ast.Call) and method(node) == "detach":
        return node
    if isinstance(node, ast.Attribute) and node.attr == "data":
        return node
    return None


@rule(
    "TM015", "detached-prediction-in-loss", category="graph", cause="detached_graph", severity=Severity.ERROR,
    confidence=0.8,
    description="The prediction passed to the loss is detached (or computed under no_grad), so no gradient reaches the model.",
)
def detached_prediction(ctx: LintContext):
    a = ctx.a
    backproped = {dotted(_backward_loss_expr(c)) for c in a.calls if a.is_backward(c) and _backward_loss_expr(c) is not None}
    for c in a.calls:
        if not a.is_any_loss_call(c) or not c.args:
            continue
        st = a.enclosing_stmt(c)
        targets = []
        if isinstance(st, ast.Assign):
            targets = [dotted(t) for t in st.targets]
        if not set(targets) & backproped:
            continue
        arg0 = c.args[0]
        det = _is_detached(ctx, arg0)
        if det is not None:
            yield ctx.finding("TM015", c, f"`{a.text(arg0)}` is detached from the graph before the loss; the model gets no gradient.",
                              data={"detach": node_key(det)})
            continue
        if isinstance(arg0, ast.Name):
            val = a.last_assignment_before(arg0.id, c)
            if val is not None:
                det = _is_detached(ctx, val)
                if det is not None:
                    yield ctx.finding("TM015", c, f"`{arg0.id}` was detached (`{a.text(val)}`) before being used as the prediction in the loss.",
                                      data={"detach": node_key(det)})
                elif in_no_grad(ctx, val) and not in_no_grad(ctx, c):
                    yield ctx.finding("TM015", c, f"`{arg0.id}` is computed under no_grad() and then used as the prediction in a backpropagated loss.",
                                      confidence=0.7, data={})


@fixer("TM015")
def _fix_detach(ctx: LintContext, ev):
    if "detach" not in ev.data:
        return
    det = ctx.a.find_node(ev.data["detach"])
    inner = det.func.value if isinstance(det, ast.Call) else det.value  # type: ignore[attr-defined]
    ctx.editor.replace_node(det, ctx.a.text(inner))


# --------------------------------------------------------------------------- TM016
@rule(
    "TM016", "tensor-loss-accumulation", category="resource", cause="graph_retention", severity=Severity.INFO,
    confidence=0.6,
    description="A running total accumulates the loss tensor itself (not loss.item()), keeping every graph alive.",
)
def tensor_loss_accumulation(ctx: LintContext):
    a = ctx.a
    backproped = {dotted(_backward_loss_expr(c)) for c in a.calls if a.is_backward(c) and _backward_loss_expr(c) is not None}
    for node in ast.walk(ctx.tree):
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add) and isinstance(node.value, ast.Name):
            if node.value.id in backproped and a.enclosing_loop(node) is not None:
                yield ctx.finding("TM016", node, f"`{a.text(node)}` accumulates a tensor with its autograd graph; use `.item()`.",
                                  data={"value": node_key(node.value)})


@fixer("TM016")
def _fix_accum(ctx: LintContext, ev):
    v = ctx.a.find_node(ev.data["value"])
    ctx.editor.replace_node(v, f"{ctx.a.text(v)}.item()")


# --------------------------------------------------------------------------- TM029
@rule(
    "TM029", "direct-forward-call", category="model_mode", cause="hooks_bypassed", severity=Severity.INFO,
    confidence=0.4,
    description="Calling model.forward(...) directly bypasses module hooks (including DDP and torch.compile wrappers).",
)
def direct_forward(ctx: LintContext):
    a = ctx.a
    for c in a.calls:
        if method(c) == "forward" and receiver(c) in a.models and receiver(c) not in ("self", "super()"):
            yield ctx.finding("TM029", c, f"`{receiver(c)}.forward(...)` bypasses hooks; call `{receiver(c)}(...)`.",
                              data={"call": node_key(c)})


@fixer("TM029")
def _fix_forward(ctx: LintContext, ev):
    c = ctx.a.find_node(ev.data["call"], ast.Call)
    ctx.editor.replace_node(c.func, receiver(c))


# --------------------------------------------------------------------------- TM030
@rule(
    "TM030", "no-seed", category="reproducibility", cause="unseeded", severity=Severity.INFO, confidence=0.3,
    description="No random seed is set; runs are not reproducible, which makes debugging and validation noisy.",
)
def no_seed(ctx: LintContext):
    a = ctx.a
    loops = a.training_loops()
    if loops and not a.file_mentions(r"manual_seed|seed_everything|set_seed|random\.seed|np\.random\.seed|PRNGKey|trainmedic\.seed"):
        yield ctx.finding("TM030", loops[0].loop, "no random seed is set anywhere in this script.")


# --------------------------------------------------------------------------- TM034
@rule(
    "TM034", "optimizer-param-subset", category="optimization", cause="optimizer_coverage", severity=Severity.INFO,
    confidence=0.4,
    description="The optimizer only receives a sub-module's parameters while the rest of the model still requires grad.",
)
def optimizer_subset(ctx: LintContext):
    a = ctx.a
    if a.file_mentions(r"requires_grad\s*=\s*False|requires_grad_\(\s*False|freeze"):
        return
    for name in a.optimizers:
        for stmt, value in a.assignments.get(name, []):
            if isinstance(value, ast.Call) and value.args:
                arg = value.args[0]
                for n in ast.walk(arg):
                    if isinstance(n, ast.Call) and method(n) == "parameters":
                        r = receiver(n)
                        if "." in r and r.split(".")[0] in a.models and not r.startswith("self."):
                            yield ctx.finding("TM034", value, f"optimizer `{name}` only updates `{r}`; other parameters of `{r.split('.')[0]}` are never trained.")
                            break
