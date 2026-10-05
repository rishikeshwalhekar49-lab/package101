"""Light-weight AST analysis of training scripts.

We do not attempt full data-flow analysis. Instead we recover the handful of
facts that matter for training-loop bugs: which names are optimizers,
schedulers, loss criteria and models, which loops are training loops, and the
order of the key calls (``zero_grad``/``backward``/``clip``/``step``) inside them.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

OPTIMIZER_CLASSES = {
    "SGD", "Adam", "AdamW", "RMSprop", "Adagrad", "Adadelta", "Adamax", "NAdam", "RAdam",
    "LBFGS", "SparseAdam", "ASGD", "Rprop", "Lion", "Adafactor", "AdamW8bit", "Adam8bit",
    "FusedAdam", "LAMB", "Lamb", "Muon", "Shampoo", "SOAP", "ZeroRedundancyOptimizer",
}
SCHEDULER_CLASSES = {
    "StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR", "OneCycleLR", "LambdaLR",
    "ReduceLROnPlateau", "CosineAnnealingWarmRestarts", "LinearLR", "ConstantLR", "SequentialLR",
    "CyclicLR", "PolynomialLR", "ChainedScheduler", "MultiplicativeLR",
}
PER_STEP_SCHEDULERS = {"OneCycleLR", "CyclicLR"}
HF_SCHEDULER_FUNCS = re.compile(r"get_\w*schedule\w*")
EPOCH_SCHEDULERS = {"StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR", "PolynomialLR"}
LOSS_CLASSES = {
    "CrossEntropyLoss", "NLLLoss", "BCELoss", "BCEWithLogitsLoss", "MSELoss", "L1Loss",
    "SmoothL1Loss", "HuberLoss", "KLDivLoss", "CTCLoss", "MarginRankingLoss", "HingeEmbeddingLoss",
    "CosineEmbeddingLoss", "TripletMarginLoss", "PoissonNLLLoss", "GaussianNLLLoss", "MultiLabelSoftMarginLoss",
}
FUNCTIONAL_LOSSES = {
    "cross_entropy", "nll_loss", "binary_cross_entropy", "binary_cross_entropy_with_logits", "mse_loss",
    "l1_loss", "smooth_l1_loss", "huber_loss", "kl_div",
}
CE_NAMES = {"cross_entropy"}
CE_CLASSES = {"CrossEntropyLoss"}
BCE_LOGITS_NAMES = {"binary_cross_entropy_with_logits"}
BCE_LOGITS_CLASSES = {"BCEWithLogitsLoss"}
NLL_NAMES = {"nll_loss"}
NLL_CLASSES = {"NLLLoss"}
SOFTMAX_NAMES = {"softmax", "Softmax"}
MODEL_NAME_RE = re.compile(r"^(model|net|network|\w+_model|model_\w+|\w+_net|policy|\w+_policy|classifier|encoder|decoder|backbone)$")
SIGMOID_NAMES = {"sigmoid", "Sigmoid"}


def dotted(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        base = dotted(node.func)
        return f"{base}()" if base else ""
    if isinstance(node, ast.Subscript):
        base = dotted(node.value)
        return f"{base}[]" if base else ""
    return ""


def call_name(call: ast.Call) -> str:
    return dotted(call.func)


def last_part(name: str) -> str:
    return name.rsplit(".", 1)[-1] if name else ""


def receiver(call: ast.Call) -> str:
    if isinstance(call.func, ast.Attribute):
        return dotted(call.func.value)
    return ""


def method(call: ast.Call) -> str:
    return call.func.attr if isinstance(call.func, ast.Attribute) else ""


def root_name(node: ast.AST) -> str:
    """The base variable of an expression: ``a.b[0].view(-1)`` -> ``a``."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            node = node.value
        elif isinstance(node, ast.Subscript):
            node = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            node = node.func.value
        elif isinstance(node, ast.Starred):
            node = node.value
        else:
            return ""


def kwarg(call: ast.Call, name: str) -> ast.keyword | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw
    return None


def const_value(node: ast.AST | None):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        v = node.operand.value
        if isinstance(v, (int, float)):
            return -v
    return None


def node_key(node: ast.AST) -> list[int]:
    return [node.lineno, node.col_offset, node.end_lineno, node.end_col_offset]  # type: ignore[attr-defined]


def iter_nodes(node: ast.AST, skip_defs: bool = True):
    """Walk ``node`` without descending into nested function/class definitions."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        yield n
        if skip_defs and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(n))


def sorted_nodes(nodes):
    return sorted(nodes, key=lambda n: (n.lineno, n.col_offset))


@dataclass
class TrainingLoop:
    loop: ast.For | ast.While
    backward_calls: list[ast.Call] = field(default_factory=list)
    step_calls: list[ast.Call] = field(default_factory=list)
    zero_grad_calls: list[ast.Call] = field(default_factory=list)
    clip_calls: list[ast.Call] = field(default_factory=list)
    sched_calls: list[ast.Call] = field(default_factory=list)
    grad_none_assigns: list[ast.stmt] = field(default_factory=list)


class Analysis:
    def __init__(self, tree: ast.Module, source: str) -> None:
        self.tree = tree
        self.source = source
        self.parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self.parents[child] = node
        self.calls: list[ast.Call] = sorted_nodes([n for n in ast.walk(tree) if isinstance(n, ast.Call)])
        self.assignments: dict[str, list[tuple[ast.stmt, ast.AST]]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    for name in _target_names(t):
                        self.assignments.setdefault(name, []).append((node, node.value))
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                for name in _target_names(node.target):
                    self.assignments.setdefault(name, []).append((node, node.value))
            elif isinstance(node, ast.AugAssign):
                for name in _target_names(node.target):
                    self.assignments.setdefault(name, []).append((node, node))
        self.module_classes = self._module_classes()
        self.optimizers = self._find_optimizers()
        self.schedulers = self._find_schedulers()
        self.criteria = self._find_criteria()
        self.models = self._find_models()
        self.imports = self._imports()

    # ------------------------------------------------------------ structure
    def parent(self, node: ast.AST) -> ast.AST | None:
        return self.parents.get(node)

    def enclosing(self, node: ast.AST, types) -> ast.AST | None:
        cur = self.parents.get(node)
        while cur is not None and not isinstance(cur, types):
            cur = self.parents.get(cur)
        return cur

    def enclosing_stmt(self, node: ast.AST) -> ast.stmt | None:
        cur: ast.AST | None = node
        while cur is not None and not isinstance(cur, ast.stmt):
            cur = self.parents.get(cur)
        return cur  # type: ignore[return-value]

    def enclosing_function(self, node: ast.AST):
        return self.enclosing(node, (ast.FunctionDef, ast.AsyncFunctionDef))

    def enclosing_loop(self, node: ast.AST):
        return self.enclosing(node, (ast.For, ast.While, ast.AsyncFor))

    def siblings(self, stmt: ast.stmt) -> list[ast.stmt] | None:
        parent = self.parents.get(stmt)
        if parent is None:
            return None
        for fname in ("body", "orelse", "finalbody", "handlers"):
            seq = getattr(parent, fname, None)
            if isinstance(seq, list) and stmt in seq:
                return seq
        return None

    def calls_in(self, node: ast.AST, skip_defs: bool = True) -> list[ast.Call]:
        return sorted_nodes([n for n in iter_nodes(node, skip_defs) if isinstance(n, ast.Call)])

    def text(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.source, node) or ""

    def find_node(self, key: list[int], types=ast.AST) -> ast.AST | None:
        for n in ast.walk(self.tree):
            if isinstance(n, types) and hasattr(n, "lineno") and node_key(n) == list(key):
                return n
        return None

    # ---------------------------------------------------------------- names
    def _module_classes(self) -> dict[str, ast.ClassDef]:
        out: dict[str, ast.ClassDef] = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                bases = {last_part(dotted(b)) for b in node.bases}
                if bases & {"Module", "LightningModule", "PreTrainedModel", "Sequential"} or bases & set(out):
                    out[node.name] = node
        return out

    def _find_optimizers(self) -> set[str]:
        names: set[str] = set()
        for name, items in self.assignments.items():
            for _stmt, value in items:
                if isinstance(value, ast.Call):
                    cn = call_name(value)
                    lp = last_part(cn)
                    if lp in SCHEDULER_CLASSES or "lr_scheduler" in cn:
                        continue
                    if lp in OPTIMIZER_CLASSES or (".optim." in f".{cn}" and lp[:1].isupper()):
                        names.add(name)
        if not names:
            for call in self.calls:
                if method(call) in ("step", "zero_grad"):
                    r = receiver(call)
                    if "opt" in r.lower():
                        names.add(r)
        return names

    def _find_schedulers(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, items in self.assignments.items():
            for _stmt, value in items:
                if isinstance(value, ast.Call):
                    lp = last_part(call_name(value))
                    if lp in SCHEDULER_CLASSES or HF_SCHEDULER_FUNCS.fullmatch(lp or ""):
                        out[name] = lp
        for call in self.calls:
            if method(call) == "step":
                r = receiver(call)
                if r and r not in out and r not in self.optimizers and ("sched" in r.lower() or "lr_" in r.lower()):
                    out[r] = "unknown"
        return out

    def _find_criteria(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, items in self.assignments.items():
            for _stmt, value in items:
                if isinstance(value, ast.Call) and last_part(call_name(value)) in LOSS_CLASSES:
                    out[name] = last_part(call_name(value))
        return out

    def _find_models(self) -> set[str]:
        models: set[str] = set()
        for call in self.calls:
            if (method(call) in ("eval", "train") and not call.args) or method(call) in ("parameters", "named_parameters", "state_dict"):
                r = receiver(call)
                if r and r not in self.optimizers and r not in self.schedulers and "." not in r.replace("self.", "", 1):
                    models.add(r)
        changed = True
        while changed:  # fixpoint: ``ref = deepcopy(policy)`` needs ``policy`` first
            changed = False
            for name, items in self.assignments.items():
                if name in models:
                    continue
                for _stmt, value in items:
                    if self._is_model_expr(value, models):
                        models.add(name)
                        changed = True
                        break
        models.discard("")
        return models

    def _is_model_expr(self, value: ast.AST, known: set[str]) -> bool:
        if not isinstance(value, ast.Call):
            return isinstance(value, ast.Name) and value.id in known
        cn = call_name(value)
        lp = last_part(cn)
        if lp in self.module_classes or lp in ("Sequential", "from_pretrained", "deepcopy", "compile"):
            if lp in ("deepcopy", "compile"):
                return bool(value.args) and (root_name(value.args[0]) in known or self._is_model_expr(value.args[0], known))
            return True
        if lp in ("to", "cuda", "cpu", "half", "float", "bfloat16", "double") and isinstance(value.func, ast.Attribute):
            return self._is_model_expr(value.func.value, known)
        return False

    def _imports(self) -> set[str]:
        out: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    out.add(a.name)
                    if a.asname:
                        out.add(f"{a.name} as {a.asname}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    out.add(f"{node.module}.{a.name}")
        return out

    def imports_module(self, prefix: str) -> bool:
        return any(i == prefix or i.startswith(prefix + ".") or i.startswith(prefix + " ") for i in self.imports)

    # ------------------------------------------------------- call predicates
    def is_optimizer_step(self, call: ast.Call) -> bool:
        return method(call) == "step" and receiver(call) in self.optimizers

    def is_scheduler_step(self, call: ast.Call) -> bool:
        return method(call) == "step" and receiver(call) in self.schedulers

    @staticmethod
    def is_backward(call: ast.Call) -> bool:
        if method(call) == "backward":
            return True
        cn = call_name(call)
        return cn.endswith("autograd.backward") or last_part(cn) == "backward" and receiver(call) in ("accelerator", "self.accelerator", "fabric")

    def is_zero_grad(self, call: ast.Call) -> bool:
        return method(call) == "zero_grad"

    @staticmethod
    def is_clip(call: ast.Call) -> bool:
        return last_part(call_name(call)) in ("clip_grad_norm_", "clip_grad_value_", "clip_grad_norm", "clip_grad_value")

    def is_loss_call(self, call: ast.Call, kinds: set[str] | None = None, classes: set[str] | None = None) -> bool:
        cn = call_name(call)
        lp = last_part(cn)
        if kinds is not None and lp in kinds and ("functional" in cn or cn.startswith(("F.", "nnf.", "torch.")) or cn == lp):
            return True
        if isinstance(call.func, ast.Name) or isinstance(call.func, ast.Attribute):
            crit = dotted(call.func)
            if crit in self.criteria and (classes is None or self.criteria[crit] in classes):
                return True
        if isinstance(call.func, ast.Call) and classes is not None:  # nn.CrossEntropyLoss()(x, y)
            if last_part(call_name(call.func)) in classes:
                return True
        return False

    def is_any_loss_call(self, call: ast.Call) -> bool:
        return self.is_loss_call(call, FUNCTIONAL_LOSSES, None)

    def is_model_call(self, call: ast.Call) -> bool:
        if isinstance(call.func, (ast.Name, ast.Attribute)):
            name = dotted(call.func)
            if name in self.models or name == "self":
                return True
            lp = last_part(name)
            if MODEL_NAME_RE.match(lp) and not lp.startswith(("get_", "build_", "make_", "create_", "load_", "init_")):
                return True
        return False

    # ------------------------------------------------------------ loops
    def training_loops(self) -> list[TrainingLoop]:
        """Innermost loops that contain both a backward call and an optimizer step."""
        loops = [n for n in ast.walk(self.tree) if isinstance(n, (ast.For, ast.While, ast.AsyncFor))]
        found: list[TrainingLoop] = []
        for loop in loops:
            calls = self.calls_in(loop)
            if not any(self.is_backward(c) for c in calls) or not any(self.is_optimizer_step(c) for c in calls):
                continue
            inner = [l for l in loops if l is not loop and self._contains(loop, l)]
            if any(
                any(self.is_backward(c) for c in self.calls_in(l)) and any(self.is_optimizer_step(c) for c in self.calls_in(l))
                for l in inner
            ):
                continue
            tl = TrainingLoop(loop)
            for c in calls:
                if self.is_backward(c):
                    tl.backward_calls.append(c)
                elif self.is_optimizer_step(c):
                    tl.step_calls.append(c)
                elif self.is_zero_grad(c):
                    tl.zero_grad_calls.append(c)
                elif self.is_clip(c):
                    tl.clip_calls.append(c)
                elif self.is_scheduler_step(c):
                    tl.sched_calls.append(c)
            for n in iter_nodes(loop):
                if isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Attribute) and t.attr == "grad" for t in n.targets
                ):
                    tl.grad_none_assigns.append(n)
            found.append(tl)
        return found

    @staticmethod
    def _contains(outer: ast.AST, inner: ast.AST) -> bool:
        return (outer.lineno, outer.col_offset) <= (inner.lineno, inner.col_offset) and (  # type: ignore[attr-defined]
            outer.end_lineno, outer.end_col_offset) >= (inner.end_lineno, inner.end_col_offset)  # type: ignore[attr-defined]

    def contains(self, outer: ast.AST, inner: ast.AST) -> bool:
        return self._contains(outer, inner)

    # ----------------------------------------------------------- dataflow-ish
    def last_assignment_before(self, name: str, node: ast.AST) -> ast.AST | None:
        """Most recent assigned value of ``name`` that precedes ``node`` in the same function."""
        func = self.enclosing_function(node)
        best = None
        for stmt, value in self.assignments.get(name, []):
            if func is not None and self.enclosing_function(stmt) is not func:
                continue
            if (stmt.lineno, stmt.col_offset) < (node.lineno, node.col_offset):  # type: ignore[attr-defined]
                if best is None or (stmt.lineno, stmt.col_offset) > (best[0].lineno, best[0].col_offset):
                    best = (stmt, value)
        return best[1] if best else None

    def file_mentions(self, *patterns: str) -> bool:
        return any(re.search(p, self.source) for p in patterns)


def _target_names(t: ast.AST) -> list[str]:
    if isinstance(t, (ast.Tuple, ast.List)):
        out: list[str] = []
        for e in t.elts:
            out.extend(_target_names(e))
        return out
    name = dotted(t)
    return [name] if name else []
