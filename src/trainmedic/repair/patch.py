"""Patches as sets of independent line-range edits against the *original* files.

Because each :class:`Edit` addresses the original text, any subset of a patch's
edits can be applied on its own -- which is exactly what delta-debugging patch
minimisation needs.
"""

from __future__ import annotations

import difflib
import io
import re
import tokenize
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

HP_NAMES = re.compile(
    r"\b(lr|learning_rate|weight_decay|momentum|betas?|eps|batch_size|epochs?|n_epochs|num_epochs|dropout|p|gamma|"
    r"step_size|max_norm|clip|warmup\w*|accum\w*|grad_accum\w*|beta|kl_coef|temperature|label_smoothing|steps|max_steps|"
    r"hidden\w*|n_layers?|num_layers|heads|n_heads|d_model|seed)\b"
)


class PatchConflict(ValueError):
    pass


@dataclass
class Edit:
    file: str
    start: int
    end: int
    old: list[str]
    new: list[str]
    kind: str = "code"

    @property
    def size(self) -> int:
        return len(self.old) + len(self.new)

    def overlaps(self, other: "Edit") -> bool:
        if self.file != other.file:
            return False
        a0, a1, b0, b1 = self.start, self.end, other.start, other.end
        if a0 == a1 and b0 == b1:
            return a0 == b0
        if a0 == a1:
            return b0 < a0 < b1
        if b0 == b1:
            return a0 < b0 < a1
        return a0 < b1 and b0 < a1

    def same(self, other: "Edit") -> bool:
        return (self.file, self.start, self.end, self.old, self.new) == (other.file, other.start, other.end, other.old, other.new)


@dataclass
class Patch:
    edits: list[Edit] = field(default_factory=list)
    description: str = ""
    origin: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ building
    @classmethod
    def from_sources(cls, file: str, old: str, new: str, **kw: Any) -> "Patch":
        a = old.splitlines(keepends=True)
        b = new.splitlines(keepends=True)
        edits = []
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            e = Edit(file, i1, i2, a[i1:i2], b[j1:j2])
            e.kind = classify_edit(e)
            edits.append(e)
        return cls(edits, **kw)

    @classmethod
    def from_file_pairs(cls, pairs: dict[str, tuple[str, str]], **kw: Any) -> "Patch":
        edits: list[Edit] = []
        for f, (old, new) in pairs.items():
            edits.extend(cls.from_sources(f, old, new).edits)
        return cls(edits, **kw)

    @classmethod
    def from_search_replace(cls, file: str, source: str, pairs: Iterable[tuple[str, str]], **kw: Any) -> "Patch":
        text = source
        for search, replace in pairs:
            text = search_replace(text, search, replace)
        return cls.from_sources(file, source, text, **kw)

    @classmethod
    def from_unified_diff(cls, diff: str, sources: dict[str, str], **kw: Any) -> "Patch":
        new = apply_unified_diff(diff, sources)
        return cls.from_file_pairs({f: (sources[f], new[f]) for f in new if new[f] != sources[f]}, **kw)

    # ------------------------------------------------------------ algebra
    def __len__(self) -> int:
        return len(self.edits)

    @property
    def files(self) -> list[str]:
        return sorted({e.file for e in self.edits})

    @property
    def size(self) -> int:
        return sum(e.size for e in self.edits)

    @property
    def kinds(self) -> set[str]:
        return {e.kind for e in self.edits}

    def subset(self, indices: Iterable[int]) -> "Patch":
        idx = sorted(set(indices))
        return Patch([self.edits[i] for i in idx], self.description, self.origin, list(self.evidence_ids), dict(self.meta))

    def merge(self, other: "Patch") -> "Patch":
        edits = list(self.edits)
        for e in other.edits:
            if any(x.same(e) for x in edits):
                continue
            if any(x.overlaps(e) for x in edits):
                raise PatchConflict(f"edits overlap in {e.file} around line {e.start + 1}")
            edits.append(e)
        return Patch(
            edits,
            "; ".join(d for d in (self.description, other.description) if d),
            "+".join(o for o in (self.origin, other.origin) if o),
            list(dict.fromkeys(self.evidence_ids + other.evidence_ids)),
        )

    def apply(self, sources: dict[str, str]) -> dict[str, str]:
        out = dict(sources)
        for f in self.files:
            if f not in out:
                raise PatchConflict(f"patch touches unknown file {f}")
            lines = out[f].splitlines(keepends=True)
            fe = sorted((e for e in self.edits if e.file == f), key=lambda e: (e.start, e.end), reverse=True)
            for i, a in enumerate(fe):
                for b in fe[i + 1 :]:
                    if a.overlaps(b) and not a.same(b):
                        raise PatchConflict(f"overlapping edits in {f}")
            for e in fe:
                if lines[e.start : e.end] != e.old:
                    raise PatchConflict(f"{f}:{e.start + 1}: context does not match")
                lines[e.start : e.end] = e.new
            out[f] = "".join(lines)
        return out

    def apply_files(self, root: str | Path = ".") -> None:
        root = Path(root)
        srcs = {f: (root / f).read_text(encoding="utf-8") for f in self.files}
        for f, text in self.apply(srcs).items():
            (root / f).write_text(text, encoding="utf-8")

    def to_unified_diff(self, sources: dict[str, str]) -> str:
        new = self.apply(sources)
        chunks = []
        for f in self.files:
            chunks.extend(
                difflib.unified_diff(
                    sources[f].splitlines(keepends=True), new[f].splitlines(keepends=True), f"a/{f}", f"b/{f}"
                )
            )
        return "".join(c if c.endswith("\n") else c + "\n" for c in chunks)

    def summary(self) -> str:
        kinds = ",".join(sorted(self.kinds)) or "-"
        return f"{len(self.edits)} edit(s), {self.size} changed line(s), kind={kinds}, origin={self.origin or '-'}"

    def risks(self) -> list[str]:
        out: list[str] = []
        for e in self.edits:
            for r in edit_risks(e):
                if r not in out:
                    out.append(r)
        return out

    # ------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        return {
            "edits": [asdict(e) for e in self.edits],
            "description": self.description,
            "origin": self.origin,
            "evidence_ids": self.evidence_ids,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Patch":
        return cls([Edit(**e) for e in d.get("edits", [])], d.get("description", ""), d.get("origin", ""),
                   list(d.get("evidence_ids", [])), dict(d.get("meta", {})))


# ---------------------------------------------------------------- helpers
def search_replace(text: str, search: str, replace: str) -> str:
    """Replace exactly one occurrence of ``search``; tolerate indentation drift."""
    if not search:
        raise PatchConflict("empty search block")
    n = text.count(search)
    if n == 1:
        return text.replace(search, replace, 1)
    if n > 1:
        raise PatchConflict("search block is ambiguous (matches several places)")
    # whitespace-insensitive line match
    lines = text.splitlines(keepends=True)
    s_lines = [l.strip() for l in search.strip("\n").splitlines()]
    for i in range(len(lines) - len(s_lines) + 1):
        if [l.strip() for l in lines[i : i + len(s_lines)]] == s_lines:
            indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
            s_indent = search.strip("\n").splitlines()[0]
            s_indent = s_indent[: len(s_indent) - len(s_indent.lstrip())]
            rep_lines = replace.strip("\n").splitlines()
            fixed = []
            for rl in rep_lines:
                body = rl[len(s_indent):] if rl.startswith(s_indent) else rl.lstrip()
                fixed.append((indent + body) if rl.strip() else "")
            tail = "\n" if lines[i + len(s_lines) - 1].endswith("\n") else ""
            new_block = "\n".join(fixed) + tail if fixed else ""
            return "".join(lines[:i]) + new_block + "".join(lines[i + len(s_lines) :])
    raise PatchConflict("search block not found")


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def apply_unified_diff(diff: str, sources: dict[str, str]) -> dict[str, str]:
    """Apply a unified diff (as produced by ``git diff`` / difflib) to in-memory sources."""
    out = dict(sources)
    current: str | None = None
    lines = diff.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("+++ "):
            name = line[4:].strip().split("\t")[0]
            name = name[2:] if name.startswith(("a/", "b/")) else name
            current = _match_file(name, out)
            i += 1
            continue
        m = _HUNK.match(line)
        if m and current is not None:
            old_block: list[str] = []
            new_block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "--- ", "+++ ", "diff ")):
                l = lines[i]
                if l.startswith("\\"):
                    i += 1
                    continue
                tag, body = (l[0], l[1:]) if l[:1] in (" ", "+", "-") else (" ", l)
                if tag in (" ", "-"):
                    old_block.append(body)
                if tag in (" ", "+"):
                    new_block.append(body)
                i += 1
            src = out[current].splitlines(keepends=True)
            start = int(m.group(1)) - 1 if old_block else int(m.group(1))
            pos = _find_block(src, old_block, max(0, start))
            if pos is None:
                raise PatchConflict(f"hunk at line {m.group(1)} of {current} does not apply")
            src[pos : pos + len(old_block)] = new_block
            out[current] = "".join(src)
            continue
        i += 1
    return out


def _match_file(name: str, sources: dict[str, str]) -> str:
    if name in sources:
        return name
    for f in sources:
        if f.endswith("/" + name) or name.endswith("/" + f) or Path(f).name == Path(name).name:
            return f
    raise PatchConflict(f"diff refers to unknown file {name}")


def _find_block(src: list[str], block: list[str], hint: int) -> int | None:
    if not block:
        return min(hint, len(src))
    norm = lambda xs: [x.rstrip("\n") for x in xs]  # noqa: E731
    nb = norm(block)
    order = sorted(range(len(src) - len(block) + 1), key=lambda p: abs(p - hint))
    for p in order:
        if norm(src[p : p + len(block)]) == nb:
            return p
    return None


def _tokens(text: str) -> list[tuple[int, str]]:
    try:
        return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(text).readline)
                if t.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return [(0, w) for w in text.split()]


def _hp_slot(tokens: list[tuple[int, str]], i: int) -> bool:
    """Is the literal at ``tokens[i]`` the value of a hyperparameter-named keyword or assignment?"""
    j = i - 1
    if j >= 0 and tokens[j][1] in ("-", "+"):
        j -= 1
    return j >= 1 and tokens[j][1] == "=" and tokens[j - 1][0] == tokenize.NAME and bool(HP_NAMES.fullmatch(tokens[j - 1][1]))


def classify_edit(e: Edit) -> str:
    """'hyperparameter' if the edit only changes literal values bound to hyperparameter-named
    keywords/variables (``lr=3e-3``, ``weight_decay = 0.1``); otherwise 'code'. A changed
    ``dim=-2`` or ``mask == 1`` is a code change even though only a number differs."""
    old, new = "".join(e.old), "".join(e.new)
    if not old.strip() or not new.strip():
        return "code"
    a, b = _tokens(old), _tokens(new)
    if len(a) == len(b):
        idx = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        literal = all(
            (a[i][0] == tokenize.NUMBER and b[i][0] == tokenize.NUMBER) or (a[i][1] in ("True", "False") and b[i][1] in ("True", "False"))
            for i in idx
        )
        if idx and literal and all(_hp_slot(a, i) for i in idx):
            return "hyperparameter"
    if len(e.old) == len(e.new) == 1:
        mo = re.match(r"\s*([\w.]+)\s*=(?!=)", old)
        mn = re.match(r"\s*([\w.]+)\s*=(?!=)", new)
        if mo and mn and mo.group(1) == mn.group(1) and HP_NAMES.fullmatch(mo.group(1).split(".")[-1]):
            return "hyperparameter"
    return "code"


RISK_PATTERNS = [
    ("removes_dropout", re.compile(r"[Dd]ropout"), "removed"),
    ("removes_mode_switch", re.compile(r"\.(eval|train)\("), "removed"),
    ("removes_zero_grad", re.compile(r"zero_grad"), "removed"),
    ("removes_normalization", re.compile(r"Normalize|BatchNorm|LayerNorm|normali[sz]e"), "removed"),
    ("removes_regularization", re.compile(r"weight_decay|clip_grad|label_smoothing"), "removed"),
    ("changes_lr", re.compile(r"\blr\b|learning_rate"), "changed"),
    ("changes_loss_reduction", re.compile(r"reduction\s*="), "changed"),
    ("changes_seed", re.compile(r"seed"), "changed"),
    ("changes_data_split", re.compile(r"random_split|Subset|train_test_split|\[:\s*\w+\]|split"), "changed"),
    ("changes_loss_function", re.compile(r"Loss\(|cross_entropy|mse_loss|nll_loss|binary_cross_entropy"), "changed"),
    ("changes_metric_computation", re.compile(r"accuracy|acc\b|correct|f1|metric|report\("), "changed"),
]


def edit_risks(e: Edit) -> list[str]:
    """Flags for edits that could silently damage a *correct* pipeline."""
    old, new = "".join(e.old), "".join(e.new)
    out = []
    for name, pat, mode in RISK_PATTERNS:
        o, n = len(pat.findall(old)), len(pat.findall(new))
        if mode == "removed" and o > n:
            out.append(name)
        elif mode == "changed" and (o or n) and old != new and (o != n or any(pat.search(l) for l in e.old)):
            out.append(name)
    if not e.new and len(e.old) >= 3:
        out.append("deletes_code")
    return out
