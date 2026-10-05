"""Rule registry and driver for the static analyser."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from ...sourceedit import EditConflict, SourceEditor, parses
from ...types import Evidence, Severity
from .analysis import Analysis, node_key


@dataclass
class Rule:
    id: str
    name: str
    category: str
    cause: str
    severity: Severity
    confidence: float
    description: str
    check: Callable[["LintContext"], Iterable[Evidence]]
    fix: Callable[["LintContext", Evidence], None] | None = None

    @property
    def fixable(self) -> bool:
        return self.fix is not None


RULES: dict[str, Rule] = {}


def rule(id: str, name: str, *, category: str, cause: str, severity: Severity, confidence: float, description: str):
    def deco(fn: Callable[["LintContext"], Iterable[Evidence]]):
        if id in RULES:
            raise ValueError(f"duplicate rule id {id}")
        RULES[id] = Rule(id, name, category, cause, severity, confidence, description, fn)
        return fn

    return deco


def fixer(id: str):
    def deco(fn: Callable[["LintContext", Evidence], None]):
        RULES[id].fix = fn
        return fn

    return deco


class LintContext:
    def __init__(self, source: str, filename: str = "<string>") -> None:
        self.source = source
        self.filename = filename
        self.tree = ast.parse(source)
        self.a = Analysis(self.tree, source)
        self.editor = SourceEditor(source)

    def finding(
        self,
        rule_id: str,
        node: ast.AST,
        message: str,
        *,
        confidence: float | None = None,
        severity: Severity | None = None,
        data: dict | None = None,
    ) -> Evidence:
        r = RULES[rule_id]
        d = {"node": node_key(node)}
        d.update(data or {})
        return Evidence(
            id="",
            source="static",
            rule=rule_id,
            message=message,
            severity=severity or r.severity,
            confidence=r.confidence if confidence is None else confidence,
            file=self.filename,
            line=getattr(node, "lineno", None),
            end_line=getattr(node, "end_lineno", None),
            category=r.category,
            cause=r.cause,
            fixable=r.fixable,
            data=d,
        )


def _ensure_rules_loaded() -> None:
    from . import rules_attention, rules_core, rules_frameworks, rules_rl  # noqa: F401


def all_rules() -> dict[str, Rule]:
    _ensure_rules_loaded()
    return RULES


def lint_source(
    source: str,
    filename: str = "<string>",
    *,
    select: Iterable[str] | None = None,
    ignore: Iterable[str] | None = None,
    min_confidence: float = 0.0,
) -> list[Evidence]:
    _ensure_rules_loaded()
    try:
        ctx = LintContext(source, filename)
    except SyntaxError as exc:
        return [
            Evidence(
                id="",
                source="static",
                rule="TM000",
                message=f"syntax error: {exc.msg}",
                severity=Severity.ERROR,
                confidence=1.0,
                file=filename,
                line=exc.lineno,
                category="syntax",
                cause="syntax_error",
            )
        ]
    sel = set(select) if select else None
    ign = set(ignore or ())
    out: list[Evidence] = []
    seen: set[tuple] = set()
    for rid, r in sorted(RULES.items()):
        if (sel is not None and rid not in sel) or rid in ign:
            continue
        try:
            for ev in r.check(ctx):
                key = (ev.rule, ev.line, ev.message)
                if key in seen or ev.confidence < min_confidence:
                    continue
                seen.add(key)
                out.append(ev)
        except Exception as exc:  # a buggy rule must never take the linter down
            out.append(
                Evidence(
                    id="",
                    source="static",
                    rule=rid,
                    message=f"internal rule error: {type(exc).__name__}: {exc}",
                    severity=Severity.INFO,
                    confidence=0.0,
                    file=filename,
                    category="internal",
                )
            )
    _suppress_noqa(source, out)
    return sorted(out, key=lambda e: (e.line or 0, e.rule))


def _suppress_noqa(source: str, findings: list[Evidence]) -> None:
    lines = source.splitlines()
    keep = []
    for ev in findings:
        if ev.line and 0 < ev.line <= len(lines):
            text = lines[ev.line - 1]
            if "# trainmedic: ignore" in text and (
                "ignore[" not in text or ev.rule in text.split("ignore[", 1)[1].split("]", 1)[0]
            ):
                continue
        keep.append(ev)
    findings[:] = keep


def lint_file(path: str | os.PathLike, **kw) -> list[Evidence]:
    p = Path(path)
    return lint_source(p.read_text(encoding="utf-8"), str(p), **kw)


SKIP_DIRS = {".git", ".hg", "__pycache__", ".venv", "venv", "env", "node_modules", "build", "dist", ".tox", ".mypy_cache"}


def iter_python_files(paths: Iterable[str | os.PathLike]) -> Iterator[Path]:
    for p in map(Path, paths):
        if p.is_dir():
            for root, dirs, files in os.walk(p):
                dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
                for f in sorted(files):
                    if f.endswith(".py"):
                        yield Path(root) / f
        elif p.suffix == ".py" or p.is_file():
            yield p


def lint_paths(paths: Iterable[str | os.PathLike], **kw) -> list[Evidence]:
    out: list[Evidence] = []
    for f in iter_python_files(paths):
        out.extend(lint_file(f, **kw))
    return out


def fix_finding(source: str, finding: Evidence) -> str | None:
    """Return ``source`` with the rule's template fix applied, or ``None``.

    The finding must have been produced from exactly this ``source`` (fix data
    refers to node positions)."""
    _ensure_rules_loaded()
    r = RULES.get(finding.rule)
    if r is None or r.fix is None:
        return None
    ctx = LintContext(source, finding.file or "<string>")
    try:
        r.fix(ctx, finding)
        if not ctx.editor.has_edits:
            return None
        new = ctx.editor.apply()
    except (EditConflict, KeyError, AttributeError, IndexError, TypeError, ValueError):
        return None
    if new == source or not parses(new):
        return None
    return new
