"""Formatting-preserving source edits addressed by AST node positions.

Fixers and mutation operators both need to change a few tokens of a file without
re-printing it (``ast.unparse`` would destroy comments and layout and make every
patch look huge). Edits are collected against the *original* text and applied
together, so their positions never shift underneath each other.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


def indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


@dataclass
class _Edit:
    start: int
    end: int
    text: str
    order: int


class EditConflict(ValueError):
    pass


class SourceEditor:
    def __init__(self, source: str) -> None:
        self.source = source
        self.lines = source.splitlines(keepends=True)
        self._starts = [0]
        for line in self.lines:
            self._starts.append(self._starts[-1] + len(line))
        self._edits: list[_Edit] = []

    # ---------------------------------------------------------------- positions
    def offset(self, lineno: int, col_byte: int) -> int:
        if lineno - 1 >= len(self.lines):
            return len(self.source)
        line = self.lines[lineno - 1]
        col = len(line.encode("utf-8")[:col_byte].decode("utf-8", errors="ignore"))
        return self._starts[lineno - 1] + col

    def span(self, node: ast.AST) -> tuple[int, int]:
        start = self.offset(node.lineno, node.col_offset)  # type: ignore[attr-defined]
        end = self.offset(node.end_lineno, node.end_col_offset)  # type: ignore[attr-defined]
        return start, end

    def text(self, node: ast.AST) -> str:
        s, e = self.span(node)
        return self.source[s:e]

    def line(self, lineno: int) -> str:
        return self.lines[lineno - 1] if 0 < lineno <= len(self.lines) else ""

    def line_indent(self, lineno: int) -> str:
        return indent_of(self.line(lineno))

    def line_start(self, lineno: int) -> int:
        return self._starts[min(lineno - 1, len(self._starts) - 1)]

    def line_end(self, lineno: int) -> int:
        return self._starts[min(lineno, len(self._starts) - 1)]

    # -------------------------------------------------------------------- edits
    def replace_span(self, start: int, end: int, text: str) -> None:
        self._edits.append(_Edit(start, end, text, len(self._edits)))

    def replace_node(self, node: ast.AST, text: str) -> None:
        s, e = self.span(node)
        self.replace_span(s, e, text)

    def insert_before_line(self, lineno: int, text: str) -> None:
        if not text.endswith("\n"):
            text += "\n"
        pos = self.line_start(lineno)
        self.replace_span(pos, pos, text)

    def insert_after_line(self, lineno: int, text: str) -> None:
        if not text.endswith("\n"):
            text += "\n"
        pos = self.line_end(lineno)
        if pos == len(self.source) and self.source and not self.source.endswith("\n"):
            text = "\n" + text
        self.replace_span(pos, pos, text)

    def insert_after_statement(self, stmt: ast.stmt, text: str) -> None:
        self.insert_after_line(stmt.end_lineno, text)  # type: ignore[arg-type]

    def delete_statement(self, stmt: ast.stmt, siblings: list[ast.stmt] | None = None) -> None:
        """Delete a whole statement. If it is the only statement in its block,
        replace it with ``pass`` so the result still parses."""
        first, last = stmt.lineno, stmt.end_lineno  # type: ignore[assignment]
        prefix_blank = self.source[self.line_start(first) : self.offset(first, stmt.col_offset)].strip() == ""
        tail = self.source[self.offset(last, stmt.end_col_offset) : self.line_end(last)].strip()  # type: ignore[arg-type]
        whole_lines = prefix_blank and (tail == "" or tail.startswith("#"))
        if siblings is not None and len(siblings) == 1:
            self.replace_node(stmt, "pass")
            return
        if whole_lines:
            self.replace_span(self.line_start(first), self.line_end(last), "")
        else:
            self.replace_node(stmt, "pass")

    def move_statement_after(self, stmt: ast.stmt, anchor: ast.stmt, siblings: list[ast.stmt] | None = None) -> None:
        indent = self.line_indent(anchor.lineno)
        body = self.statement_text(stmt)
        self.delete_statement(stmt, siblings)
        self.insert_after_statement(anchor, reindent(body, indent))

    def move_statement_before(self, stmt: ast.stmt, anchor: ast.stmt, siblings: list[ast.stmt] | None = None) -> None:
        indent = self.line_indent(anchor.lineno)
        body = self.statement_text(stmt)
        self.delete_statement(stmt, siblings)
        self.insert_before_line(anchor.lineno, reindent(body, indent))

    def statement_text(self, stmt: ast.stmt) -> str:
        return self.source[self.offset(stmt.lineno, stmt.col_offset) : self.offset(stmt.end_lineno, stmt.end_col_offset)]  # type: ignore[arg-type]

    def apply(self) -> str:
        edits = sorted(self._edits, key=lambda e: (e.start, e.end, e.order))
        for a, b in zip(edits, edits[1:]):
            if b.start < a.end:
                raise EditConflict(f"overlapping edits at offsets {a.start}-{a.end} and {b.start}-{b.end}")
        out = self.source
        for ed in sorted(edits, key=lambda e: (e.start, e.order), reverse=True):
            out = out[: ed.start] + ed.text + out[ed.end :]
        return out

    @property
    def has_edits(self) -> bool:
        return bool(self._edits)


def reindent(text: str, indent: str) -> str:
    lines = text.splitlines()
    if not lines:
        return indent
    base = indent_of(lines[0]) if lines[0].strip() else ""
    out = []
    for i, ln in enumerate(lines):
        if i == 0:
            out.append(indent + ln.lstrip())
        else:
            stripped = ln[len(base) :] if ln.startswith(base) else ln.lstrip()
            out.append(indent + stripped if ln.strip() else "")
    return "\n".join(out)


def parses(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False
