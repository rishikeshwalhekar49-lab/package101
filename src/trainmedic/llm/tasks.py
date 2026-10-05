"""LLM tasks: evidence-grounded diagnosis, evidence-grounded repair with an
explicit abstain option, and a deliberately naive repair baseline (for
measuring damage on bug-free code)."""

from __future__ import annotations

from typing import Any

from ..diagnose import taxonomy
from ..repair.patch import Patch, PatchConflict
from ..types import Diagnosis, Evidence, RootCause
from .backend import LLMBackend

MAX_PROMPT_CHARS = 600_000

SYSTEM = f"""You are trainmedic, an expert at debugging deep-learning training code (PyTorch first; also JAX/Keras).
You receive the training code, evidence collected by static analysis and by runtime training invariants, and a
trace summary. Your job is to find *silent* bugs: code that runs without errors but trains or evaluates wrongly.

Principles:
- Ground every claim in the evidence. Cite evidence ids (e.g. "S1", "V2"). A claim with no supporting evidence is
  speculation; mark low confidence or leave it out.
- It is normal for code to be correct. If the evidence does not show a real defect, say so: answer bug_found=false
  (diagnosis) or abstain=true (repair). Proposing a change to correct code is harmful.
- Repairs must be minimal: change only the lines needed to fix the cited defect. Do not refactor, rename, reformat,
  add features, change hyperparameters, seeds, regularisation, data splits or metrics unless the evidence implicates
  exactly that.
- Edits are search/replace blocks. `search` must be copied verbatim from the file (enough lines to be unique);
  `replace` is the new text for exactly that block, with the same indentation.

Root-cause catalogue (use these ids when one fits; otherwise use a short snake_case id):
{taxonomy.catalogue_text()}
"""

DIAGNOSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["bug_found", "summary", "attribution", "causes"],
    "properties": {
        "bug_found": {"type": "boolean"},
        "summary": {"type": "string"},
        "attribution": {
            "type": "string",
            "enum": ["user_code", "framework_config", "compiler", "distributed", "hardware_or_numerics", "unknown"],
        },
        "causes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cause_id", "title", "description", "confidence", "evidence_ids", "file", "line"],
                "properties": {
                    "cause_id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                },
            },
        },
    },
}

_EDITS = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["file", "search", "replace"],
        "properties": {"file": {"type": "string"}, "search": {"type": "string"}, "replace": {"type": "string"}},
    },
}

REPAIR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["abstain", "reason", "candidates"],
    "properties": {
        "abstain": {"type": "boolean"},
        "reason": {"type": "string"},
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["description", "evidence_ids", "edits"],
                "properties": {
                    "description": {"type": "string"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "edits": _EDITS,
                },
            },
        },
    },
}

NAIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["explanation", "edits"],
    "properties": {"explanation": {"type": "string"}, "edits": _EDITS},
}


# ------------------------------------------------------------------ prompts
def code_block(sources: dict[str, str]) -> str:
    parts = []
    for f, text in sources.items():
        numbered = "\n".join(f"{i:4d} | {line}" for i, line in enumerate(text.splitlines(), 1))
        parts.append(f"### File: {f}\n```python\n{numbered}\n```")
    out = "\n\n".join(parts)
    if len(out) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"the code to analyse is {len(out)} characters, above the {MAX_PROMPT_CHARS} limit; pass fewer files"
        )
    return out


def evidence_block(evidence: list[Evidence]) -> str:
    if not evidence:
        return "(no evidence: static analysis and runtime invariants found nothing)"
    lines = []
    for e in evidence:
        lines.append(f"- {e.id} [{e.source}/{e.rule}] conf={e.confidence:.2f} cause={e.cause} at {e.location()}: {e.message}")
    return "\n".join(lines)


def _diagnosis_prompt(sources, evidence, trace_summary, rules_diag) -> str:
    s = [
        "Diagnose silent training bugs in this code.",
        "",
        code_block(sources),
        "",
        "## Evidence",
        evidence_block(evidence),
    ]
    if trace_summary:
        s += ["", "## Trace summary", str(trace_summary)]
    if rules_diag is not None:
        s += ["", "## Rule-based pre-diagnosis", rules_diag.summary]
    s += ["", "Return bug_found=false with an empty causes list if the evidence does not establish a defect."]
    return "\n".join(s)


def _repair_prompt(sources, evidence, diagnosis, max_candidates, hints) -> str:
    s = [
        f"Propose up to {max_candidates} alternative minimal patches for the diagnosed defect(s), or abstain.",
        "",
        code_block(sources),
        "",
        "## Evidence",
        evidence_block(evidence),
        "",
        "## Diagnosis",
        diagnosis.summary if diagnosis else "(none)",
    ]
    if diagnosis:
        for c in diagnosis.causes[:5]:
            s.append(f"- {c.cause_id} ({c.confidence:.2f}) {c.title}: {c.description} [evidence {', '.join(c.evidence_ids)}]")
    if hints:
        s += ["", "## Fixes that worked before for similar evidence (use only if they apply)", *hints]
    s += [
        "",
        "Each candidate must cite the evidence ids it addresses. Prefer the smallest change. If the code is correct, "
        "set abstain=true and return no candidates.",
    ]
    return "\n".join(s)


# --------------------------------------------------------------------- tasks
def llm_diagnose(
    backend: LLMBackend,
    sources: dict[str, str],
    evidence: list[Evidence],
    *,
    trace_summary: Any = None,
    rules_diagnosis: Diagnosis | None = None,
) -> Diagnosis:
    data = backend.complete_json(SYSTEM, _diagnosis_prompt(sources, evidence, trace_summary, rules_diagnosis), DIAGNOSIS_SCHEMA)
    causes = []
    for c in data.get("causes", []):
        tc = taxonomy.get(c.get("cause_id"))
        causes.append(
            RootCause(
                cause_id=c.get("cause_id") or "unknown",
                category=tc.category,
                title=c.get("title") or tc.title,
                description=c.get("description", ""),
                confidence=float(max(0.0, min(1.0, c.get("confidence", 0.5)))),
                evidence_ids=list(c.get("evidence_ids", [])),
                file=c.get("file") or None,
                line=c.get("line") or None,
                suggested_fix=tc.fix,
            )
        )
    return Diagnosis(bool(data.get("bug_found")), causes, data.get("summary", ""), data.get("attribution", "unknown"), "llm")


def merge_diagnoses(rules: Diagnosis, llm: Diagnosis, evidence: list[Evidence], *, allow_ungrounded: bool = False,
                    threshold: float = 0.5) -> Diagnosis:
    known = {e.id for e in evidence}
    merged: dict[str, RootCause] = {c.cause_id: RootCause(**c.to_dict()) for c in rules.causes}
    for c in llm.causes:
        grounded = bool(set(c.evidence_ids) & known)
        conf = c.confidence if (grounded or allow_ungrounded) else c.confidence * 0.4
        c.evidence_ids = [e for e in c.evidence_ids if e in known]
        if c.cause_id in merged:
            m = merged[c.cause_id]
            m.confidence = round(1 - (1 - m.confidence) * (1 - conf * 0.5), 4)
            m.evidence_ids = list(dict.fromkeys(m.evidence_ids + c.evidence_ids))
            if c.description and c.description not in m.description:
                m.description = f"{m.description} | LLM: {c.description}" if m.description else c.description
            m.file = m.file or c.file
            m.line = m.line or c.line
        else:
            c.confidence = round(conf, 4)
            if not grounded:
                c.title = f"[ungrounded] {c.title}"
            merged[c.cause_id] = c
    causes = sorted(merged.values(), key=lambda c: -c.confidence)
    bug = any(c.confidence >= threshold for c in causes) and (rules.bug_found or llm.bug_found)
    summary = llm.summary if llm.bug_found else rules.summary
    attribution = llm.attribution if llm.attribution != "unknown" else rules.attribution
    return Diagnosis(bug, causes, summary, attribution, "hybrid")


def llm_repair(
    backend: LLMBackend,
    sources: dict[str, str],
    evidence: list[Evidence],
    diagnosis: Diagnosis | None,
    *,
    max_candidates: int = 3,
    hints: list[str] | None = None,
) -> tuple[bool, str, list[Patch], list[str]]:
    """Returns (abstained, reason, patches, errors)."""
    data = backend.complete_json(SYSTEM, _repair_prompt(sources, evidence, diagnosis, max_candidates, hints), REPAIR_SCHEMA)
    patches, errors = [], []
    for i, cand in enumerate(data.get("candidates", [])[:max_candidates]):
        try:
            patches.append(_edits_to_patch(sources, cand.get("edits", []), origin=f"llm:{backend.name}#{i}",
                                           description=cand.get("description", ""), evidence_ids=list(cand.get("evidence_ids", []))))
        except PatchConflict as exc:
            errors.append(f"candidate {i}: {exc}")
    return bool(data.get("abstain")), data.get("reason", ""), patches, errors


def naive_llm_repair(backend: LLMBackend, sources: dict[str, str]) -> tuple[list[Patch], list[str]]:
    """Baseline without evidence or an abstain option ("this code has a bug, fix it")."""
    prompt = "This training code has a bug. Find it and fix it.\n\n" + code_block(sources)
    data = backend.complete_json("You are an expert machine-learning engineer. Return search/replace edits.", prompt, NAIVE_SCHEMA)
    try:
        return [_edits_to_patch(sources, data.get("edits", []), origin=f"naive-llm:{backend.name}",
                                description=data.get("explanation", ""), evidence_ids=[])], []
    except PatchConflict as exc:
        return [], [str(exc)]


def _edits_to_patch(sources: dict[str, str], edits: list[dict[str, str]], **kw: Any) -> Patch:
    from ..repair.patch import search_replace

    new = dict(sources)
    for e in edits:
        f = _resolve(e.get("file", ""), new)
        new[f] = search_replace(new[f], e.get("search", ""), e.get("replace", ""))
    return Patch.from_file_pairs({f: (sources[f], new[f]) for f in sources if new[f] != sources[f]}, **kw)


def _resolve(name: str, sources: dict[str, str]) -> str:
    if name in sources:
        return name
    for f in sources:
        if f.endswith(name) or name.endswith(f) or f.split("/")[-1] == name.split("/")[-1]:
            return f
    if len(sources) == 1:
        return next(iter(sources))
    raise PatchConflict(f"unknown file {name!r}")
