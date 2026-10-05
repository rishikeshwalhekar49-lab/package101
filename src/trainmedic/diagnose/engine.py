"""Evidence fusion: turn static findings, invariant violations, probe failures,
compile diffs and config problems into ranked root causes with an attribution
(user code vs framework configuration vs compiler vs distributed vs numerics)."""

from __future__ import annotations

from collections import defaultdict

from ..types import Attribution, Diagnosis, Evidence, RootCause, Severity
from . import taxonomy

SOURCE_ATTRIBUTION = {
    "compile": Attribution.COMPILER.value,
    "config": Attribution.FRAMEWORK_CONFIG.value,
    "distributed": Attribution.DISTRIBUTED.value,
}


def _combine(confs: list[float]) -> float:
    p = 1.0
    for c in confs:
        p *= 1.0 - max(0.0, min(0.999, c))
    return 1.0 - p


def fuse(evidence: list[Evidence], *, min_confidence: float = 0.0) -> list[RootCause]:
    """Group evidence by root cause; corroboration across sources raises confidence."""
    groups: dict[str, list[Evidence]] = defaultdict(list)
    for ev in evidence:
        if ev.confidence < min_confidence or ev.severity == Severity.INFO and ev.confidence < 0.5:
            continue
        groups[ev.cause or "unknown"].append(ev)
    # merge related causes when one of them is backed by a static finding with a location
    causes = list(groups)
    for a in causes:
        for b in causes:
            if a != b and a in groups and b in groups and taxonomy.related(a, b):
                a_static = any(e.source == "static" for e in groups[a])
                b_static = any(e.source == "static" for e in groups[b])
                if a_static and not b_static:
                    groups[a].extend(groups.pop(b))
    out: list[RootCause] = []
    for cid, evs in groups.items():
        c = taxonomy.get(cid)
        conf = _combine([e.confidence for e in evs])
        sources = {e.source for e in evs}
        if len(sources) >= 2:
            conf = min(0.99, conf + 0.1)
        loc = next((e for e in evs if e.source == "static" and e.line), None) or next((e for e in evs if e.file), None)
        msgs = []
        for e in sorted(evs, key=lambda e: -e.confidence):
            if e.message not in msgs:
                msgs.append(e.message)
        out.append(
            RootCause(
                cause_id=cid, category=c.category, title=c.title, description=" | ".join(msgs[:3]),
                confidence=round(conf, 4), evidence_ids=[e.id for e in evs],
                file=loc.file if loc else None, line=loc.line if loc else None, suggested_fix=c.fix,
            )
        )
    return sorted(out, key=lambda r: (-r.confidence, r.cause_id))


def attribute(evidence: list[Evidence], causes: list[RootCause]) -> str:
    """Attribute the dominant symptom to user code, framework config, compiler, ..."""
    by_id = {e.id: e for e in evidence}
    scores: dict[str, float] = defaultdict(float)
    for rc in causes:
        att = taxonomy.get(rc.cause_id).attribution
        for eid in rc.evidence_ids:
            ev = by_id.get(eid)
            if ev is not None and ev.source in SOURCE_ATTRIBUTION:
                att = SOURCE_ATTRIBUTION[ev.source]
        scores[att] += rc.confidence
    if not scores:
        return Attribution.UNKNOWN.value
    return max(scores.items(), key=lambda kv: kv[1])[0]


def diagnose(evidence: list[Evidence], *, threshold: float = 0.5) -> Diagnosis:
    causes = fuse(evidence)
    actionable = [c for c in causes if c.confidence >= threshold and c.cause_id not in ("unseeded",)]
    if not actionable:
        return Diagnosis(
            bug_found=False, causes=causes, attribution=attribute(evidence, causes) if causes else Attribution.UNKNOWN.value,
            summary="No evidence above the confidence threshold; no bug claimed." if not causes else
            f"Only weak signals ({', '.join(c.cause_id for c in causes[:3])}); no bug claimed.",
            source="rules",
        )
    top = actionable[0]
    where = f" at {top.file}:{top.line}" if top.file and top.line else ""
    summary = f"Most likely root cause: {top.title}{where} (confidence {top.confidence:.2f})."
    if len(actionable) > 1:
        summary += " Also: " + "; ".join(f"{c.title} ({c.confidence:.2f})" for c in actionable[1:4]) + "."
    return Diagnosis(True, causes, summary, attribute(evidence, actionable), "rules")
