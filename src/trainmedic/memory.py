"""Experience memory: remember which fixes worked for which evidence signature.

Every repair attempt is recorded with its outcome -- ``accepted`` (validated fix),
``rejected`` (did not validate), ``false_alarm`` (proposed on code later found
correct) or ``damage`` (made a correct pipeline worse). Suggestions are only
re-offered when their track record is good, and fixes that ever damaged correct
code need a higher bar -- the guard against "over-applying remembered fixes".
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .repair.patch import Patch, PatchConflict, search_replace
from .types import Evidence

OUTCOMES = ("accepted", "rejected", "false_alarm", "damage")


def default_path() -> Path:
    env = os.environ.get("TRAINMEDIC_MEMORY")
    if env:
        return Path(env)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "trainmedic" / "memory.jsonl"


def signature(evidence: list[Evidence], min_confidence: float = 0.5) -> list[str]:
    return sorted({e.rule for e in evidence if e.confidence >= min_confidence and e.rule not in ("TM030",)})


def _pairs(patch: Patch, sources: dict[str, str]) -> list[list[str]]:
    out = []
    for e in patch.edits:
        lines = sources.get(e.file, "").splitlines(keepends=True)
        old, new = "".join(e.old), "".join(e.new)
        if not old:  # insertion: anchor on the previous line
            anchor = lines[e.start - 1] if e.start > 0 and e.start - 1 < len(lines) else ""
            old, new = anchor, anchor + new
        out.append([old, new])
    return out


def _key(pairs: list[list[str]]) -> str:
    norm = json.dumps([[" ".join(a.split()), " ".join(b.split())] for a, b in pairs])
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


@dataclass
class MemoryRecord:
    signature: list[str]
    causes: list[str]
    pairs: list[list[str]]
    outcome: str
    origin: str = ""
    description: str = ""
    script: str = ""
    timestamp: str = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))

    @property
    def key(self) -> str:
        return _key(self.pairs)


class ExperienceMemory:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_path()

    # ---------------------------------------------------------------- io
    def records(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(MemoryRecord(**json.loads(line)))
                except (TypeError, json.JSONDecodeError):
                    continue
        return out

    def record(self, evidence: list[Evidence], causes: list[str], patch: Patch, sources: dict[str, str], outcome: str,
               script: str = "") -> MemoryRecord:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        rec = MemoryRecord(signature(evidence), sorted(set(causes)), _pairs(patch, sources), outcome, patch.origin,
                           patch.description[:300], script)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec)) + "\n")
        return rec

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    # ------------------------------------------------------------ queries
    def fix_stats(self) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for r in self.records():
            s = stats.setdefault(r.key, {"pairs": r.pairs, "signatures": [], "description": r.description,
                                         **{o: 0 for o in OUTCOMES}})
            s[r.outcome] += 1
            if r.signature not in s["signatures"]:
                s["signatures"].append(r.signature)
        for s in stats.values():
            tried = sum(s[o] for o in OUTCOMES)
            s["support"] = tried
            s["precision"] = s["accepted"] / tried if tried else 0.0
        return stats

    def suggest(self, evidence: list[Evidence], sources: dict[str, str], *, min_precision: float = 0.6,
                min_support: int = 2, min_similarity: float = 0.6) -> list[Patch]:
        sig = set(signature(evidence))
        if not sig:
            return []
        out: list[Patch] = []
        for key, s in self.fix_stats().items():
            sim = max((len(sig & set(x)) / len(sig | set(x)) for x in s["signatures"] if sig | set(x)), default=0.0)
            bar = max(min_precision, 0.8) if s["damage"] or s["false_alarm"] else min_precision
            if sim < min_similarity or s["support"] < min_support or s["precision"] < bar:
                continue
            for f, text in sources.items():
                try:
                    new = text
                    for a, b in s["pairs"]:
                        new = search_replace(new, a, b)
                except PatchConflict:
                    continue
                if new != text:
                    out.append(Patch.from_sources(f, text, new, origin=f"memory:{key}", description=s["description"],
                                                  evidence_ids=[e.id for e in evidence if e.rule in sig]))
                    break
        return out

    def hints(self, evidence: list[Evidence], limit: int = 3) -> list[str]:
        sig = set(signature(evidence))
        out = []
        for key, s in sorted(self.fix_stats().items(), key=lambda kv: -kv[1]["precision"]):
            if any(sig & set(x) for x in s["signatures"]) and s["accepted"]:
                ex = s["pairs"][0]
                out.append(f"- ({s['accepted']}/{s['support']} accepted) {s['description'] or key}: "
                           f"replace `{ex[0].strip()[:120]}` with `{ex[1].strip()[:160]}`")
            if len(out) >= limit:
                break
        return out
