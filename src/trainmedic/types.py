"""Core data types shared by every stage of the pipeline.

Everything that suggests "something is wrong" (a static lint finding, a violated
training invariant, a numeric anomaly, a failed probe, a config problem) is an
:class:`Evidence`. Diagnosis turns evidence into :class:`RootCause` objects, and
repair turns root causes into patches (see :mod:`trainmedic.repair.patch`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def rank(self) -> int:
        return {"info": 0, "warning": 1, "error": 2}[self.value]


class Source(str, Enum):
    STATIC = "static"
    INVARIANT = "invariant"
    NUMERIC = "numeric"
    PROBE = "probe"
    COMPILE = "compile"
    CONFIG = "config"
    DISTRIBUTED = "distributed"
    RUNTIME_ERROR = "runtime_error"
    LLM = "llm"


@dataclass
class Evidence:
    """One observation suggesting a defect.

    ``confidence`` is the probability-like belief (0..1) that this observation
    indicates a real bug *in a correct-by-default pipeline*. Abstention and
    diagnosis both rely on it, so rules are calibrated conservatively.
    """

    id: str
    source: str
    rule: str
    message: str
    severity: Severity = Severity.WARNING
    confidence: float = 0.5
    file: str | None = None
    line: int | None = None
    end_line: int | None = None
    step: int | None = None
    category: str | None = None
    cause: str | None = None
    fixable: bool = False
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["severity"] = self.severity.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Evidence":
        d = dict(d)
        d["severity"] = Severity(d.get("severity", "warning"))
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def location(self) -> str:
        if self.file and self.line:
            return f"{self.file}:{self.line}"
        if self.file:
            return self.file
        if self.step is not None:
            return f"step {self.step}"
        return "-"

    def short(self) -> str:
        return f"[{self.id}] {self.severity.value.upper()} {self.rule} ({self.confidence:.2f}) {self.location()}: {self.message}"


@dataclass
class RootCause:
    cause_id: str
    category: str
    title: str
    description: str
    confidence: float
    evidence_ids: list[str] = field(default_factory=list)
    file: str | None = None
    line: int | None = None
    suggested_fix: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RootCause":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class Attribution(str, Enum):
    USER_CODE = "user_code"
    FRAMEWORK_CONFIG = "framework_config"
    COMPILER = "compiler"
    DISTRIBUTED = "distributed"
    HARDWARE_OR_NUMERICS = "hardware_or_numerics"
    UNKNOWN = "unknown"


@dataclass
class Diagnosis:
    bug_found: bool
    causes: list[RootCause] = field(default_factory=list)
    summary: str = ""
    attribution: str = Attribution.UNKNOWN.value
    source: str = "rules"

    @property
    def top(self) -> RootCause | None:
        return self.causes[0] if self.causes else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bug_found": self.bug_found,
            "causes": [c.to_dict() for c in self.causes],
            "summary": self.summary,
            "attribution": self.attribution,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Diagnosis":
        return cls(
            bug_found=bool(d.get("bug_found")),
            causes=[RootCause.from_dict(c) for c in d.get("causes", [])],
            summary=d.get("summary", ""),
            attribution=d.get("attribution", Attribution.UNKNOWN.value),
            source=d.get("source", "rules"),
        )


class EvidenceIds:
    """Hands out stable, short evidence ids per source (S1, V1, N1, ...)."""

    PREFIX = {
        "static": "S",
        "invariant": "V",
        "numeric": "N",
        "probe": "P",
        "compile": "C",
        "config": "K",
        "distributed": "D",
        "runtime_error": "E",
        "llm": "L",
    }

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def next(self, source: str) -> str:
        prefix = self.PREFIX.get(source, "X")
        self._counts[prefix] = self._counts.get(prefix, 0) + 1
        return f"{prefix}{self._counts[prefix]}"

    def renumber(self, evidence: list[Evidence]) -> list[Evidence]:
        for ev in evidence:
            ev.id = self.next(ev.source)
        return evidence
