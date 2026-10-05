"""trainmedic -- detect, diagnose, repair and statistically validate silent bugs
in PyTorch training code.

Pipeline: static lint + runtime invariants (detect) -> rule/LLM root-cause
analysis (diagnose) -> template/LLM patches with abstention (repair) ->
compute-budgeted multi-seed acceptance and patch minimisation (validate).

Training scripts only need the light-weight helpers re-exported here
(``seed_everything``, ``scaled``, ``report``); everything else is imported lazily.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .reporting import budget, report, reported, scaled, seed, seed_everything

__version__ = "0.1.0"

_LAZY = {
    "watch": ("trainmedic.api", "watch"),
    "check_script": ("trainmedic.api", "check_script"),
    "Tracer": ("trainmedic.detect.tracer", "Tracer"),
    "Trace": ("trainmedic.detect.trace", "Trace"),
    "InvariantSet": ("trainmedic.detect.invariants", "InvariantSet"),
    "lint_source": ("trainmedic.detect.lint", "lint_source"),
    "lint_file": ("trainmedic.detect.lint", "lint_file"),
    "lint_paths": ("trainmedic.detect.lint", "lint_paths"),
    "probes": ("trainmedic.detect.probes", None),
    "compile_diff": ("trainmedic.detect.compile_diff", "compile_diff"),
    "check_deepspeed_config": ("trainmedic.detect.config_check", "check_deepspeed_config"),
    "diagnose": ("trainmedic.diagnose", "diagnose"),
    "Patch": ("trainmedic.repair.patch", "Patch"),
    "RepairSession": ("trainmedic.repair.agent", "RepairSession"),
    "RepairConfig": ("trainmedic.repair.agent", "RepairConfig"),
    "ddmin": ("trainmedic.repair.minimize", "ddmin"),
    "Runner": ("trainmedic.validate.runner", "Runner"),
    "RunSpec": ("trainmedic.validate.runner", "RunSpec"),
    "SequentialValidator": ("trainmedic.validate.halving", "SequentialValidator"),
    "Guardian": ("trainmedic.remediate.guardian", "Guardian"),
    "SimulatedDataParallel": ("trainmedic.remediate.simddp", "SimulatedDataParallel"),
    "ExperienceMemory": ("trainmedic.memory", "ExperienceMemory"),
}

__all__ = ["__version__", "budget", "report", "reported", "scaled", "seed", "seed_everything", *_LAZY]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod, attr = _LAZY[name]
        m = import_module(mod)
        return m if attr is None else getattr(m, attr)
    raise AttributeError(f"module 'trainmedic' has no attribute {name!r}")
