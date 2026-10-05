"""Diagnosis: fuse evidence into ranked root causes and attribute them
(user code / framework config / compiler / distributed / numerics)."""

from .engine import attribute, diagnose, fuse

__all__ = ["attribute", "diagnose", "fuse"]
