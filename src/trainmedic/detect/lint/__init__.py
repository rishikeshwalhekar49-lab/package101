"""Static analysis of training scripts (no execution required)."""

from .engine import RULES, Rule, all_rules, fix_finding, iter_python_files, lint_file, lint_paths, lint_source

__all__ = ["RULES", "Rule", "all_rules", "fix_finding", "iter_python_files", "lint_file", "lint_paths", "lint_source"]
