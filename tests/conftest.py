from __future__ import annotations

from pathlib import Path

import pytest

PIPELINES = Path(__file__).resolve().parents[1] / "src" / "trainmedic" / "bench" / "pipelines"


@pytest.fixture
def pipelines_dir() -> Path:
    return PIPELINES


@pytest.fixture
def write_script(tmp_path):
    def _write(source: str, name: str = "train.py") -> Path:
        p = tmp_path / name
        p.write_text(source)
        return p

    return _write
