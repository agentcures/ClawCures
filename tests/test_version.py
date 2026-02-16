from __future__ import annotations

from pathlib import Path
import tomllib

import refua_campaign


def test_package_version_matches_pyproject() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    assert refua_campaign.__version__ == pyproject["project"]["version"]
