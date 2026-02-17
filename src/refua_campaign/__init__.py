"""ClawCures package."""

from importlib.metadata import version as _distribution_version
from pathlib import Path
import tomllib

__all__ = ["__version__"]


def _read_version_from_pyproject() -> str | None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    version = project.get("version")
    if not version:
        return None
    return str(version)


def _resolve_version() -> str:
    local_version = _read_version_from_pyproject()
    if local_version is not None:
        return local_version
    return _distribution_version("ClawCures")


__version__ = _resolve_version()
