#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

is_supported_python() {
  local candidate="$1"
  "$candidate" - <<'PY' >/dev/null 2>&1
import sys
major, minor = sys.version_info[:2]
raise SystemExit(0 if (major, minor) >= (3, 11) and (major, minor) < (3, 14) else 1)
PY
}

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
else
  python_bin=""
  candidates=(
    "$repo_root/.venv_releasecheck313/bin/python"
    "$repo_root/.venv/bin/python"
    python3.13
    python3.12
    python3.11
    python3
    python
  )
  for candidate in "${candidates[@]}"; do
    if [[ "$candidate" == */* ]]; then
      [[ -x "$candidate" ]] || continue
    else
      command -v "$candidate" >/dev/null 2>&1 || continue
    fi
    if is_supported_python "$candidate"; then
      python_bin="$candidate"
      break
    fi
  done
fi

if [[ -z "${python_bin:-}" ]]; then
  echo "error: no supported Python interpreter found (requires >=3.11,<3.14)." >&2
  echo "set PYTHON_BIN to a Python 3.11-3.13 executable." >&2
  exit 1
fi

"$python_bin" - <<'PY' >/dev/null
import importlib.util
missing = [name for name in ("build", "twine") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "error: missing release dependencies for selected Python: "
        + ", ".join(missing)
        + ". Install with: python -m pip install build twine"
    )
PY

version="$("$python_bin" - <<'PY'
from pathlib import Path
import tomllib

pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
print(pyproject["project"]["version"])
PY
)"

outdir="dist/release-${version}"
mkdir -p "$outdir"

"$python_bin" -m build --outdir "$outdir"
"$python_bin" -m twine check "$outdir"/*

echo "Release artifacts are ready:"
echo "  $outdir"
echo "Upload with:"
echo "  $python_bin -m twine upload \"$outdir\"/*"
