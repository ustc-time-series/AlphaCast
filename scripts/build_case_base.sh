#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Override explicitly, or use the active environment / project venv.
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="$PYTHON_BIN"
elif [[ -x "$ROOT/.venv-foundation/bin/python" ]]; then
    PYTHON="$ROOT/.venv-foundation/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
elif [[ -x "$ROOT/../AlphaCast++/.venv-foundation/bin/python" ]]; then
    PYTHON="$ROOT/../AlphaCast++/.venv-foundation/bin/python"
else
    PYTHON=python3
fi
exec "$PYTHON" -u "$ROOT/initialize_case_base.py" "$@"
