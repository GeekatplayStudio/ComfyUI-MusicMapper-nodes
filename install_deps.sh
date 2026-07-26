#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -x "$SCRIPT_DIR/../../../python_embeded/python.exe" ]; then
  PYTHON_CMD="$SCRIPT_DIR/../../../python_embeded/python.exe"
elif [ -x "$SCRIPT_DIR/../../../python_embedded/python.exe" ]; then
  PYTHON_CMD="$SCRIPT_DIR/../../../python_embedded/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "Could not find a usable Python interpreter." >&2
  echo "Please run: pip install -r requirements.txt manually in your ComfyUI python environment." >&2
  exit 1
fi

echo "==================================================="
echo "Geekatplay Studio - MusicMapper Dependency Installer"
echo "Using Python: $PYTHON_CMD"
echo "==================================================="
echo "Installing dependencies from requirements.txt..."
"$PYTHON_CMD" -m pip install -r requirements.txt

echo
echo "==================================================="
echo "Installation complete successfully!"
echo "Please restart ComfyUI."
echo "==================================================="
