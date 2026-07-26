#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==================================================="
echo "Geekatplay Studio - MusicMapper Dependency Installer"
echo "Created by Vladimir Chopine"
echo "==================================================="
echo

if [ -x "$SCRIPT_DIR/../../../python_embeded/python.exe" ]; then
  PYTHON_CMD="$SCRIPT_DIR/../../../python_embeded/python.exe"
elif [ -x "$SCRIPT_DIR/../../../python_embedded/python.exe" ]; then
  PYTHON_CMD="$SCRIPT_DIR/../../../python_embedded/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "[ERROR] Could not find a usable Python interpreter." >&2
  echo "Please run: pip install -r requirements.txt manually in your ComfyUI python environment." >&2
  exit 1
fi

echo "Using Python: $PYTHON_CMD"
echo "Installing audio dependencies from requirements.txt..."
echo
"$PYTHON_CMD" -m pip install -r requirements.txt

echo
echo "---------------------------------------------------"
if command -v ollama >/dev/null 2>&1; then
  echo "[OK] Ollama is detected on your system."
else
  echo "[TIP] Ollama is not installed or not in PATH."
  echo "If you want local LLM music prompt generation, download Ollama from:"
  echo "https://ollama.com/download/mac"
  echo "Then run: ollama pull llama3"
  echo "(MusicMapper will also work offline using its built-in rules engine!)"
fi
echo "---------------------------------------------------"

echo
echo "==================================================="
echo "Installation complete successfully!"
echo "Please restart ComfyUI to load the Geekatplay Studio nodes."
echo "==================================================="
