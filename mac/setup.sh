#!/bin/bash
set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"
VENV="${HANDWRITER_VENV:-$HOME/.venvs/handwriter}"
PY="$VENV/bin/python"

echo "Handwriter setup in $ROOT"

UV="$(command -v uv || true)"
if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then
  UV="$HOME/.local/bin/uv"
fi
if [ -z "$UV" ]; then
  echo "Installing uv (a Python installer that needs no admin password) into your home folder..."
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
  UV="$HOME/.local/bin/uv"
fi

if [ ! -x "$PY" ]; then
  echo "Creating a Python 3.12 environment at $VENV..."
  "$UV" venv --python 3.12 "$VENV"
fi

if [ "$(uname -s)" = "Linux" ]; then
  "$UV" pip install --python "$PY" torch --index-url https://download.pytorch.org/whl/cpu
fi
echo "Installing packages (several minutes the first time)..."
"$UV" pip install --python "$PY" -r ml/requirements-mac.txt

mkdir -p data
"$PY" - <<'EOF'
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, "ml")
from hw import plotter

path = Path("data/plotter.json")
if path.exists():
    print("Kept your existing plotter settings in data/plotter.json")
else:
    pen = plotter.pen_from({**asdict(plotter.Pen()), **plotter.FALU_PENCIL})
    plotter.save_pen(pen, path)
    print("Saved the Falu pencil holder settings to data/plotter.json (pen offsets still need calibrating)")
EOF

"$PY" - <<'EOF'
import sys
sys.path.insert(0, "ml")
from hw.glyphs import GlyphWriter
from hw.layout import layout_page
w = GlyphWriter("data/glyphs.json", "data")
out = layout_page(w, "Setup check.", {}, 1)
print(f"Check passed: wrote a test line with {len(out['pages'][0])} strokes from {len(w.glyphs)} letters.")
EOF

if [ ! -f app/dist/index.html ]; then
  echo "The web app build is missing from app/dist. Pull the latest copy from GitHub."
  exit 1
fi

echo
echo "Done. Start it with:  bash mac/start.sh"
