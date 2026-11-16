#!/bin/bash
cd "$(dirname "$0")/.."
VENV="${HANDWRITER_VENV:-$HOME/.venvs/handwriter}"
PY="$VENV/bin/python"
PORT="${HANDWRITER_PORT:-8000}"
if [ ! -x "$PY" ]; then
  echo "Run setup first:  bash mac/setup.sh"
  exit 1
fi
URL="http://localhost:$PORT"
echo "Handwriter is starting at $URL  (close this window or press Ctrl+C to stop it)"
(
  for _ in $(seq 1 60); do
    if curl -s -o /dev/null "$URL/api/status"; then
      if command -v open >/dev/null 2>&1; then open "$URL"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"; fi
      break
    fi
    sleep 1
  done
) &
exec "$PY" -m uvicorn hw.server:app --app-dir ml --host 127.0.0.1 --port "$PORT"
