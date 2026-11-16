#!/bin/bash
cd "$(dirname "$0")"
if [ ! -x "${HANDWRITER_VENV:-$HOME/.venvs/handwriter}/bin/python" ]; then
  bash ./setup.sh || { echo "Setup failed. Press Return to close."; read -r; exit 1; }
fi
bash ./start.sh
