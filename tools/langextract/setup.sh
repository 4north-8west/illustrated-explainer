#!/bin/bash
# One-time setup of the langextract helper venv. Idempotent.
set -e
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --disable-pip-version-check -r requirements.txt
echo "[langextract] venv ready at $(pwd)/.venv"
