#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-dashboard}"

cd "$ROOT"

case "$MODE" in
  dashboard)
    echo "Levantando dashboard Streamlit..."
    docker compose up dashboard --build
    ;;
  api)
    echo "Levantando API minima..."
    docker compose up api --build
    ;;
  all)
    echo "Levantando dashboard + API..."
    docker compose up dashboard api --build
    ;;
  *)
    echo "Uso: bash scripts/run_app.sh [dashboard|api|all]" >&2
    exit 1
    ;;
esac
