#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Auto-detect Python runtime:
#   1. CONDA_ENV variable set  → conda run -n <env>
#   2. .venv exists at project root → use it directly (no activation needed)
#   3. fallback → python in current PATH (activated venv, system python, etc.)
if [ -n "${CONDA_ENV:-}" ]; then
    run() { conda run -n "$CONDA_ENV" python "$@"; }
    echo "Entorno: conda ($CONDA_ENV)"
elif [ -f "$ROOT/.venv/bin/python" ]; then
    run() { "$ROOT/.venv/bin/python" "$@"; }
    echo "Entorno: .venv"
else
    run() { python "$@"; }
    echo "Entorno: python en PATH ($(which python))"
fi

echo "=== Sprint 2 Pipeline ==="
echo "Root: $ROOT"
cd "$ROOT"

echo ""
echo "--- [1/4] Master table DEV (fit threshold) ---"
run src/data/build_master_table.py --profile-source dev --threshold-mode auto

echo ""
echo "--- [2/4] Features RFM DEV ---"
run src/features/build_rfm_features.py

echo ""
echo "--- [3/4] Feature selection experiments DEV ---"
run src/features/feature_selection_experiments.py

echo ""
echo "--- [4/4] Evaluacion vs baseline DEV ---"
run src/models/evaluate_model.py

echo ""
echo "=== Simulacion mensual: holdout (apply threshold) ==="

echo ""
echo "--- [5/5] Master table HOLDOUT ---"
run src/data/build_master_table.py \
    --profile-source holdout \
    --threshold-mode apply

echo ""
echo "--- [6/6] Features RFM HOLDOUT ---"
run src/features/build_rfm_features.py \
    --input-path data/processed/03_master_table_clean_holdout.parquet \
    --output-path data/processed/holdout_features_rfm.parquet \
    --metadata-path data/processed/holdout_features_rfm_metadata.json

echo ""
echo "=== Pipeline completo OK ==="
echo "Artefactos DEV:"
echo "  data/processed/03_master_table_clean.parquet"
echo "  data/processed/05_features_rfm.parquet"
echo "  data/processed/06_features_selected.parquet"
echo "  data/processed/08_evaluation_metrics.json"
echo "Artefactos HOLDOUT:"
echo "  data/processed/03_master_table_clean_holdout.parquet"
echo "  data/processed/holdout_features_rfm.parquet"
