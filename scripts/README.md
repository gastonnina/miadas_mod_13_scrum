# Scripts del Proyecto

Este directorio contiene scripts de soporte para ingesta, EDA, splits temporales y ejecución operativa.

## Orden recomendado por etapas

1. Preparación de datos base
- `csv_to_parquet.py`
  - Convierte fuentes CSV a Parquet.
  - Ejecutar cuando se incorporan datos crudos nuevos en formato CSV.

2. Split temporal para validación final (Sprint actual)
- `create_temporal_split.py`
  - Crea particiones `dev` y `holdout_3m` por rango temporal desde `data/raw`.
  - Salida en `data/splits/temporal_2018q4/`.
  - Genera también `holdout_3m_ids.parquet` y `holdout_3m_metadata.json`.

3. Pipeline Sprint 2 completo (dev + holdout) — un solo comando
- `build_features.sh`
  - Orquesta los 6 pasos del Sprint 2 en orden: master table dev → features dev → selección experimental → evaluación vs baseline → master table holdout → features holdout.
  - Usa el entorno conda `ai-miadas` por defecto (sobreescribible con `CONDA_ENV=otro`).
  - Requiere haber completado los pasos 1 y 2 primero.

  ```bash
  bash scripts/build_features.sh
  ```

  Pasos individuales equivalentes:

  ```bash
  CONDA="conda run -n ai-miadas python"

  # [1/4] Master table DEV — calcula y persiste umbral P80 (fit)
  $CONDA src/data/build_master_table.py --profile-source dev --threshold-mode auto

  # [2/4] Features RFM DEV — genera 18 variables nuevas sobre master table dev
  $CONDA src/features/build_rfm_features.py

  # [3/4] Selección experimental — 8 experimentos, elige corr_le_0.85 (28 features)
  $CONDA src/features/feature_selection_experiments.py

  # [4/4] Evaluación vs baseline — métricas sobre split temporal 2018-07-01
  $CONDA src/models/evaluate_model.py

  # [5/6] Master table HOLDOUT — aplica umbral fijo (apply), no recalcula
  $CONDA src/data/build_master_table.py --profile-source holdout --threshold-mode apply

  # [6/6] Features RFM HOLDOUT — simulación mensual ago-oct 2018
  $CONDA src/features/build_rfm_features.py \
      --input-path data/processed/03_master_table_clean_holdout.parquet \
      --output-path data/processed/holdout_features_rfm.parquet \
      --metadata-path data/processed/holdout_features_rfm_metadata.json
  ```

  Artefactos generados:

  | Artefacto | Descripción |
  |---|---|
  | `data/processed/03_master_table_clean_dev.parquet` | Master table limpia (dev) |
  | `data/processed/05_features_rfm.parquet` | Features RFM + 18 derivadas (dev) |
  | `data/processed/06_features_selected.parquet` | 28 features seleccionadas |
  | `data/processed/08_evaluation_metrics.json` | Métricas finales vs baseline |
  | `data/processed/03_master_table_clean_holdout.parquet` | Master table limpia (holdout) |
  | `data/processed/holdout_features_rfm.parquet` | Features RFM holdout |
  | `data/processed/premium_threshold_dev.json` | Umbral P80 = 197.01 persistido |

4. Entrenamiento de modelos (Sprint 3)
- `train_model.sh`
  - Ejecuta entrenamiento de modelos sobre las features seleccionadas del Sprint 2.

5. Ejecución de análisis y aplicaciones
- `run_eda.sh`
  - Atajo para correr notebooks/flujo EDA del repositorio.
- `run_app.sh`
  - Levanta servicios de app/API/dashboard según configuración local.

## Notas operativas

- `data/raw` se mantiene como fuente base sin sobreescribir.
- Para flujo temporal, consumir salidas de `data/splits/temporal_2018q4/`.
- Notebooks adaptados para `raw/dev`:
  - `01_build_master_table.ipynb` y `02_eda_premium_customers.ipynb` quedan con `PROFILE_SOURCE=dev` por defecto.
  - si se necesita volver a `raw`, ejecutar explícitamente con `PROFILE_SOURCE=raw`.
