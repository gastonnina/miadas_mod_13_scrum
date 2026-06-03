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

3. Construcción de master tables por split
- Se realiza en los notebooks existentes (no en script dedicado en esta etapa):
  - `notebooks/sprint_01_eda/01_build_master_table.ipynb`
  - Usar `PROFILE_SOURCE=dev` para construir sobre el split `dev`.

4. Feature engineering y entrenamiento (flujo general)
- `build_features.sh`
  - Ejecuta proceso de generación de features del proyecto.
- `train_model.sh`
  - Ejecuta entrenamiento de modelos según el flujo definido.

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
