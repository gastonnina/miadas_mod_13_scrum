# Identificacion de Clientes Premium - Olist <!-- omit in toc -->

Proyecto academico del ultimo modulo de la:
**MAESTRIA EN INTELIGENCIA ARTIFICIAL Y DATA SCIENCE PARA LA TRANSFORMACION DE NEGOCIOS**.

Modulo:
**Implementacion de Soluciones de Inteligencia Artificial Aplicada a los Negocios**.

## Tabla de contenidos <!-- omit in toc -->
- [👥 Integrantes del grupo](#-integrantes-del-grupo)
- [🎯 Tema del proyecto](#-tema-del-proyecto)
- [🧭 Objetivo general](#-objetivo-general)
- [🏗️ Arquitectura general](#️-arquitectura-general)
- [🧠 Definicion del target (actual)](#-definicion-del-target-actual)
- [🛠️ Stack tecnologico](#️-stack-tecnologico)
- [📊 Dataset](#-dataset)
- [⚙️ Configuracion del entorno](#️-configuracion-del-entorno)
- [📁 Estructura del repositorio](#-estructura-del-repositorio)
- [📦 Entregables por sprint](#-entregables-por-sprint)
- [🚀 Scripts de ejecucion](#-scripts-de-ejecucion)
  - [Prerequisitos (una sola vez)](#prerequisitos-una-sola-vez)
  - [Sprint 2 y Sprint 4 — Pipeline reproducible completo](#sprint-2-y-sprint-4--pipeline-reproducible-completo)
  - [Sprint 3 — Generacion y validacion del modelo final](#sprint-3--generacion-y-validacion-del-modelo-final)
  - [Sprint 4 — MVP en Docker](#sprint-4--mvp-en-docker)
- [🗺️ Esquema de datos](#️-esquema-de-datos)
- [🧪 Metodologia de trabajo (resumen)](#-metodologia-de-trabajo-resumen)
- [🚧 Estado del proyecto](#-estado-del-proyecto)


## 👥 Integrantes del grupo

- Gaston Nina
- Gerick Toro
- Marcelo De la Quintana

## 🎯 Tema del proyecto

**Tema 3: Identificacion de clientes premium**.

Este trabajo se desarrolla con base en el documento:
`Plan de Trabajo Último Modulo IA.pdf`
ubicado en la raiz del repositorio.

## 🧭 Objetivo general

Disenar e implementar una solucion de analitica e inteligencia artificial para identificar clientes premium en el ecosistema Olist, generando una base para acciones de negocio como fidelizacion, segmentacion y priorizacion comercial.

## 🏗️ Arquitectura general

CSV raw
-> Parquet
-> Split temporal `dev` / `backtest` / `holdout_3m`
-> Master table
-> Feature engineering (RFM y derivadas)
-> Seleccion de variables
-> Modelado supervisado
-> `modelo_final.pkl`
-> Dashboard Streamlit + API FastAPI

## 🧠 Definicion del target (actual)

La variable objetivo `is_premium` se define sobre gasto neto acumulado por cliente:

- `is_premium = 1` si `total_spent >= P80`.
- `is_premium = 0` para el resto.

Valores usados en los sprints finales:
- Umbral P80 persistido desde desarrollo: `BRL 197.01`.
- Distribucion del backtest oficial: `1.30%` premium reales (`1,253 / 96,096`).
- Justificacion: alta asimetria del gasto y necesidad de focalizar campanas sobre una minoria de clientes de alto valor.

## 🛠️ Stack tecnologico

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- PyArrow
- JupyterLab
- FastAPI
- Streamlit
- Docker

## 📊 Dataset

- Fuente: Kaggle - Brazilian E-Commerce Public Dataset by Olist
- URL: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Formatos usados en el proyecto: CSV (origen) y Parquet (trabajo analitico)

## ⚙️ Configuracion del entorno

**Opcion A — venv (equipo)**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
```

**Opcion B — conda (alternativa personal)**

```bash
conda activate ai-miadas
pip install -e .
```

## 📁 Estructura del repositorio

- `data/`: datos crudos, intermedios y procesados
- `notebooks/`: analisis por sprints
- `src/`: scripts de carga, features, modelado y evaluacion
- `models/`: artefactos de modelos baseline/final
- `reports/`: entregables documentales por sprint
- `docs/`: arquitectura, referencias y modelo de datos (DBML)
- `app/`: API (FastAPI) y dashboard (Streamlit)
- `docker/`: contenedores para API y dashboard
- `scripts/`: automatizaciones de ejecucion

## 📦 Entregables por sprint

Sprint 1 (`EDA + baseline`):
- Notebook EDA raw data: [notebooks/sprint_01_eda/00_eda_raw_data.ipynb](notebooks/sprint_01_eda/00_eda_raw_data.ipynb)
- Notebook master table: [notebooks/sprint_01_eda/01_build_master_table.ipynb](notebooks/sprint_01_eda/01_build_master_table.ipynb)
- Notebook EDA premium: [notebooks/sprint_01_eda/02_eda_premium_customers.ipynb](notebooks/sprint_01_eda/02_eda_premium_customers.ipynb)
- Reporte Sprint 1 (PDF): [reports/sprint_01/informes/sprint_01_reporte.pdf](reports/sprint_01/informes/sprint_01_reporte.pdf)
- Presentacion final Sprint 1 (PDF): [reports/sprint_01/informes/sprint_01_presentacion.pdf](reports/sprint_01/informes/sprint_01_presentacion.pdf)

Sprint 2 (`pipeline de features`):
- Estado: `Completado`
- DAG tecnico reproducible: [reports/sprint_02/dag_pipeline_sprint_2.md](reports/sprint_02/dag_pipeline_sprint_2.md)
- Documentacion del pipeline: [reports/sprint_02/documentacion_pipeline.md](reports/sprint_02/documentacion_pipeline.md)
- Experimentos de seleccion de variables: [reports/sprint_02/feature_selection_experiments.md](reports/sprint_02/feature_selection_experiments.md)
- Notebook integrador: [notebooks/sprint_02_pipeline/02_pipeline_features.ipynb](notebooks/sprint_02_pipeline/02_pipeline_features.ipynb)
- Evaluacion vs baseline: [reports/sprint_02/evaluation_vs_baseline.md](reports/sprint_02/evaluation_vs_baseline.md)
- Preguntas y respuestas de codigo: [reports/sprint_02/preguntas_respuestas_codigo_sprint_2.md](reports/sprint_02/preguntas_respuestas_codigo_sprint_2.md)
- Reporte Sprint 2 (PDF): [reports/sprint_02/informes/sprint_02_reporte.pdf](reports/sprint_02/informes/sprint_02_reporte.pdf)
- Presentacion Sprint 2 (PDF): [reports/sprint_02/informes/sprint_02_presentacion.pdf](reports/sprint_02/informes/sprint_02_presentacion.pdf)
- Presentacion editable Sprint 2 (PPTX): [reports/sprint_02/presentacion_sprint_2.pptx](reports/sprint_02/presentacion_sprint_2.pptx)

Sprint 3 (`modelado y comparacion`):
- Estado: `Completado`
- Benchmark exploratorio: [notebooks/sprint_03_modeling/00_exploracion_benchmark.ipynb](notebooks/sprint_03_modeling/00_exploracion_benchmark.ipynb)
- Comparacion de modelos: [reports/sprint_03/comparacion_modelos.md](reports/sprint_03/comparacion_modelos.md)
- Comparacion de modelos (resumen ejecutivo): [reports/sprint_03/comparacion_modelos_gas.md](reports/sprint_03/comparacion_modelos_gas.md)
- Resultados de tuning: [reports/sprint_03/tuning_results.md](reports/sprint_03/tuning_results.md)
- Artefactos generados del modelado: [reports/sprint_03/phase1_model_artifacts.md](reports/sprint_03/phase1_model_artifacts.md)
- Informe Sprint 3: [reports/sprint_03/informe_sprint_3.md](reports/sprint_03/informe_sprint_3.md)
- Reporte Sprint 3 (PDF): [reports/sprint_03/informes/sprint_03_reporte.pdf](reports/sprint_03/informes/sprint_03_reporte.pdf)
- Presentacion Sprint 3 (PDF): [reports/sprint_03/informes/sprint_3_presentacion.pdf](reports/sprint_03/informes/sprint_3_presentacion.pdf)

Sprint 4 (`integracion y demo`):
- Estado: `MVP funcional completado`
- Dashboard Streamlit: [app/dashboard/app.py](app/dashboard/app.py)
- API minima: [app/api/main.py](app/api/main.py)
- Guia operativa del MVP: [reports/sprint_04/guia_ejecucion_mvp.md](reports/sprint_04/guia_ejecucion_mvp.md)
- Pitch de Demo Day: [reports/sprint_04/pitch_demo_day.md](reports/sprint_04/pitch_demo_day.md)
- Informe etico y de gobernanza: [reports/sprint_04/informe_etico_gobernanza.md](reports/sprint_04/informe_etico_gobernanza.md)
- Reporte Sprint 4 (LaTeX/PDF): [reports/sprint_04/informes/sprint_04_reporte.pdf](reports/sprint_04/informes/sprint_04_reporte.pdf)

## 🚀 Scripts de ejecucion

### Prerequisitos (una sola vez)

Activar el entorno primero (venv o conda, ver seccion anterior), luego:

```bash
python scripts/csv_to_parquet.py        # CSV → Parquet
python3 scripts/create_temporal_split.py # split dev / backtest / holdout_3m
```

### Sprint 2 y Sprint 4 — Pipeline reproducible completo

**Con venv activo:**
```bash
bash scripts/build_features.sh
```

**Con conda (sin activar el env):**
```bash
CONDA_ENV=ai-miadas bash scripts/build_features.sh
```

El script detecta automaticamente el entorno: `CONDA_ENV` definido → conda, `.venv` en la raiz → venv, de lo contrario usa el `python` en PATH.

Ejecuta en orden los 7 pasos: master table dev → features dev → seleccion experimental → evaluacion vs baseline → master table backtest → features backtest → alineacion final para scoring del MVP.

Pasos individuales (con el entorno activado):

```bash
# [1/7] Master table DEV (umbral P80 = 197.01, modo fit)
python src/data/build_master_table.py --profile-source dev --threshold-mode auto

# [2/7] Features RFM DEV (18 variables nuevas → 51 columnas)
python src/features/build_rfm_features.py

# [3/7] Seleccion experimental (8 experimentos → ganador corr_le_0.85, 28 features)
python src/features/feature_selection_experiments.py

# [4/7] Evaluacion vs baseline Sprint 1 (split temporal 2018-07-01)
python src/models/evaluate_model.py

# [5/6] Master table HOLDOUT — simulacion mensual ago-oct 2018 (umbral apply)
python3 src/data/build_master_table.py --profile-source backtest --threshold-mode apply

# [6/6] Features RFM HOLDOUT
python src/features/build_rfm_features.py \
    --input-path data/processed/03_master_table_clean_backtest.parquet \
    --output-path data/processed/backtest_features_rfm.parquet \
    --metadata-path data/processed/backtest_features_rfm_metadata.json

# [7/7] Alineacion final al modelo oficial del MVP
python3 scripts/build_holdout_scoring_dataset.py --profile-name backtest
```

Para detalle de parametros y artefactos generados: [scripts/README.md](scripts/README.md)

### Sprint 3 — Generacion y validacion del modelo final

La fuente oficial del artefacto final es el notebook:

```bash
notebooks/sprint_03_modeling/04_evaluacion_modelo_final.ipynb
```

Ese notebook reconstruye el pipeline final, valida el modelo sobre el split temporal oficial de Sprint 3 y exporta:

```bash
models/final/modelo_final.pkl
```

Como referencia adicional, el script operativo que genera artefactos serializados intermedios del Sprint 3 es:

```bash
python src/models/run_phase1_notebook_artifacts.py
```

Genera estos `.pkl`:

- `models/baseline/lightgbm_baseline.pkl`
- `models/baseline/xgboost_baseline.pkl`
- `models/final/lightgbm_tuned.pkl`
- `models/final/xgboost_tuned.pkl`

Tambien persiste:

- `data/processed/13_phase1_summary.json`
- `reports/sprint_03/phase1_model_artifacts.md`
- tablas de importancias y auditoria de features en `data/processed/`

Nota: `scripts/train_model.sh` aun no esta implementado; por ahora la fuente oficial para generar los `.pkl` es `src/models/run_phase1_notebook_artifacts.py`.

### Sprint 4 — MVP en Docker

1. Generar artefactos de scoring del backtest:

```bash
bash scripts/build_features.sh
```

2. Levantar solo el dashboard:

```bash
docker compose up dashboard --build
```

3. Levantar dashboard y API:

```bash
docker compose up --build
```

4. URLs por defecto:

- Dashboard: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`
- API health: `http://localhost:8000/health`

Si el puerto `8000` ya esta ocupado:

```bash
API_PORT=8001 docker compose up api --build
```

Notas operativas del MVP:

- Los contenedores del dashboard y la API usan `Python 3.12`.
- `pipeline` y `pipeline-cron` quedaron en el perfil `pipeline`, por lo que `docker compose up` levanta por defecto el MVP y no el pipeline batch.
- La validacion metodologica principal del clasificador corresponde al Sprint 3; el backtest de Sprint 4 se usa para validacion operativa, explicabilidad y ROI del MVP.
- Para detalle completo de comandos: [reports/sprint_04/guia_ejecucion_mvp.md](reports/sprint_04/guia_ejecucion_mvp.md)

## 🗺️ Esquema de datos

- Archivo DBML local: `docs/dbdiagram/olist_schema.dbml`
- Vista online del modelo: https://dbdiagram.io/d/M13_G3-6a179a0bb62396d22c8862b8

## 🧪 Metodologia de trabajo (resumen)

1. Entendimiento y exploracion del problema de negocio.
2. Construccion del pipeline de datos y variables.
3. Entrenamiento y evaluacion de modelos.
4. Validacion temporal del modelo final (Sprint 3).
5. Integracion en un MVP con dashboard, API, explicabilidad y ROI (Sprint 4).

## 🚧 Estado del proyecto

En desarrollo academico por sprints.
Estado actual: **Sprint 4 completado a nivel MVP**,
con pipeline reutilizado, modelo final, dashboard Streamlit, API minima y documentacion de soporte para Demo Day.
