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
  - [Sprint 2 — Pipeline reproducible completo](#sprint-2--pipeline-reproducible-completo)
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
-> Master table  
-> Feature engineering (RFM y derivadas)  
-> Modelado supervisado  
-> Seguimiento de experimentos (MLflow)  
-> API / Dashboard

## 🧠 Definicion del target (actual)

La variable objetivo `is_premium` se define sobre gasto neto acumulado por cliente:

- `is_premium = 1` si `total_spent >= P80`.
- `is_premium = 0` para el resto.

Valores usados en Sprint 1:
- Umbral P80: `205.18 USD`.
- Distribucion: ~`80%` regulares y ~`20%` premium.
- Justificacion: alta asimetria del gasto y necesidad de criterio robusto frente a outliers.

## 🛠️ Stack tecnologico

- Python 3.12
- Pandas
- NumPy
- Scikit-learn
- PyArrow
- JupyterLab
- MLflow
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
- `src/`: scripts de carga, features, modelado y visualizacion
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
- Hipotesis, target y metricas: [reports/sprint_01/documento_hipotesis_target_metricas.md](reports/sprint_01/documento_hipotesis_target_metricas.md)
- Storytelling breve: [reports/sprint_01/storytelling_breve.md](reports/sprint_01/storytelling_breve.md)
- Reporte Sprint 1 (PDF): [reports/sprint_01/informes/sprint_01_reporte.pdf](reports/sprint_01/informes/sprint_01_reporte.pdf)
- Presentacion final Sprint 1 (PDF): [reports/sprint_01/informes/sprint_01_presentacion.pdf](reports/sprint_01/informes/sprint_01_presentacion.pdf)

Sprint 2 (`pipeline de features`):
- Estado: `Completado`
- Plan y checklist: [reports/sprint_02/plan_implementacion_sprint_2.md](reports/sprint_02/plan_implementacion_sprint_2.md)
- DAG tecnico reproducible: [reports/sprint_02/dag_pipeline_sprint_2.md](reports/sprint_02/dag_pipeline_sprint_2.md)
- Documentacion del pipeline: [reports/sprint_02/documentacion_pipeline.md](reports/sprint_02/documentacion_pipeline.md)
- Notebook integrador: [notebooks/sprint_02_pipeline/02_pipeline_features.ipynb](notebooks/sprint_02_pipeline/02_pipeline_features.ipynb)
- Evaluacion vs baseline: [reports/sprint_02/evaluation_vs_baseline.md](reports/sprint_02/evaluation_vs_baseline.md)

Sprint 3 (`modelado y comparacion`):
- Estado: `En progreso`

Sprint 4 (`integracion y demo`):
- Estado: `Pendiente`

## 🚀 Scripts de ejecucion

### Prerequisitos (una sola vez)

Activar el entorno primero (venv o conda, ver seccion anterior), luego:

```bash
python scripts/csv_to_parquet.py        # CSV → Parquet
python scripts/create_temporal_split.py # split dev / holdout_3m
```

### Sprint 2 — Pipeline reproducible completo

**Con venv activo:**
```bash
bash scripts/build_features.sh
```

**Con conda (sin activar el env):**
```bash
CONDA_ENV=ai-miadas bash scripts/build_features.sh
```

El script detecta automaticamente el entorno: `CONDA_ENV` definido → conda, `.venv` en la raiz → venv, de lo contrario usa el `python` en PATH.

Ejecuta en orden los 6 pasos: master table dev → features → seleccion experimental → evaluacion vs baseline → master table holdout → features holdout.

Pasos individuales (con el entorno activado):

```bash
# [1/4] Master table DEV (umbral P80 = 197.01, modo fit)
python src/data/build_master_table.py --profile-source dev --threshold-mode auto

# [2/4] Features RFM DEV (18 variables nuevas → 51 columnas)
python src/features/build_rfm_features.py

# [3/4] Seleccion experimental (8 experimentos → ganador corr_le_0.85, 28 features)
python src/features/feature_selection_experiments.py

# [4/4] Evaluacion vs baseline Sprint 1 (split temporal 2018-07-01)
python src/models/evaluate_model.py

# [5/6] Master table HOLDOUT — simulacion mensual ago-oct 2018 (umbral apply)
python src/data/build_master_table.py --profile-source holdout --threshold-mode apply

# [6/6] Features RFM HOLDOUT
python src/features/build_rfm_features.py \
    --input-path data/processed/03_master_table_clean_holdout.parquet \
    --output-path data/processed/holdout_features_rfm.parquet \
    --metadata-path data/processed/holdout_features_rfm_metadata.json
```

Para detalle de parametros y artefactos generados: [scripts/README.md](scripts/README.md)

### Sprint 3 — Generacion de modelos `.pkl`

El script operativo que hoy genera los artefactos serializados de modelos es:

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

## 🗺️ Esquema de datos

- Archivo DBML local: `docs/dbdiagram/olist_schema.dbml`
- Vista online del modelo: https://dbdiagram.io/d/M13_G3-6a179a0bb62396d22c8862b8

## 🧪 Metodologia de trabajo (resumen)

1. Entendimiento y exploracion del problema de negocio.
2. Construccion del pipeline de datos y variables.
3. Entrenamiento y evaluacion de modelos.
4. Integracion en una demo con componentes de despliegue.

## 🚧 Estado del proyecto

En desarrollo academico por sprints.
Estado actual: **Sprint 2 completado** (pipeline reproducible,
features, seleccion experimental, metricas y KPIs finales).
