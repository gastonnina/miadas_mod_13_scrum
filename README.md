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
- [🧠 Definicion preliminar del target](#-definicion-preliminar-del-target)
- [🛠️ Stack tecnologico](#️-stack-tecnologico)
- [📊 Dataset](#-dataset)
- [⚙️ Configuracion del entorno](#️-configuracion-del-entorno)
- [📁 Estructura del repositorio](#-estructura-del-repositorio)
- [🧪 Metodologia de trabajo (resumen)](#-metodologia-de-trabajo-resumen)
- [🚧 Estado del proyecto](#-estado-del-proyecto)


## 👥 Integrantes del grupo

- Gaston Humberto Gerick Marcelo Nelson
- Nina Sossa Toro Rodriguez De la Quintana Illanes

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

## 🧠 Definicion preliminar del target

La variable objetivo `is_premium` se construira a partir del comportamiento transaccional del cliente.

Definicion inicial (sujeta a validacion en EDA):
- `is_premium = 1` para clientes con mayor valor acumulado de compra (`total_spent`) y recurrencia (`frequency`) en el periodo analizado.
- `is_premium = 0` para el resto de clientes.

Criterio preliminar sugerido para baseline:
- Usar percentil (por ejemplo p75 o p80) sobre `total_spent` combinado con umbral minimo de `frequency`.

La definicion final se cerrara despues del analisis exploratorio y validacion con metricas de negocio.

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

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
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

## 🧪 Metodologia de trabajo (resumen)

1. Entendimiento y exploracion del problema de negocio.
2. Construccion del pipeline de datos y variables.
3. Entrenamiento y evaluacion de modelos.
4. Integracion en una demo con componentes de despliegue.

## 🚧 Estado del proyecto

En desarrollo academico por sprints, de acuerdo con el plan del modulo.
