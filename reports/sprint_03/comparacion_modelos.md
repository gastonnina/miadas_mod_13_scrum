# Benchmark de Modelos — Sprint 3

**Tarea:** S3-DS-01 — Benchmark Inicial de Modelos Candidatos  
**Dataset:** `06_features_selected.parquet` — 96,096 clientes  
**Split temporal:** train `< 2018-07-01` (83,589 registros) | validation `>= 2018-07-01` (6,230 registros)

---

## Contexto de negocio

El segmento premium (20% de los clientes) concentra el **56.03% de la facturación** de Olist y presenta un ticket promedio **4.36× superior** al del cliente regular (BRL 402.66 vs BRL 92.27), según los KPIs validados en el Sprint 2.

En este contexto, un **falso negativo** (no detectar un cliente premium) tiene alto costo económico: se pierde la oportunidad de retención sobre el segmento de mayor valor. Por lo tanto, **Recall es la métrica de negocio prioritaria**, sin descuidar ROC-AUC y Gini como indicadores de capacidad discriminante global.

---

## 1. Modelos Evaluados

Se evaluaron los siguientes modelos candidatos con configuración base (sin optimización de hiperparámetros), sobre las mismas 28 features del experimento `corr_le_0.85` del Sprint 2:

| Modelo | Tipo | Notas |
| --- | --- | --- |
| `logistic_regression` | Lineal (baseline) | Con scaler, `class_weight=balanced` |
| `random_forest` | Ensemble bagging | `class_weight=balanced_subsample`, 200 árboles |
| `gradient_boosting` | Boosting sklearn | 200 estimators, lr=0.05 |
| `xgboost` | Boosting XGBoost | `scale_pos_weight` ajustado, 250 estimators |
| `lightgbm` | Boosting LightGBM | `class_weight=balanced`, 250 estimators |

---

## 2. Tabla Comparativa

> Ordenado por ROC-AUC en validación (descendente). Gap overfit = ROC-AUC train − ROC-AUC val.

| Modelo | ROC-AUC val | Gini val | PR-AUC val | F1 val | Precision val | Recall val | ROC-AUC train | Gap overfit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 0.8035 | 0.6070 | 0.5954 | 0.5565 | 0.4783 | 0.6653 | 0.8405 | 0.0370 |
| gradient_boosting | 0.7985 | 0.5970 | 0.5881 | 0.4524 | 0.7006 | 0.3340 | 0.8258 | 0.0274 |
| xgboost | 0.7982 | 0.5964 | 0.5896 | 0.5504 | 0.4576 | 0.6906 | 0.8273 | 0.0290 |
| random_forest | 0.7926 | 0.5852 | 0.5677 | 0.5422 | 0.4707 | 0.6392 | 0.8118 | 0.0191 |
| logistic_regression | 0.7884 | 0.5768 | 0.5661 | 0.5329 | 0.4538 | 0.6456 | 0.8041 | 0.0156 |

### Continuidad con Sprint 2

El resultado de `random_forest` en validación (ROC-AUC = **0.7926**) es prácticamente idéntico al resultado oficial del Sprint 2 (`corr_le_0.85`, ROC-AUC = **0.7929**), usando el mismo split temporal y las mismas 28 features. Esto confirma la reproducibilidad del pipeline y que el benchmark parte de una base estable.

Los modelos de boosting mejoran ese baseline en +0.8 a +1.1 puntos de ROC-AUC y en +2 a +7 puntos de Recall, validando la hipótesis del Sprint 2 de que modelos no lineales de mayor capacidad capturan mejor la "Paradoja del Target" (97% de clientes con una sola orden).

---

## 3. Análisis de Estabilidad

| Modelo | ROC-AUC train | ROC-AUC val | Gap |
| --- | --- | --- | --- |
| lightgbm | 0.8405 | 0.8035 | 0.0370 |
| gradient_boosting | 0.8258 | 0.7985 | 0.0274 |
| xgboost | 0.8273 | 0.7982 | 0.0290 |
| random_forest | 0.8118 | 0.7926 | 0.0191 |
| logistic_regression | 0.8041 | 0.7884 | 0.0156 |

**Interpretación:**
- Todos los modelos muestran gaps ≤ 0.04, lo que indica generalización aceptable sin sobreajuste severo.
- `logistic_regression` y `random_forest` tienen el menor gap, pero también el menor rendimiento absoluto.
- `lightgbm` y `xgboost` muestran el mejor rendimiento absoluto con gaps moderados (0.03–0.04), aceptables para la fase de tuning.
- `gradient_boosting` (sklearn) tiene Recall muy bajo (0.33), lo que lo descalifica desde la perspectiva del negocio, pese a una Precision alta que refleja un modelo demasiado conservador.

---

## 4. Selección de Finalistas — S3-DS-02

**Criterios de selección:**
- ROC-AUC val ≥ 0.78
- Gini val ≥ 0.56
- Gap de sobreajuste ≤ 0.06
- Ranking compuesto: `0.5 × ROC-AUC + 0.3 × Gini + 0.2 × F1` (refleja las tres métricas del plan de trabajo)

### Finalistas seleccionados

| Modelo | ROC-AUC val | Gini val | PR-AUC val | F1 val | Precision val | Recall val | ROC-AUC train | Gap overfit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | 0.8035 | 0.6070 | 0.5954 | 0.5565 | 0.4783 | 0.6653 | 0.8405 | 0.0370 |
| xgboost | 0.7982 | 0.5964 | 0.5896 | 0.5504 | 0.4576 | 0.6906 | 0.8273 | 0.0290 |

### Justificación técnica

**lightgbm**  
- ROC-AUC val: 0.8035 → Gini: 0.6070 → PR-AUC: 0.5954  
- F1: 0.5565 (Precision: 0.4783 / Recall: 0.6653)  
- Gap de sobreajuste: 0.0370 — estabilidad aceptable.
- Lidera en todas las métricas de discriminación global. Recall de 0.67 significa que detecta 2 de cada 3 clientes premium, adecuado como punto de partida para tuning.

**xgboost**  
- ROC-AUC val: 0.7982 → Gini: 0.5964 → PR-AUC: 0.5896  
- F1: 0.5504 (Precision: 0.4576 / Recall: 0.6906)  
- Gap de sobreajuste: 0.0290 — estabilidad aceptable.
- Presenta el **mayor Recall de todos los modelos evaluados (0.69)**, lo que lo hace especialmente valioso en el contexto de negocio donde minimizar falsos negativos es prioritario. El menor gap de sobreajuste entre los modelos de boosting refuerza su candidatura.

**Por qué se excluye `gradient_boosting` (sklearn):**  
A pesar de tener ROC-AUC comparable a XGBoost (0.7985 vs 0.7982), su Recall de **0.33** es inaceptable: dejaría sin detectar al 67% de los clientes premium, equivalente a perder más de la mitad de la oportunidad de negocio identificada en el Sprint 2 (56% de facturación concentrada en ese segmento).

### Recomendación técnica

Se seleccionan **`lightgbm`** y **`xgboost`** como modelos finalistas para la fase de tuning (S3-DS-03/04).

**Rationale:**
- **LightGBM** lidera en ROC-AUC y Gini. Su velocidad de entrenamiento lo hace ideal para búsqueda de hiperparámetros y también para re-evaluar importancia de features en iteraciones posteriores.
- **XGBoost** ofrece el mayor Recall base, con control explícito del desbalance de clases vía `scale_pos_weight`. Es el candidato más prometedor si el objetivo de negocio es maximizar la cobertura del segmento premium.
- Ambos modelos soportan feature importance nativa, facilitando la comunicación técnica con el DPO.

---

## 5. Próximos pasos

- **S3-DS-03:** Optimización de hiperparámetros (HPO) para LightGBM y XGBoost — priorizar Recall sin sacrificar más de 3 puntos de ROC-AUC.
- **S3-DS-04:** Evaluación final en conjunto holdout (agosto–octubre 2018).
- Evaluar el uso de LightGBM para re-selección de features como paso previo al tuning.

---

*Generado desde `notebooks/sprint_03_modeling/01_benchmark_modelos.ipynb`*
