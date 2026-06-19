# Informe Técnico — Sprint 3: Hiperparametrización y Modelo Final

**Proyecto:** Identificación de Clientes Premium — Olist Brazilian E-Commerce  
**Caso de Estudio:** Caso 3 — Clasificación Supervisada (`is_premium`)  
**Sprint:** 3 de 4 | Semana 3  
**Entregable principal:** Modelo de clasificación serializado (`.pkl`) + análisis comparativo de modelos  

---

## 0. Contexto: ¿De dónde venimos?

Para entender las decisiones del Sprint 3, es necesario conocer los hallazgos de los sprints anteriores.

### Sprint 1 — Definición del problema y EDA

El análisis exploratorio (EDA) reveló dos hallazgos clave:

**1. Concentración del gasto (Curva de Pareto):**  
El 20% de los clientes con mayor gasto acumulado concentra el **56.03% de la facturación total** de Olist. Este segmento —que llamamos "clientes premium"— tiene un ticket promedio de **BRL 402.66**, mientras que el cliente regular promedia **BRL 92.27** (diferencia de 4.36×).

> *¿Qué es el ticket promedio?* Es el valor monetario promedio que un cliente gasta cada vez que compra. Un cliente premium gasta 4 veces más por transacción que uno regular.

**2. La "Paradoja del Target" (hallazgo crítico):**  
El 97% de los clientes del dataset tiene una única orden de compra en toda su historia. Esto hace que las métricas RFM clásicas (Recency, Frequency, Monetary) sean débiles predictores, porque casi nadie tiene historial de frecuencia real. Los modelos deben aprender a discriminar premium vs. regular usando principalmente el valor monetario de esa única compra y características del comportamiento en esa transacción.

**Definición de la variable objetivo `is_premium`:**

- Se calculó el percentil 80 (P80) del gasto total acumulado por cliente en el conjunto de desarrollo.
- Umbral resultante: **BRL 197.01**
- Un cliente es `is_premium = 1` si su `total_spent >= 197.01`, de lo contrario `is_premium = 0`.
- Tasa de positivos en el dataset: **~21.3% en train** / **~22.8% en validación**.

> *¿Por qué P80 y no P90?* Usar P80 captura exactamente al 20% superior de clientes, que es el segmento identificado por la Curva de Pareto. Es un umbral con justificación de negocio directa.

---

### Sprint 2 — Ingeniería de datos y selección de features

Se construyó un pipeline reproducible que genera 28 variables a partir de los datos históricos de cada cliente. Estas variables se organizan en dominios:

| Dominio | Ejemplos de variables |
|---|---|
| Comportamiento transaccional | `total_orders`, `total_items`, `total_products` |
| Valor del cliente | `recency_days`, `customer_lifetime_days` |
| Logística | `avg_delivery_days`, `avg_estimated_delivery_days`, `delivery_gap`, `late_deliveries` |
| Pagos | `max_payment_installments`, `payment_methods_count`, `credit_card_flag` |
| Reseñas | `avg_review_score`, `reviews_per_order` |
| Geografía | `customer_state`, `region_group`, `far_region_flag` |
| Producto | `top_category`, `top_category_group`, `top_category_is_high_value` |

Se evaluaron múltiples experimentos de selección de features y se eligió el experimento **`corr_le_0.85`** como el mejor balance entre cantidad de variables y capacidad predictiva, con un ROC-AUC de referencia de **0.7929** usando Random Forest base.

> *¿Qué es ROC-AUC?* Es el Área bajo la Curva ROC (Receiver Operating Characteristic). Mide qué tan bien el modelo distingue entre clientes premium y regulares en todos los umbrales posibles. Un valor de 1.0 es perfecto; 0.5 es equivalente a clasificar al azar.

**Variables excluidas explícitamente del modelo (fuga de información):**  
Variables como `total_spent`, `avg_ticket`, `avg_order_price` se excluyeron porque son el criterio directo de definición de `is_premium`. Usarlas haría el modelo trivialmente perfecto en entrenamiento pero inútil en producción.

> *¿Qué es fuga de información (data leakage)?* Ocurre cuando el modelo aprende usando datos que no estarían disponibles en el momento de hacer la predicción real. En este caso, conocer cuánto gastó un cliente haría trivial saber si es premium.

**Split temporal (separación tren/validación por fecha):**

| Conjunto | Criterio | Registros |
|---|---|---|
| **Train** (entrenamiento) | `last_purchase < 2018-07-01` | 83,589 clientes |
| **Validation** (validación) | `last_purchase >= 2018-07-01` | 6,230 clientes |

> *¿Por qué split temporal y no aleatorio?* En producción, el modelo siempre predice sobre clientes futuros. Usar split temporal simula esa condición: se entrena con historia pasada y se evalúa con datos más recientes. Un split aleatorio inflaría artificialmente las métricas.

---

## 1. Objetivo del Sprint 3

Tomar los 28 features del Sprint 2 y responder: **¿qué modelo y con qué hiperparámetros clasifica mejor a los clientes premium?**

El sprint se dividió en cuatro fases:

```
S3-DS-01  Benchmark inicial (5 modelos, configuración base)
    ↓
S3-DS-02  Selección de finalistas (criterio técnico + negocio)
    ↓
S3-DS-03  Optimización de hiperparámetros LightGBM (Optuna)
S3-DS-04  Optimización de hiperparámetros XGBoost (Optuna)
    ↓
         Evaluación del modelo ganador + exportación .pkl
```

---

## 2. S3-DS-01: Benchmark de Modelos Candidatos

**Notebook:** `notebooks/sprint_03_modeling/01_benchmark_modelos.ipynb`  
**Artefacto:** `data/processed/09_model_benchmark.json`

### 2.1 Modelos evaluados

Se compararon 5 modelos con configuración base (sin optimizar hiperparámetros), todos entrenados sobre las mismas 28 features y el mismo split temporal:

| Modelo | Tipo | Configuración base |
|---|---|---|
| `logistic_regression` | Lineal (baseline) | `class_weight='balanced'`, scaler estándar |
| `random_forest` | Ensemble bagging | 200 árboles, `class_weight='balanced_subsample'` |
| `gradient_boosting` | Boosting sklearn | 200 estimadores, `learning_rate=0.05` |
| `xgboost` | Boosting XGBoost | 250 estimadores, `scale_pos_weight=3.70` |
| `lightgbm` | Boosting LightGBM | 250 estimadores, `class_weight='balanced'` |

> *¿Por qué `class_weight='balanced'` o `scale_pos_weight`?* El dataset tiene desbalance de clases: aproximadamente 3.70 clientes regulares por cada premium. Sin corrección, el modelo aprendería a clasificar a todos como "regulares" y obtendría 78% de accuracy por inercia, sin aprender nada útil. Estos parámetros compensan el desbalance.

> *¿Qué es ensemble bagging vs. boosting?*  
> - **Bagging** (Random Forest): entrena múltiples árboles en paralelo sobre muestras aleatorias y promedia sus predicciones. Reduce varianza.  
> - **Boosting** (GradientBoosting, XGBoost, LightGBM): entrena árboles secuencialmente, donde cada árbol corrige los errores del anterior. Reduce sesgo y tiene mayor capacidad para capturar patrones complejos.

### 2.2 Resultados del benchmark

> Ordenado por ROC-AUC en validación (descendente). Gap overfit = ROC-AUC train − ROC-AUC validación.

| Modelo | ROC-AUC val | Gini val | PR-AUC val | F1 val | Precision val | Recall val | Gap overfit |
|---|---:|---:|---:|---:|---:|---:|---:|
| **lightgbm** | **0.8035** | **0.6070** | **0.5954** | **0.5565** | 0.4783 | 0.6653 | 0.037 |
| gradient_boosting | 0.7985 | 0.5970 | 0.5881 | 0.4524 | **0.7006** | 0.3340 | 0.027 |
| xgboost | 0.7982 | 0.5964 | 0.5896 | 0.5504 | 0.4576 | **0.6906** | 0.029 |
| random_forest | 0.7926 | 0.5852 | 0.5677 | 0.5422 | 0.4707 | 0.6392 | 0.019 |
| logistic_regression | 0.7884 | 0.5768 | 0.5661 | 0.5329 | 0.4538 | 0.6456 | 0.016 |

**Glosario de métricas:**

- **ROC-AUC**: Capacidad discriminante global del modelo (0.5 = azar, 1.0 = perfecto).
- **Gini** = 2 × ROC-AUC − 1: Versión normalizada de ROC-AUC, popular en scoring financiero. Mide qué tan mejor que el azar es el modelo (0 = igual que azar, 1 = perfecto).
- **PR-AUC** (Precision-Recall AUC): Área bajo la curva Precision-Recall. Más informativa que ROC-AUC cuando las clases están desbalanceadas, porque penaliza los falsos positivos en el contexto de la clase minoritaria.
- **F1**: Media armónica de Precision y Recall. Equilibra ambas métricas.
- **Precision**: De todos los clientes que el modelo predice como premium, ¿qué fracción realmente lo es? Alta precision → pocos falsos positivos.
- **Recall**: De todos los clientes que realmente son premium, ¿qué fracción detecta el modelo? Alto recall → pocos falsos negativos.
- **Gap overfit**: Si es muy alto (>0.10), el modelo memorizó el training y no generaliza bien a datos nuevos.

### 2.3 Continuidad con Sprint 2

El resultado de `random_forest` en validación (ROC-AUC = **0.7926**) es prácticamente idéntico al resultado oficial del Sprint 2 (`corr_le_0.85`, ROC-AUC = **0.7929**). Esta coincidencia confirma que el pipeline del Sprint 3 es exactamente el mismo que el del Sprint 2 — los resultados son reproducibles.

Los modelos de boosting mejoran ese baseline entre +0.6 y +1.5 puntos de ROC-AUC, validando la hipótesis de que modelos no lineales de mayor capacidad capturan mejor la Paradoja del Target.

---

## 3. S3-DS-02: Selección de Finalistas

### 3.1 Criterios de selección

Se aplicaron cuatro criterios combinados:

| Criterio | Umbral | Justificación |
|---|---|---|
| ROC-AUC val ≥ 0.78 | mínimo aceptable | Mejora significativa sobre baseline Sprint 2 |
| Gini val ≥ 0.56 | mínimo aceptable | Equivale a ROC-AUC ≥ 0.78 |
| Gap overfit ≤ 0.06 | estabilidad | Modelos con gap mayor son inestables en producción |
| Ranking compuesto | top 2 | `0.5 × ROC-AUC + 0.3 × Gini + 0.2 × F1` |

> *¿Por qué un ranking compuesto en lugar de ordenar solo por ROC-AUC?*  
> El `gradient_boosting` tenía ROC-AUC=0.7985 (segundo lugar) pero Recall=0.33, lo que significa que dejaría sin detectar al 67% de los clientes premium. Ordenar solo por ROC-AUC habría seleccionado un modelo que falla en el objetivo de negocio. El ranking compuesto incorpora F1 (que castiga ese bajo Recall) y selecciona correctamente a XGBoost sobre GradientBoosting.

### 3.2 Finalistas seleccionados

**LightGBM** y **XGBoost** pasan a la fase de optimización.

| Argumento | LightGBM | XGBoost |
|---|---|---|
| ROC-AUC val | 0.8035 (mejor) | 0.7982 |
| Recall val | 0.6653 | **0.6906 (mejor de todos)** |
| F1 val | 0.5565 | 0.5504 |
| Gap overfit | 0.037 | 0.029 |
| Fortaleza principal | Discriminación global | Mayor Recall base |

**¿Por qué se descarta `gradient_boosting` (sklearn)?**  
Recall = 0.33: dejaría sin detectar al 67% de los clientes premium. En términos de negocio, significaría perder la oportunidad de retención sobre más de la mitad del segmento responsable del 56% de la facturación.

**Prioridad de negocio — ¿por qué Recall es la métrica clave?**  
Un falso negativo (clasificar como regular a alguien que es premium) tiene un costo alto: se pierde la oportunidad de acción sobre el cliente de mayor valor. Un falso positivo (clasificar como premium a alguien que no lo es) tiene menor costo: se ofrece un beneficio innecesario a un cliente regular. Por esto, **Recall es la métrica de negocio prioritaria**, complementada por ROC-AUC y Gini como indicadores de capacidad discriminante global.

---

## 4. S3-DS-03/04: Optimización de Hiperparámetros

**Notebook:** `notebooks/sprint_03_modeling/02_hyperparameter_tuning.ipynb`  
**Artefacto:** `data/processed/10_tuning_results.json`

### 4.1 Método: Optuna TPE

Se utilizó **Optuna** con el método de muestreo **TPE** (Tree-structured Parzen Estimator) para buscar los mejores hiperparámetros de cada modelo.

> *¿Qué es HPO (Hyperparameter Optimization)?* Los modelos de machine learning tienen configuraciones internas llamadas hiperparámetros (por ejemplo, cuántos árboles usar, qué tan profundos, qué tasa de aprendizaje). Estos no se aprenden de los datos; hay que definirlos antes de entrenar. HPO busca la combinación óptima de forma automática.

> *¿Qué es TPE (Tree-structured Parzen Estimator)?* Es un algoritmo bayesiano de optimización. En lugar de probar combinaciones de hiperparámetros al azar (como GridSearch), TPE aprende de los intentos anteriores para dirigir la búsqueda hacia las regiones del espacio que prometen mejores resultados. Esto lo hace significativamente más eficiente que la búsqueda por grilla (GridSearchCV).

> *¿Qué es Optuna?* Librería Python de código abierto para HPO automático. Permite definir el espacio de búsqueda y el objetivo a optimizar, y gestiona los trials (intentos) internamente.

**Configuración:**
- 50 trials por modelo (50 combinaciones de hiperparámetros probadas)
- Objetivo: maximizar ROC-AUC en el conjunto de validación
- Mismo split temporal que el benchmark

### 4.2 Espacio de búsqueda

**LightGBM:**

| Hiperparámetro | Rango | Efecto |
|---|---|---|
| `num_leaves` | 20–150 | Complejidad del árbol; más hojas = más expresivo pero más riesgo de sobreajuste |
| `max_depth` | 3–10 | Profundidad máxima del árbol |
| `learning_rate` | 0.01–0.20 | Cuánto aprende cada árbol; valores pequeños son más robustos |
| `n_estimators` | 100–500 | Número de árboles en el ensemble |
| `min_child_samples` | 10–100 | Mínimo de muestras en una hoja; mayor valor = más regularización |
| `subsample` | 0.6–1.0 | Fracción de datos usada por árbol (bagging) |
| `colsample_bytree` | 0.6–1.0 | Fracción de features usada por árbol |
| `reg_alpha`, `reg_lambda` | log-uniforme | Regularización L1 y L2 |

**XGBoost:** espacio similar con parámetros equivalentes (`max_depth`, `min_child_weight`, `gamma`, etc.)

### 4.3 Resultados del tuning

| Modelo | ROC-AUC val | Gini val | PR-AUC val | F1 val | Recall val | Gap overfit | Mejora AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM baseline | 0.8035 | 0.6070 | 0.5954 | 0.5565 | 0.6653 | 0.037 | — |
| **LightGBM tuneado** | **0.8081** | **0.6162** | **0.6010** | **0.5576** | 0.6617 | 0.039 | **+0.0046** |
| XGBoost baseline | 0.7982 | 0.5964 | 0.5896 | 0.5504 | 0.6906 | 0.029 | — |
| XGBoost tuneado | 0.8070 | 0.6140 | 0.5997 | 0.5560 | 0.6505 | 0.054 | +0.0088 |

**Observación importante sobre XGBoost tuneado:**  
El tuning mejoró el ROC-AUC de XGBoost en +0.0088 (la mayor mejora absoluta), pero su **Recall bajó de 0.6906 a 0.6505**. Esto ocurre porque Optuna optimizó para maximizar ROC-AUC, no Recall. El modelo ganó capacidad discriminante global a costa de detectar menos clientes premium. Este trade-off debe considerarse si en una iteración futura se quisiera optimizar Recall directamente.

### 4.4 Hiperparámetros óptimos del modelo ganador (LightGBM)

```json
{
  "num_leaves":        59,
  "max_depth":         8,
  "learning_rate":     0.0540,
  "n_estimators":      255,
  "min_child_samples": 68,
  "subsample":         0.9059,
  "colsample_bytree":  0.8554,
  "reg_alpha":         6.7e-08,
  "reg_lambda":        0.0006
}
```

---

## 5. Modelo Ganador: LightGBM Tuneado

### 5.1 ¿Por qué LightGBM?

1. **Mayor ROC-AUC y Gini** de todos los modelos evaluados, tanto en baseline como después del tuning.
2. **Gap de sobreajuste controlado** (0.039): diferencia pequeña entre entrenamiento y validación, lo que indica que el modelo generaliza bien.
3. **Velocidad**: LightGBM es significativamente más rápido que XGBoost en entrenamiento, especialmente con muchos trials de HPO. Facilita iteraciones futuras.
4. **Feature importance nativa**: el modelo permite analizar directamente qué variables influyen más en la predicción, facilitando la comunicación con el equipo de negocio.

> *¿Qué es sobreajuste (overfitting)?* Cuando un modelo aprende "de memoria" los datos de entrenamiento y no generaliza bien a datos nuevos. Un indicador es un gap alto entre métricas de train y validación. Gaps < 0.05 son típicamente aceptables.

### 5.2 Métricas finales (umbral óptimo = 0.55)

| Métrica | Valor | Interpretación |
|---|---:|---|
| ROC-AUC | **0.8081** | Discriminación global excelente (baseline S2: 0.7929) |
| Gini | **0.6162** | 62% mejor que clasificación aleatoria |
| PR-AUC | **0.6010** | Buena precisión en la clase minoritaria (premium) |
| F1 | **0.5576** | Balance razonable Precision-Recall |
| Precision | 0.5225 | De cada 2 clientes predichos como premium, ~1 realmente lo es |
| Recall | **0.6041** | Detecta el 60% de los clientes premium reales* |

> *Nota: estos valores son con el umbral óptimo (0.55). Con umbral por defecto (0.50), el Recall es 0.6617 y Precision es 0.4818 — el Recall es mayor pero la Precision baja.*

### 5.3 Análisis del umbral de decisión

Un modelo de clasificación produce una probabilidad entre 0 y 1 para cada cliente. La decisión de "premium" o "regular" depende de un **umbral**: si la probabilidad supera el umbral, se clasifica como premium.

- **Umbral por defecto (0.50):** Recall=0.6617, Precision=0.4818, F1=0.5576
- **Umbral óptimo (0.55):** Recall=0.6041, Precision=0.5225, F1=0.5603 ← mayor F1
- **Umbral bajo (0.30):** Recall muy alto pero Precision muy baja → muchos falsos positivos

> *¿Cómo elegir el umbral?* Depende del objetivo de negocio. Si se prioriza Recall (no perder ningún cliente premium), se baja el umbral. Si se prioriza Precision (invertir solo en clientes verdaderamente premium), se sube. Para la fase de producción, el equipo de negocio (DPO) debe tomar esta decisión.

**Figura de referencia:** `reports/figures/sprint_03_final_threshold.png`

---

## 6. Interpretación de Feature Importance

**Notebook:** `notebooks/sprint_03_modeling/03_interpretacion_feature_audit.ipynb`  
**Artefactos:** `data/processed/13_feature_audit_lightgbm.parquet`, `13_feature_audit_xgboost.parquet`

### 6.1 Top 10 variables — LightGBM tuneado (por importancia de ganancia)

| # | Variable | Dominio | Importancia | Interpretación |
|---|---|---|---:|---|
| 1 | `top_category` | Producto | 21.98% | Categoría de producto más comprada por el cliente |
| 2 | `recency_days` | Temporal | 21.02% | Días desde la última compra hasta el corte |
| 3 | `avg_estimated_delivery_days` | Logística | 10.62% | Tiempo de entrega estimado promedio |
| 4 | `avg_delivery_days` | Logística | 8.53% | Tiempo de entrega real promedio |
| 5 | `delivery_gap` | Derivada | 6.73% | Diferencia entre entrega estimada y real |
| 6 | `max_payment_installments` | Pagos | 6.64% | Máximo de cuotas usadas en alguna compra |
| 7 | `customer_state` | Geografía | 4.16% | Estado de residencia del cliente |
| 8 | `total_items` | Transaccional | 4.12% | Total de ítems comprados en toda la historia |
| 9 | `avg_review_score` | Reviews | 2.64% | Calificación promedio dada por el cliente |
| 10 | `region_group` | Geografía | 2.24% | Región del país del cliente |

> *¿Qué es importancia por ganancia (gain)?* Mide cuánto reduce el error del modelo cada variable cuando se usa para hacer una división en los árboles. Una variable con alta ganancia divide bien los ejemplos entre premium y regular.

**Interpretación de negocio:**
- **`top_category`**: Los clientes premium compran en categorías de mayor valor (electrónica, relojes, instrumentos musicales). Esto explica el alto ticket promedio.
- **`recency_days`**: Los clientes más recientes tienden a ser premium (o al menos a tener pedidos más recientes con mayor valor). Esto refleja la dinámica temporal del mercado.
- **Variables logísticas (`avg_estimated_delivery_days`, `delivery_gap`)**: Los clientes premium suelen comprar productos con envíos más complejos o largos, probablemente desde regiones más remotas o con productos más grandes/caros.
- **`max_payment_installments`**: Compras en cuotas sugieren capacidad de crédito y productos de mayor valor.

**Figura de referencia:** `reports/figures/sprint_03_feature_importance.png`

---

## 7. Validación y Reproducibilidad del Modelo

**Notebook:** `notebooks/sprint_03_modeling/04_evaluacion_modelo_final.ipynb`  
**Artefacto:** `models/final/modelo_final.pkl`

### 7.1 Pipeline serializado

El modelo final es un **pipeline sklearn** con dos componentes:

```
Pipeline
├── preprocessor (ColumnTransformer)
│   ├── num → SimpleImputer(median) → 22 variables numéricas
│   └── cat → SimpleImputer(most_frequent) + OneHotEncoder → 6 variables categóricas
└── model → LGBMClassifier(hiperparámetros óptimos, class_weight='balanced')
```

> *¿Por qué un Pipeline y no solo el modelo?* Un Pipeline encapsula todas las transformaciones de datos junto con el modelo en un único objeto. Cuando se llama `pipeline.predict(X)`, automáticamente aplica el mismo preprocesamiento que se usó en entrenamiento. Esto garantiza que en producción no se olvide ningún paso de transformación.

> *¿Qué es OneHotEncoder?* Convierte variables categóricas (texto) en columnas binarias (0/1). Por ejemplo, `customer_state = "SP"` se convierte en una columna `state_SP = 1` y el resto en 0.

### 7.2 Cómo cargar y usar el modelo

```python
import pickle
import pandas as pd

# Cargar el modelo
with open('models/final/modelo_final.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Preparar datos de un nuevo cliente (mismo formato que 06_features_selected.parquet)
nuevo_cliente = pd.DataFrame([{
    'total_orders': 2,
    'total_items': 5,
    'recency_days': 30,
    # ... (28 features en total)
}])

# Obtener probabilidad de ser premium
probabilidad = pipeline.predict_proba(nuevo_cliente)[0, 1]
es_premium = probabilidad >= 0.55  # umbral óptimo

print(f'Probabilidad de premium: {probabilidad:.2%}')
print(f'Clasificación: {"PREMIUM" if es_premium else "regular"}')
```

### 7.3 Verificación round-trip

El notebook 04 verifica que el modelo serializado produce exactamente las mismas predicciones que el modelo en memoria (delta ROC-AUC < 1×10⁻⁸), confirmando que el `.pkl` es confiable y reproducible.

**Nota de compatibilidad:** El archivo `models/final/modelo_final.pkl` fue generado con Python 3.11 y sklearn de la versión instalada en el entorno `ai-miadas`. Para cargarlo correctamente se debe usar el mismo entorno. PKLs generados en otras versiones de sklearn pueden presentar incompatibilidades.

---

## 8. Artefactos Generados — Inventario Completo

### Notebooks (en orden de ejecución)

| Notebook | Tarea | Estado |
|---|---|---|
| `00_exploracion_benchmark.ipynb` | Exploración inicial (referencia del colega) | Referencia |
| `01_benchmark_modelos.ipynb` | S3-DS-01/02: Benchmark + selección finalistas | ✅ Ejecutado |
| `02_hyperparameter_tuning.ipynb` | S3-DS-03/04: HPO Optuna 50 trials | ✅ Ejecutado |
| `03_interpretacion_feature_audit.ipynb` | Interpretación de feature importance | ✅ Ejecutado |
| `04_evaluacion_modelo_final.ipynb` | Evaluación y exportación del pkl final | ✅ Ejecutado |

### Modelos serializados

| Archivo | Descripción | Uso |
|---|---|---|
| `models/baseline/lightgbm_baseline.pkl` | LightGBM sin tuning | Comparación |
| `models/baseline/xgboost_baseline.pkl` | XGBoost sin tuning | Comparación |
| `models/final/lightgbm_tuned.pkl` | LightGBM tuneado (Optuna) | Finalista |
| `models/final/xgboost_tuned.pkl` | XGBoost tuneado (Optuna) | Runner-up |
| **`models/final/modelo_final.pkl`** | **Pipeline completo — modelo de producción** | **✅ Entregable principal** |

### Datos procesados

| Archivo | Descripción |
|---|---|
| `data/processed/09_model_benchmark.json` | Métricas benchmark de 5 modelos (S3-DS-01) |
| `data/processed/10_tuning_results.json` | Métricas y parámetros óptimos del HPO |
| `data/processed/11_lightgbm_baseline_importance.parquet` | Importancias LightGBM baseline |
| `data/processed/12_lightgbm_tuned_importance.parquet` | Importancias LightGBM tuneado |
| `data/processed/13_feature_audit_lightgbm.parquet` | Auditoría completa de features LightGBM |
| `data/processed/13_feature_audit_xgboost.parquet` | Auditoría completa de features XGBoost |

### Gráficos (`reports/figures/`)

| Archivo | Contenido |
|---|---|
| `sprint_03_benchmark_roc_auc.png` | Comparativa ROC-AUC train vs. val — 5 modelos baseline |
| `sprint_03_tuning_comparison.png` | Baseline vs. tuneado por métrica (ROC-AUC, F1, Recall) |
| `sprint_03_optimization_history.png` | Evolución de trials Optuna (LightGBM y XGBoost) |
| `sprint_03_roc_pr_curves.png` | Curvas ROC y PR de los 4 modelos (baseline + tuneado) |
| `sprint_03_feature_importance.png` | Top 20 features por modelo |
| `sprint_03_threshold_analysis.png` | Análisis de umbral del ganador (del notebook de tuning) |
| `sprint_03_confusion_matrix.png` | Matrices de confusión (umbral 0.50 vs. 0.55) |
| `sprint_03_final_roc_pr.png` | Curvas ROC y PR del modelo final (del notebook de evaluación) |
| `sprint_03_final_threshold.png` | Análisis de umbral del modelo final |
| `sprint_03_score_distribution.png` | Distribución de probabilidades por clase |

### Reportes markdown

| Archivo | Descripción |
|---|---|
| `reports/sprint_03/comparacion_modelos.md` | Benchmark completo con justificación técnica y de negocio |
| `reports/sprint_03/tuning_results.md` | Resultados del HPO con parámetros óptimos |
| `reports/sprint_03/informe_sprint_3.md` | **Este documento** |

---

## 9. Cumplimiento de Entregables Oficiales

| Entregable | Criterio (peso) | Estado | Evidencia |
|---|---|---|---|
| Modelo final `.pkl` | Reproducible (20%) | ✅ | `models/final/modelo_final.pkl` + verificación round-trip en nb 04 |
| Notebook comparativo | Documentación (20%) | ✅ | `02_hyperparameter_tuning.ipynb` (36 celdas ejecutadas) |
| Gráficos performance | Selección modelo (30%) | ✅ | 10 PNGs en `reports/figures/` |
| HPO con metodología | Optimización (30%) | ✅ | Optuna TPE, 50 trials, `10_tuning_results.json` |

---

## 10. Decisiones Técnicas Clave

| Decisión | Alternativa descartada | Justificación |
|---|---|---|
| **Ranking compuesto** para seleccionar finalistas (0.5×AUC + 0.3×Gini + 0.2×F1) | Ordenar solo por ROC-AUC | GradientBoosting tenía mejor AUC que XGBoost pero Recall=0.33 (inaceptable para el negocio) |
| **Optuna TPE** para HPO | GridSearchCV | TPE es 10-50× más eficiente en espacios continuos de hiperparámetros; 50 trials cubren el espacio mejor que una grilla |
| **LightGBM** como modelo ganador | XGBoost | Mejor ROC-AUC y Gini; velocidad superior para iteraciones futuras |
| **Umbral 0.55** en lugar de 0.50 | 0.50 (por defecto sklearn) | Maximiza F1 en el conjunto de validación; trade-off consciente entre Precision y Recall |
| **Split temporal** (no aleatorio) | Stratified K-Fold | Simula la condición real de producción: siempre se predice sobre clientes futuros |
| **Pipeline sklearn completo** | Solo modelo `.pkl` | Encapsula preprocesamiento; garantiza que en producción se apliquen exactamente las mismas transformaciones |

---

## 11. Próximos Pasos — Sprint 4

- **API de scoring (FastAPI):** Endpoint que recibe las features de un cliente y retorna `is_premium` + probabilidad usando el `modelo_final.pkl`.
- **Dashboard (Streamlit):** Visualización del segmento premium, métricas de negocio y predicciones en tiempo real.
- **Evaluación ética y de gobernanza:** Analizar si el modelo discrimina por geografía (`customer_state`, `region_group`) o método de pago.
- **Reentrenamiento mensual:** Script que, dado nuevos datos de transacciones, ejecuta el pipeline completo desde la ingesta hasta la exportación de un nuevo `.pkl`.

---

*Generado desde: `notebooks/sprint_03_modeling/01_benchmark_modelos.ipynb`, `02_hyperparameter_tuning.ipynb`, `03_interpretacion_feature_audit.ipynb`, `04_evaluacion_modelo_final.ipynb`*  
*Datos fuente: `data/processed/09_model_benchmark.json`, `data/processed/10_tuning_results.json`*
