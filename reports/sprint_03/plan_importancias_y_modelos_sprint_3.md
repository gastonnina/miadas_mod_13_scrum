# Plan de Importancias y Modelos - Sprint 3

## Objetivo

Ajustar el flujo de Sprint 3 para que deje trazabilidad completa entre:

- variables disponibles para modelado
- variables seleccionadas para entrenamiento
- importancia antes de hiperparametrizacion
- importancia despues de hiperparametrizacion
- artefactos `.pkl` persistidos para revision posterior

El criterio acordado es conservar ambos candidatos finalistas del Sprint 3:

- `LightGBM`
- `XGBoost`

Por lo tanto, el sprint debe terminar con:

- `2` modelos baseline serializados
- `2` modelos tuneados serializados
- tablas de importancia consultables en `csv` o `parquet`

## Decision metodologica

## Enfoque por fases

Para evitar bloquear el avance por falta de pipeline operativo, el Sprint 3 se ejecutara en dos fases.

### Fase 1: entrega minima desde notebooks

La primera fase prioriza producir los artefactos academicos y de defensa sin depender todavia de una implementacion formal por linea de comandos.

En esta fase se debe resolver primero:

- entrenamiento baseline de `LightGBM` y `XGBoost` desde notebook
- entrenamiento tuneado de `LightGBM` y `XGBoost` desde notebook
- persistencia de los modelos baseline en:
  - `models/baseline/lightgbm_baseline.pkl`
  - `models/baseline/xgboost_baseline.pkl`
- persistencia de los modelos tuneados en:
  - `models/final/lightgbm_tuned.pkl`
  - `models/final/xgboost_tuned.pkl`
- extraccion de importancias desde `feature_importances_`
- construccion de la tabla final por modelo
- export de tablas finales en `parquet`

#### Estructura esperada de la tabla en Fase 1

La estructura base acordada para la tabla es:

| Campo | Descripcion |
| --- | --- |
| `Nro` | correlativo |
| `Dominio` | tabla o capa de donde proviene el campo |
| `Variable` | nombre de la columna |
| `flagSelected` | `1` si entra al entrenamiento del modelo, `0` si no entra |
| `importancia_seleccion` | importancia del modelo baseline |
| `importancia_modelo_final` | importancia del modelo tuneado |

#### Ejemplo tipo docente

La tabla debe verse conceptualmente asi:

| Nro | Dominio | Variable | flagSelected | importancia_seleccion | importancia_modelo_final |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | Poblacion Objetivo | customer_unique_id | 0 | 0.00% | 0.00% |
| 2 | Target | is_premium | 0 | 0.00% | 0.00% |
| 3 | Order Items | total_items | 1 | 11.00% | 8.00% |
| 4 | Order Items | total_products | 1 | 8.76% | 5.00% |
| 5 | features_derivadas | products_per_order | 1 | 1.05% | 4.00% |

Notas:

- `customer_unique_id` e `is_premium` aparecen para que la tabla sea entendible en la defensa, aunque no entren al modelo
- las variables seleccionadas llevan `flagSelected = 1`
- las importancias se relacionan siempre con el nombre exacto de la columna usada en entrenamiento

#### Tabla base de Fase 1

La siguiente tabla base ya puede construirse en la Fase 1 usando las columnas reales de `selected_model_columns` y las dos variables de contexto.

| Nro | Dominio | Variable | flagSelected | importancia_seleccion | importancia_modelo_final |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | customers | customer_unique_id | 0 | 0.00% | 0.00% |
| 2 | target | is_premium | 0 | 0.00% | 0.00% |
| 3 | orders | total_orders | 1 | 0.00% | 0.00% |
| 4 | order_items_products | total_items | 1 | 0.00% | 0.00% |
| 5 | order_items_products | total_products | 1 | 0.00% | 0.00% |
| 6 | order_reviews | avg_review_score | 1 | 0.00% | 0.00% |
| 7 | orders | avg_delivery_days | 1 | 0.00% | 0.00% |
| 8 | orders | avg_estimated_delivery_days | 1 | 0.00% | 0.00% |
| 9 | orders | delivered_orders | 1 | 0.00% | 0.00% |
| 10 | orders | late_deliveries | 1 | 0.00% | 0.00% |
| 11 | order_payments | payment_methods_count | 1 | 0.00% | 0.00% |
| 12 | order_payments | max_payment_installments | 1 | 0.00% | 0.00% |
| 13 | orders | recency_days | 1 | 0.00% | 0.00% |
| 14 | orders | customer_lifetime_days | 1 | 0.00% | 0.00% |
| 15 | orders | cancellation_rate | 1 | 0.00% | 0.00% |
| 16 | features_derivadas | products_per_order | 1 | 0.00% | 0.00% |
| 17 | features_derivadas | max_to_avg_price_ratio | 1 | 0.00% | 0.00% |
| 18 | features_derivadas | installments_gt_1_flag | 1 | 0.00% | 0.00% |
| 19 | features_derivadas | installments_gt_6_flag | 1 | 0.00% | 0.00% |
| 20 | features_derivadas | credit_card_flag | 1 | 0.00% | 0.00% |
| 21 | features_derivadas | voucher_flag | 1 | 0.00% | 0.00% |
| 22 | features_derivadas | delivery_gap | 1 | 0.00% | 0.00% |
| 23 | order_reviews | reviews_per_order | 1 | 0.00% | 0.00% |
| 24 | features_derivadas | far_region_flag | 1 | 0.00% | 0.00% |
| 25 | features_derivadas | top_category_is_high_value | 1 | 0.00% | 0.00% |
| 26 | customers | customer_state | 1 | 0.00% | 0.00% |
| 27 | order_payments | main_payment_type | 1 | 0.00% | 0.00% |
| 28 | order_items_products | top_category | 1 | 0.00% | 0.00% |
| 29 | features_derivadas | region_group | 1 | 0.00% | 0.00% |
| 30 | features_derivadas | top_category_group | 1 | 0.00% | 0.00% |

Uso en Fase 1:

- esta tabla se puede inicializar con importancias en `0.00%`
- luego se reemplazan `importancia_seleccion` e `importancia_modelo_final` con los valores reales extraidos desde cada notebook
- debe generarse una version para `LightGBM` y otra para `XGBoost`

La fase 1 usa principalmente:

- `notebooks/sprint_03_modeling/01_benchmark_modelos.ipynb`
- `notebooks/sprint_03_modeling/02_hyperparameter_tuning.ipynb`

Esta es la fase que debe correrse primero.

### Fase 2: formalizacion en pipeline reproducible

La segunda fase convierte la logica ya validada en notebooks en un flujo reproducible por linea de comandos.

En esta fase se debe resolver:

- entrenamiento reproducible fuera del notebook
- scoring sobre `holdout`
- evaluacion final automatizada
- generacion de reportes y artefactos operativos

La fase 2 no bloquea la ejecucion inicial de Sprint 3. Su objetivo es dejar la solucion operativa y repetible despues de validar la logica en notebook.

### Modelos baseline

Se guardaran los modelos base antes del tuning:

- `models/baseline/lightgbm_baseline.pkl`
- `models/baseline/xgboost_baseline.pkl`

Estos modelos representan el estado previo a la hiperparametrizacion y son la fuente oficial para la columna:

- `importancia_seleccion`

### Modelos tuneados

Se guardaran los modelos optimizados luego del tuning:

- `models/final/lightgbm_tuned.pkl`
- `models/final/xgboost_tuned.pkl`

Estos modelos representan el estado posterior a la hiperparametrizacion y son la fuente oficial para la columna:

- `importancia_modelo_final`

### Esquema de validacion acordado

No se utilizara:

- `5-fold cross-validation`
- `KFold`
- `StratifiedKFold`
- validacion cruzada aleatoria clasica

La validacion oficial del proyecto se mantiene temporal y consistente con Sprint 2 y Sprint 3:

- `train`: `last_purchase < 2018-07-01`
- `validation`: `last_purchase >= 2018-07-01`

La logica operativa queda asi:

- en `dev`, los modelos se entrenan sobre `train`
- en `dev`, los modelos se comparan y tunean sobre `validation`
- en `holdout`, solo se realiza evaluacion final del modelo ya seleccionado

La razon para no usar cross-validation clasica es evitar leakage temporal y no mezclar pasado con futuro en un problema donde el orden cronologico si importa.

## Alcance de la tabla de variables

La tabla final no necesita incluir todo el universo de columnas de `data/processed/05_features_rfm.parquet`.

Para la entrega y la defensa se utilizara una version mas simple y directa:

- variables que si entraron al modelo, tomadas desde `selected_model_columns`
- `customer_unique_id` como identificador de la poblacion objetivo
- `is_premium` como variable objetivo

La fuente oficial de seleccion se mantiene en:

- `data/processed/06_features_selected_metadata.json`

Por lo tanto, cada fila de la tabla final representara:

- una variable seleccionada para entrenamiento
- o una variable de contexto obligatoria para lectura del docente (`customer_unique_id`, `is_premium`)

## Manejo de multiples modelos

Como ahora se conservaran dos familias de modelo, no conviene mezclar ambas en una sola tabla con una unica columna de importancia, porque eso perderia trazabilidad.

La recomendacion operativa es generar una tabla por modelo:

### LightGBM

- `data/processed/11_feature_audit_lightgbm.parquet`

Columnas:

- `Nro`
- `Dominio`
- `Variable`
- `flagSelected`
- `importancia_seleccion`
- `importancia_modelo_final`
- `modelo = lightgbm`

### XGBoost

- `data/processed/11_feature_audit_xgboost.parquet`

Columnas:

- `Nro`
- `Dominio`
- `Variable`
- `flagSelected`
- `importancia_seleccion`
- `importancia_modelo_final`
- `modelo = xgboost`

## Regla para `flagSelected`

La seleccion oficial debe salir de:

- `data/processed/06_features_selected_metadata.json`

Regla:

- `1` si la variable esta en `selected_model_columns`
- `0` si la variable es solo de contexto (`customer_unique_id`, `is_premium`)

Esto implica que en esta tabla ya no se listan todas las columnas descartadas del universo original.

## Regla para importancias

### `importancia_seleccion`

Debe obtenerse desde `feature_importances_` del `.pkl` baseline del modelo correspondiente:

- `lightgbm_baseline.pkl` para la tabla de LightGBM
- `xgboost_baseline.pkl` para la tabla de XGBoost

Operacion esperada:

- leer `feature_importances_`
- relacionar cada valor con su nombre de columna en el mismo orden de entrenamiento
- normalizar a porcentaje sobre la suma total de importancias
- guardar el resultado en la columna `importancia_seleccion`

### `importancia_modelo_final`

Debe obtenerse desde `feature_importances_` del `.pkl` tuneado del modelo correspondiente:

- `lightgbm_tuned.pkl` para la tabla de LightGBM
- `xgboost_tuned.pkl` para la tabla de XGBoost

Operacion esperada:

- leer `feature_importances_`
- relacionar cada valor con su nombre de columna en el mismo orden de entrenamiento
- normalizar a porcentaje sobre la suma total de importancias
- guardar el resultado en la columna `importancia_modelo_final`

### Variables no entrenadas

Las variables de contexto que no entran al entrenamiento deben tener:

- `flagSelected = 0`
- `importancia_seleccion = 0`
- `importancia_modelo_final = 0`

Esto aplica a:

- `customer_unique_id`
- `is_premium`

## Dominio de las variables

El campo `Dominio` debe salir del pipeline existente.

### Dominios base

- `customers`
- `orders`
- `order_payments`
- `order_reviews`
- `order_items_products`
- `features_derivadas`
- `target`

### Asignacion preliminar

- `customer_unique_id`, `customer_zip_code_prefix`, `customer_city`, `customer_state` -> `customers`
- `total_orders`, `first_purchase`, `last_purchase`, `avg_delivery_days`, `avg_estimated_delivery_days`, `delivered_orders`, `canceled_orders`, `late_deliveries`, `recency_days`, `customer_lifetime_days`, `cancellation_rate`, `late_delivery_rate` -> `orders`
- `payment_methods_count`, `max_payment_installments`, `avg_payment_installments`, `total_spent`, `avg_ticket`, `avg_order_price`, `avg_freight_value`, `avg_freight_ratio`, `main_payment_type` -> `order_payments`
- `avg_review_score`, `total_reviews`, `reviews_per_order` -> `order_reviews`
- `total_items`, `total_products`, `max_item_price`, `avg_item_price`, `top_category` -> `order_items_products`
- `items_per_order`, `products_per_order`, `max_to_avg_price_ratio`, `freight_to_item_ratio`, `installments_gt_1_flag`, `installments_gt_6_flag`, `credit_card_flag`, `boleto_flag`, `voucher_flag`, `payment_complexity_flag`, `has_late_delivery`, `has_cancellation`, `delivery_gap`, `region_group`, `far_region_flag`, `top_category_group`, `top_category_is_high_value` -> `features_derivadas`
- `is_premium` -> `target`

## Notebooks a modificar

Los notebooks quedan reservados para trabajo analitico y de defensa, no como mecanismo principal de ejecucion operativa del modelo final.

En la practica, para este sprint se usaran primero como mecanismo de ejecucion de la Fase 1.

### `notebooks/sprint_03_modeling/01_benchmark_modelos.ipynb`

Debe ajustarse para:

- comparar candidatos de modelo
- entrenar y persistir `LightGBM baseline`
- entrenar y persistir `XGBoost baseline`
- exportar sus importancias baseline
- registrar ruta del `.pkl` y artefactos asociados

Este notebook si debe contener:

- benchmark comparativo
- analisis de metricas
- justificacion tecnica de seleccion

Este notebook no debe ser el flujo oficial para:

- entrenamiento productivo final
- scoring recurrente de nueva data
- evaluacion operacional sobre `holdout`

### `notebooks/sprint_03_modeling/02_hyperparameter_tuning.ipynb`

Debe ajustarse para:

- comparar tuning entre candidatos
- persistir `LightGBM tuned`
- persistir `XGBoost tuned`
- exportar importancias post-tuning de ambos modelos
- dejar referencia clara al modelo ganador oficial del sprint

Este notebook si debe contener:

- exploracion de hiperparametros
- comparacion baseline vs tuneado
- graficos de performance e importancia
- seleccion del modelo ganador

Este notebook no debe ser el mecanismo principal para:

- reentrenar el modelo oficial cada vez
- correr evaluacion repetible sobre `holdout`
- servir inferencia sobre nueva data

## Frontera entre notebook y pipeline

La separacion notebook vs pipeline aplica de forma gradual:

- en Fase 1, el notebook si se usa para producir los artefactos iniciales del sprint
- en Fase 2, esa logica debe migrarse a un flujo reproducible por comandos

### Lo que queda en notebook

- comparacion de modelos candidatos
- benchmark inicial
- tuning exploratorio
- graficos y narrativa de defensa
- decision metodologica de que modelo y parametros ganan

### Lo que queda en pipeline

- entrenamiento reproducible del modelo ya elegido
- validacion temporal sobre `dev`
- guardado del `.pkl` oficial
- export de metricas, importancias y reportes
- carga del `.pkl` para scoring o evaluacion en `holdout`

### Lo que no debe mezclarse

No conviene que el pipeline operativo:

- vuelva a comparar todos los modelos candidatos
- vuelva a ejecutar benchmark exploratorio
- dependa del notebook para producir el artefacto final

No conviene que el notebook:

- sea la unica forma de generar el modelo oficial
- sea la unica forma de evaluar `holdout`
- concentre la logica operativa de entrenamiento y scoring

## Ajustes requeridos en pipeline

Esta seccion corresponde a la Fase 2 del plan.

Ademas de los notebooks, el repositorio debe quedar con un flujo reproducible por linea de comandos para no depender solo de ejecucion manual en notebook.

### Objetivo del pipeline

El pipeline debe soportar al menos dos capacidades:

- entrenar el modelo ya elegido, en version baseline o tuneada, sobre `dev`
- aplicar o evaluar un modelo entrenado sobre nueva data, por ejemplo `holdout`

### Comando de entrenamiento

Debe existir un flujo equivalente a:

```bash
train-model \
  --model-type lightgbm|xgboost \
  --stage baseline|tuned \
  --train-split dev \
  --validation-cutoff 2018-07-01 \
  --params-json <ruta_opcional> \
  --output-model <ruta_pkl> \
  --output-metrics <ruta_json> \
  --output-importance <ruta_parquet> \
  --output-report <ruta_md>
```

Responsabilidades de este flujo:

- cargar `data/processed/06_features_selected.parquet`
- reconstruir el split temporal oficial de `dev`
- entrenar sobre `dev-train`
- evaluar sobre `dev-validation`
- persistir `.pkl`
- persistir metricas
- persistir importancia de variables
- persistir un reporte resumido en `md`

### Salidas minimas del entrenamiento

Cada ejecucion de entrenamiento debe producir como minimo:

- un modelo serializado en `.pkl`
- una tabla de importancia en `.parquet`
- un archivo de metricas en `json`
- un reporte ejecutivo en `md`

### Comando de aplicacion o evaluacion sobre nueva data

Debe existir un flujo equivalente a:

```bash
score-model \
  --model-path <ruta_pkl> \
  --input-path <dataset_parquet> \
  --dataset-name holdout \
  --output-predictions <ruta_parquet> \
  --output-metrics <ruta_json> \
  --output-report <ruta_md>
```

Responsabilidades de este flujo:

- cargar un modelo ya entrenado
- recibir un dataset nuevo con la misma estructura de features
- generar predicciones o scores
- si existe target, calcular metricas
- guardar resultados para revision posterior
- generar un reporte de resultados en `md`

### Salidas minimas de scoring o evaluacion

Cada ejecucion sobre nueva data debe producir como minimo:

- predicciones o scores en `.parquet`
- metricas en `json`, si el dataset incluye target
- un reporte resumido en `md`

### Caso especifico `holdout`

Sprint 2 ya deja disponible el dataset:

- `data/processed/holdout_features_rfm.parquet`

Ese artefacto sale del flujo documentado en `scripts/build_features.sh` y de la simulacion temporal agosto-octubre 2018.

Para Sprint 3, la logica sobre `holdout` debe ser:

- no volver a tunear
- no volver a seleccionar features
- no recalcular hiperparametros
- usar el modelo tuneado ya definido en `dev`
- reutilizar `data/processed/holdout_features_rfm.parquet` como entrada de scoring
- ejecutar solo inferencia y evaluacion final

Si el dataset `holdout` ya existe y no hubo cambio de fuente, no hace falta reconstruirlo.

### Scripts esperados

La implementacion puede quedar en:

- `src/models/train_model.py`
- `src/models/predict.py`
- `src/models/evaluate_model.py`
- `scripts/train_model.sh`
- `scripts/score_model.sh`

No es obligatorio que los nombres sean exactamente esos, pero el plan si exige que exista esa separacion funcional:

- entrenamiento reproducible
- scoring o inferencia sobre nueva data
- evaluacion final sobre `holdout`

## Artefactos nuevos recomendados

### Modelos

- `models/baseline/lightgbm_baseline.pkl`
- `models/baseline/xgboost_baseline.pkl`
- `models/final/lightgbm_tuned.pkl`
- `models/final/xgboost_tuned.pkl`

### Importancias intermedias

- `data/processed/11_lightgbm_baseline_importance.parquet`
- `data/processed/11_xgboost_baseline_importance.parquet`
- `data/processed/12_lightgbm_tuned_importance.parquet`
- `data/processed/12_xgboost_tuned_importance.parquet`

### Reportes de entrenamiento

- `reports/sprint_03/lightgbm_baseline_training_report.md`
- `reports/sprint_03/xgboost_baseline_training_report.md`
- `reports/sprint_03/lightgbm_tuned_training_report.md`
- `reports/sprint_03/xgboost_tuned_training_report.md`

### Predicciones y evaluacion sobre holdout

- `data/processed/14_holdout_lightgbm_predictions.parquet`
- `data/processed/14_holdout_xgboost_predictions.parquet`
- `data/processed/14_holdout_lightgbm_metrics.json`
- `data/processed/14_holdout_xgboost_metrics.json`
- `reports/sprint_03/holdout_lightgbm_report.md`
- `reports/sprint_03/holdout_xgboost_report.md`

### Tablas finales para consulta

- `data/processed/13_feature_audit_lightgbm.parquet`
- `data/processed/13_feature_audit_xgboost.parquet`

## Secuencia de trabajo

```text
FASE 1
1. Confirmar lista oficial de variables seleccionadas desde Sprint 2
2. Entrenar baseline de LightGBM y XGBoost en `dev-train` desde notebook
3. Evaluar baseline de LightGBM y XGBoost en `dev-validation`
4. Guardar PKL baseline de ambos modelos
5. Extraer importancia baseline de ambos modelos
6. Ejecutar tuning de LightGBM y XGBoost usando `dev-train` y `dev-validation`
7. Guardar PKL tuneado de ambos modelos
8. Extraer importancia tuneada de ambos modelos
9. Construir tabla final por modelo desde notebook
10. Exportar tablas finales en parquet y csv
11. Elegir y documentar el modelo ganador oficial del sprint

FASE 2
12. Implementar flujo reproducible de entrenamiento fuera de notebook
13. Implementar scoring sobre `holdout`
14. Evaluar en `holdout` el modelo ya tuneado y seleccionado
15. Generar reportes operativos finales
```

## Criterios de cierre

El trabajo se considera completo cuando:

- la Fase 1 puede correrse completa desde notebooks
- existen `4` archivos `.pkl` persistidos
- existen tablas de importancia baseline y tuneada para ambos modelos
- existen tablas finales consultables para `LightGBM` y `XGBoost`
- existen salidas tabulares en `parquet` y `csv` para las tablas finales
- el modelo ganador del sprint queda identificado de forma explicita

Si se completa tambien la Fase 2, adicionalmente:

- existen salidas tabulares en `parquet` para importancias y predicciones
- existen reportes en `md` para entrenamiento y evaluacion final
- existen tablas finales consultables para `LightGBM` y `XGBoost`
- el flujo puede ejecutarse sin depender del notebook

## Recomendacion final

Para la defensa con el docente conviene mostrar:

- una tabla completa por modelo
- y ademas una conclusion ejecutiva donde se indique cual de los dos queda como modelo oficial final

Asi se conserva trazabilidad tecnica total sin perder claridad en la entrega principal.
