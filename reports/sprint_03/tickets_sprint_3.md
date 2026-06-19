# Tickets Propuestos - Sprint 3

## Logica heredada de Sprint 2

En Sprint 2 el trabajo se organizo como una secuencia de etapas reproducibles:

1. un ticket principal con objetivo y entregables del sprint
2. subtareas tecnicas alineadas al pipeline real
3. cada subtarea produce artefactos verificables
4. el cierre del sprint consolida metricas, notebook y documentacion

Aplicando la misma logica, Sprint 3 debe separarse en tickets que sigan el flujo:

`benchmark de modelos -> tuning reproducible -> export del modelo final -> notebook y graficos de defensa`

## Ticket Principal

### Tarea 3 - Sprint 3: Hiperparametrizacion y modelo final

- Abre: `miercoles, 10 de junio de 2026, 00:00`
- Pendiente: `sabado, 13 de junio de 2026, 12:00`

**Enunciado**

Seleccionar modelo, optimizar hiperparametros y exportar modelo final reproducible para clasificacion de clientes premium.

**Entregables**

- modelo final serializado (`.pkl` o `.joblib`)
- notebook comparativo de tuning y metricas
- graficos de performance y feature importance
- documentacion breve del modelo ganador y sus parametros

**Criterios de evaluacion**

- seleccion del modelo: `30%`
- optimizacion e hiperparametrizacion: `30%`
- documentacion clara de resultados: `20%`
- modelo final reproducible: `20%`

## Subtickets tecnicos propuestos

### Ticket 3.1 - Benchmark inicial de modelos candidatos

- Abre: `miercoles, 10 de junio de 2026, 00:00`
- Pendiente: `miercoles, 10 de junio de 2026, 23:59`

**Objetivo**

Comparar varios algoritmos candidatos sobre el mismo set oficial de Sprint 2 para elegir cuales pasan a tuning.

**Entrada oficial**

- `data/processed/06_features_selected.parquet`
- cutoff temporal oficial: `2018-07-01`

**Trabajo esperado**

- entrenar al menos `3` modelos comparables
- usar el mismo split temporal de Sprint 2
- reportar `ROC-AUC`, `Gini`, `F1`, `precision` y `recall`
- dejar una tabla comparativa reproducible

**Modelos sugeridos**

- `RandomForestClassifier`
- `XGBoost` o `GradientBoosting`
- `LogisticRegression` como referencia

**Artefactos esperados**

- `reports/sprint_03/comparacion_modelos.md`
- `data/processed/09_model_benchmark.json`

**Criterio de cierre**

Quedan seleccionados `1` o `2` modelos finalistas para tuning con justificacion basada en AUC y estabilidad.

### Ticket 3.2 - Tuning reproducible del modelo finalista

- Abre: `jueves, 11 de junio de 2026, 00:00`
- Pendiente: `jueves, 11 de junio de 2026, 23:59`

**Objetivo**

Optimizar hiperparametros del modelo finalista sin romper la logica temporal ni introducir leakage.

**Trabajo esperado**

- definir grilla o espacio de busqueda
- correr tuning con semilla fija
- comparar baseline del modelo vs mejor configuracion
- guardar parametros ganadores y metricas

**Artefactos esperados**

- `data/processed/10_tuning_results.parquet`
- `data/processed/10_best_model_config.json`
- `reports/sprint_03/tuning_summary.md`

**Criterio de cierre**

Existe una configuracion ganadora con mejora defendible frente al benchmark inicial.

### Ticket 3.3 - Entrenamiento final y export del modelo

- Abre: `viernes, 12 de junio de 2026, 00:00`
- Pendiente: `viernes, 12 de junio de 2026, 18:00`

**Objetivo**

Reentrenar el modelo ganador con la configuracion final y exportarlo como artefacto reutilizable.

**Trabajo esperado**

- entrenar con el set oficial seleccionado
- serializar pipeline completo de preprocesamiento + modelo
- verificar que el artefacto pueda cargarse y predecir

**Artefactos esperados**

- `models/modelo_final_sprint_3.pkl`
- `data/processed/11_final_model_metrics.json`
- `data/processed/11_feature_importance.parquet`

**Criterio de cierre**

El modelo exportado se puede cargar sin errores y reproduce metricas coherentes con el tuning.

### Ticket 3.4 - Notebook comparativo y graficos de defensa

- Abre: `viernes, 12 de junio de 2026, 18:00`
- Pendiente: `sabado, 13 de junio de 2026, 12:00`

**Objetivo**

Consolidar la narrativa analitica del Sprint 3 con comparacion de tuning, metricas finales y visualizaciones.

**Trabajo esperado**

- construir notebook final del sprint
- mostrar tabla benchmark vs modelo tuned
- incluir curva ROC, matriz de confusion y feature importance
- explicar por que se eligio el modelo ganador

**Artefactos esperados**

- `notebooks/sprint_03_modeling/04_hyperparameter_tuning.ipynb`
- `reports/sprint_03/modelo_final_resumen.md`
- figuras exportadas en `reports/figures/`

**Criterio de cierre**

La defensa del Sprint 3 queda soportada por notebook, tablas y graficos consistentes con el artefacto final.

## Secuencia recomendada

```text
3.1 Benchmark de modelos
-> 3.2 Tuning del finalista
-> 3.3 Export del modelo final
-> 3.4 Notebook y graficos de defensa
```

## Mensaje tecnico del Sprint 3

Si Sprint 2 respondio "que variables sirven y que tan bien separa el modelo", Sprint 3 debe responder:

- que algoritmo final conviene usar
- con que hiperparametros
- con que metricas finales reproducibles
- y con que artefacto exportable listo para integracion
