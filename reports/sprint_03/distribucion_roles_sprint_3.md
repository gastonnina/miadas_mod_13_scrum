# Distribucion de Roles - Sprint 3

## Objetivo del sprint

Seleccionar modelo, optimizar hiperparametros y exportar modelo final reproducible para clasificacion de clientes premium.

## Entregables oficiales

- modelo final serializado (`.pkl`)
- notebook comparativo de tuning y metricas
- graficos de performance y feature importance

## Logica de division

Para repartir el Sprint 3 entre 3 personas sin pisarse, conviene dividir por capas de trabajo:

1. benchmarking y seleccion de candidatos
2. tuning y entrenamiento final
3. notebook, visualizacion y narrativa de defensa

Esa division mantiene un flujo secuencial claro, pero tambien permite trabajo paralelo en documentacion, visualizacion y validacion.

## Propuesta por roles

### Persona 1 - Responsable de benchmark y seleccion de modelo

**Rol sugerido**

Lider de modelado comparativo.

**Responsabilidades**

- tomar como insumo `data/processed/06_features_selected.parquet`
- correr benchmark inicial de modelos candidatos
- comparar al menos:
  - `LogisticRegression`
  - `RandomForestClassifier`
  - `GradientBoosting` o `XGBoost` si esta disponible
- mantener el mismo split temporal oficial:
  - `train`: `last_purchase < 2018-07-01`
  - `validation`: `last_purchase >= 2018-07-01`
- definir cual o cuales modelos pasan a tuning

**Entregables a su cargo**

- tabla comparativa de modelos
- archivo resumen con metricas por modelo
- recomendacion tecnica del modelo finalista

**Artefactos esperados**

- `reports/sprint_03/comparacion_modelos.md`
- `data/processed/09_model_benchmark.json`

**Criterio de cierre**

Deja una recomendacion clara del mejor candidato basada principalmente en `ROC-AUC`, `Gini`, `F1` y estabilidad entre train/validation.

### Persona 2 - Responsable de tuning y export del modelo final

**Rol sugerido**

Lider de entrenamiento final y reproducibilidad.

**Responsabilidades**

- tomar el modelo finalista definido por la Persona 1
- ejecutar tuning reproducible con semilla fija
- definir grilla o espacio de hiperparametros
- comparar baseline del modelo vs mejor configuracion
- entrenar el modelo ganador final
- serializar pipeline completo de preprocesamiento + modelo
- validar que el `.pkl` cargue correctamente y pueda predecir

**Entregables a su cargo**

- mejor configuracion del modelo
- modelo final exportado
- metricas finales del modelo ganador

**Artefactos esperados**

- `data/processed/10_tuning_results.parquet`
- `data/processed/10_best_model_config.json`
- `models/modelo_final_sprint_3.pkl`
- `data/processed/11_final_model_metrics.json`

**Criterio de cierre**

El artefacto final se puede cargar sin errores, reproduce metricas coherentes con el tuning y queda listo para integracion posterior.

### Persona 3 - Responsable de notebook, graficos y defensa

**Rol sugerido**

Lider de analisis, visualizacion y documentacion.

**Responsabilidades**

- construir el notebook comparativo del Sprint 3
- integrar resultados del benchmark y del tuning
- generar graficos de:
  - curva ROC
  - matriz de confusion
  - comparacion de metricas
  - feature importance
- redactar la narrativa tecnica del sprint
- asegurar que lo mostrado en notebook coincida con los artefactos finales

**Entregables a su cargo**

- notebook final del sprint
- graficos de performance
- resumen ejecutivo de defensa

**Artefactos esperados**

- `notebooks/sprint_03_modeling/04_hyperparameter_tuning.ipynb`
- `reports/sprint_03/modelo_final_resumen.md`
- figuras exportadas en `reports/figures/`

**Criterio de cierre**

Queda lista la evidencia visual y narrativa para exposicion, con trazabilidad completa hacia el modelo final exportado.

## Dependencias entre roles

La secuencia recomendada es:

```text
Persona 1: benchmark y seleccion
-> Persona 2: tuning y modelo final
-> Persona 3: notebook final y defensa
```

Pero hay trabajo paralelo posible:

- Persona 3 puede ir armando estructura del notebook mientras Persona 1 y 2 generan resultados
- Persona 2 puede preparar el script de tuning antes de recibir el modelo finalista

## Reparto equilibrado por carga

Si quieren un reparto practico y balanceado:

- Persona 1: decision tecnica del algoritmo
- Persona 2: implementacion reproducible del modelo final
- Persona 3: presentacion analitica y defensa

Asi cada integrante tiene:

- una responsabilidad clara
- un entregable visible
- una parte defendible durante la exposicion

## Como defender el trabajo de cada uno

### Persona 1 puede decir

"Yo me encargue de la comparacion objetiva entre modelos candidatos usando el mismo split temporal del Sprint 2, para seleccionar el algoritmo mas conveniente con base en AUC, Gini, F1 y estabilidad."

### Persona 2 puede decir

"Yo me encargue de la hiperparametrizacion del modelo finalista, del entrenamiento final y de exportar un artefacto reproducible en formato `.pkl` listo para reutilizacion."

### Persona 3 puede decir

"Yo consolide los resultados en el notebook comparativo, genere los graficos de performance y feature importance, y prepare la narrativa tecnica para la defensa del Sprint 3."

## Opcion con nombres del equipo

Si quieren repartirlo con los integrantes actuales del repo:

- `Gaston Nina`: tuning y export del modelo final
- `Gerick Toro`: benchmark y seleccion de candidatos
- `Marcelo De la Quintana`: notebook, graficos y defensa

Esta asignacion es solo sugerida. Pueden rotarla segun quien tenga mas comodidad con modelado, codigo o presentacion.
