# Material para Google NotebookLM - Sprint 4

## Archivos recomendados para subir

### Contexto general del proyecto

- `README.md`
- `docs/plan_de_trabajo.md`

### Recapitulación de sprints anteriores

- `reports/sprint_02/informes/sprint_02_reporte.pdf`
- `reports/sprint_03/informes/sprint_03_reporte.pdf`
- `reports/sprint_03/informe_sprint_3.md`

### Sprint 4 - núcleo técnico y de negocio

- `reports/sprint_04/plan_sprint_4.md`
- `reports/sprint_04/reporte_cientifico_datos_sprint_4.md`
- `reports/sprint_04/informe_etico_gobernanza.md`
- `reports/sprint_04/pitch_demo_day.md`
- `reports/sprint_04/checklist_cierre_final.md`
- `reports/sprint_04/guia_ejecucion_mvp.md`

### Activos visuales clave

- `reports/figures/sprint_04_mvp_architecture.png`
- `reports/figures/sprint_04_holdout_evaluation_roi.png`
- `reports/figures/sprint_04_shap_beeswarm.png`
- `reports/figures/sprint_03_feature_importance.png`
- `reports/figures/sprint_03_confusion_matrix.png`
- `reports/figures/sprint_03_final_roc_pr.png`

### Soporte metodológico

- `notebooks/sprint_04_integration/04_demo_validation.ipynb`
- `reports/sprint_04/informes/sprint_04_reporte.tex`

## Recomendación de carga

Si NotebookLM empieza a mezclar demasiada información, cargar primero:

1. `README.md`
2. `reports/sprint_03/informe_sprint_3.md`
3. `reports/sprint_04/reporte_cientifico_datos_sprint_4.md`
4. `reports/sprint_04/informe_etico_gobernanza.md`
5. `reports/sprint_04/pitch_demo_day.md`
6. `reports/figures/sprint_04_holdout_evaluation_roi.png`
7. `reports/figures/sprint_04_shap_beeswarm.png`
8. `reports/figures/sprint_04_mvp_architecture.png`

## Prompts sugeridos para generar storytelling de 15 minutos

### Prompt 1 - estructura completa con carátula

```text
Actúa como asesor de storytelling para una defensa académica de 15 minutos sobre un proyecto de machine learning aplicado a negocio.

Quiero un guion y estructura de diapositivas en español para una presentación de 15 minutos con carátula incluida.

El proyecto trata sobre identificación de clientes premium en Olist mediante un pipeline reproducible, un modelo LightGBM final, validación backtest, dashboard Streamlit, API mínima, explicabilidad SHAP y análisis de ROI.

Necesito que la presentación:
1. incluya carátula
2. haga una recapitulación rápida y clara de los Sprint 1, 2 y 3
3. explique que Sprint 3 es la validación metodológica principal del modelo
4. explique que el foco principal es el Sprint 4 como integración operativa del MVP
5. evite presentar el ROC-AUC backtest como mejora milagrosa frente a Sprint 3
6. muestre que el dashboard separa scoring individual, validación Sprint 3 y backtest/ROI
4. conecte siempre problema -> solución -> evidencia -> impacto -> gobernanza
7. reserve un bloque para demo
8. cierre con conclusiones y siguientes pasos

Entrégame:
- número sugerido de diapositivas
- título de cada diapositiva
- objetivo de cada diapositiva
- mensaje clave de cada diapositiva
- transición verbal entre diapositivas
- recomendación de quién debería presentar cada bloque (DE, DS, DPO)
```

### Prompt 2 - versión más ejecutiva

```text
Genera un storytelling ejecutivo de 15 minutos para jurado académico no totalmente técnico.

Quiero una narrativa clara que explique:
- cuál era el problema de negocio original
- qué se hizo en Sprint 1, 2 y 3 de forma breve
- por qué Sprint 3 sigue siendo la referencia metodológica del modelo
- por qué Sprint 4 es la integración final del MVP
- cómo funciona el MVP
- qué significa el impacto de ROI
- cómo defendemos la explicabilidad SHAP
- por qué el ROC-AUC backtest alto no debe venderse de forma ingenua
- qué riesgos éticos y de gobernanza reconocemos
- cómo explicar que las métricas de Sprint 3 y las de backtest responden a escenarios distintos

Dame el resultado en formato:
- diapositiva
- título
- texto breve sugerido
- idea visual sugerida
- frase oral sugerida del presentador
```

### Prompt 3 - guion oral literal

```text
Redáctame un guion oral de 15 minutos en español, con tono profesional y académico, para tres presentadores.

Condiciones:
- incluir una carátula inicial
- resumir Sprint 1, Sprint 2 y Sprint 3 en menos de 3 minutos en total
- dedicar el resto al Sprint 4
- incluir explicación del dashboard, API, tabs del dashboard, SHAP beeswarm, ROI y gobernanza
- incluir una forma defendible de explicar por qué el ROC-AUC backtest es 0.9876 sin decir algo engañoso
- cerrar con una conclusión fuerte orientada a valor de negocio y madurez técnica

Quiero que el guion venga dividido por presentador:
- Presentador 1
- Presentador 2
- Presentador 3
```

### Prompt 4 - prompts para crear diapositivas bonitas

```text
Usa el material del proyecto para proponer una presentación visualmente sobria, académica y moderna.

Quiero una estructura de diapositivas para 15 minutos con:
- carátula
- recapitulación rápida de sprints 1 a 3
- objetivo general del proyecto
- una diapositiva corta que diferencie Sprint 3 (validación del modelo) de Sprint 4 (operación del MVP)
- bloque central del sprint 4
- demo
- SHAP
- ROI
- gobernanza
- cierre

Sugiere para cada diapositiva:
- tipo de visual
- poco texto
- frase principal
- qué archivo del proyecto usar como evidencia visual
```

## Prompt adicional para evitar errores con el AUC holdout

```text
Ayúdame a redactar una explicación académicamente responsable de por qué el ROC-AUC backtest de 0.9876 no debe venderse como una mejora milagrosa automática respecto al ROC-AUC de validación de Sprint 3.

Quiero una explicación breve, sólida y defendible que mencione:
- diferencia entre sobreajuste y escenario backtest más fácil
- relación entre la definición de la etiqueta premium y las features transaccionales
- por qué aun así el resultado es útil para negocio
- y por qué la comparación metodológica principal sigue siendo la validación temporal del Sprint 3
```

## Prompt adicional para alinear la demo del dashboard

```text
Genera un guion breve para explicar un dashboard de Streamlit de un MVP de machine learning.

El dashboard tiene tres tabs:
- Scoring Individual
- Validacion Sprint 3
- Holdout y ROI

Quiero que el guion explique:
- por qué el primer tab sirve para interpretar un cliente individual con SHAP local
- por qué el segundo tab representa la validación metodológica principal del modelo
- por qué el tercero muestra validación operativa del MVP y valor de negocio
- cómo decir esto en lenguaje claro frente a un jurado académico

No quiero tono comercial exagerado. Quiero precisión, claridad y consistencia metodológica.
```

## Guía sugerida de storytelling diapositiva por diapositiva

Usar esta estructura como referencia base para pedir a NotebookLM o para redactar manualmente el guion final. La idea es que cada diapositiva tenga un mensaje claro, una evidencia concreta y una transición natural hacia la siguiente.

### Diapositiva 1 - Carátula

- Título: `Identificación de Clientes Premium para Optimizar Campañas Comerciales en Olist`
- Debe incluir:
  - nombre del proyecto
  - maestría / módulo
  - integrantes
  - Sprint 4 / Demo Day
- Mensaje clave:
  - `Nuestro objetivo fue convertir un problema de segmentación comercial en un MVP funcional, explicable y útil para negocio.`

### Diapositiva 2 - Problema de negocio

- Debe incluir:
  - qué limita una campaña masiva
  - por qué importa identificar clientes premium
  - impacto esperado en costo y retorno
- Mensaje clave:
  - `No todos los clientes generan el mismo valor, pero una campaña masiva los trata igual.`
- Transición sugerida:
  - `Por eso el reto no era solo predecir, sino priorizar mejor.`

### Diapositiva 3 - Objetivo del proyecto

- Debe incluir:
  - objetivo general
  - definición práctica de la decisión del modelo
  - qué entrega el MVP
- Mensaje clave:
  - `Buscamos estimar qué clientes tienen mayor probabilidad de pertenecer al segmento premium para focalizar campañas.`

### Diapositiva 4 - Recapitulación Sprint 1 a 3

- Debe incluir:
  - Sprint 1: entendimiento del problema y baseline
  - Sprint 2: pipeline reproducible y features
  - Sprint 3: selección y validación del modelo final
- Mensaje clave:
  - `Sprint 3 es la referencia metodológica principal del clasificador.`
- Frase oral sugerida:
  - `Antes de integrar el MVP, primero validamos que el modelo final tuviera una base técnica defendible.`

### Diapositiva 5 - Qué aporta Sprint 4

- Debe incluir:
  - dashboard Streamlit
  - API FastAPI
  - SHAP
  - ROI
  - gobernanza
- Mensaje clave:
  - `Sprint 4 no cambia el modelo; demuestra que ya puede operar como un MVP.`

### Diapositiva 6 - Arquitectura del MVP

- Evidencia visual sugerida:
  - `reports/figures/sprint_04_mvp_architecture.png`
- Debe explicar:
  - pipeline holdout
  - `modelo_final.pkl`
  - dashboard
  - API
  - Docker Compose
- Mensaje clave:
  - `El pipeline prepara los datos y el mismo artefacto del modelo alimenta dashboard y API.`

### Diapositiva 7 - Demo del dashboard

- Debe incluir:
  - `Scoring Individual`
  - `Validacion Sprint 3`
  - `Backtest y ROI`
- Mensaje clave:
  - `El dashboard separa explicación individual, validación metodológica y validación operativa.`
- Frase oral sugerida:
  - `Esto evita confundir el análisis por cliente con la evaluación global del modelo.`

### Diapositiva 8 - Explicabilidad del modelo

- Evidencia visual sugerida:
  - `reports/figures/sprint_04_shap_beeswarm.png`
- Debe incluir:
  - SHAP local en dashboard
  - SHAP global beeswarm en notebook/informe
- Mensaje clave:
  - `El modelo no actúa como una caja negra: podemos explicar tanto un caso individual como el comportamiento general.`

### Diapositiva 9 - Validación metodológica Sprint 3

- Debe incluir:
  - `ROC-AUC val = 0.8081`
  - `Gini val = 0.6162`
  - `PR-AUC val = 0.6010`
  - `Precision val = 0.5225`
  - `Recall val = 0.6041`
  - `F1 val = 0.5603`
- Mensaje clave:
  - `Estas son las métricas que usamos como referencia principal para juzgar la calidad del clasificador.`

### Diapositiva 10 - Backtest operativo Sprint 4

- Debe incluir:
  - `ROC-AUC backtest = 0.9876`
  - `Gini = 0.9751`
  - `Precision = 43.80%`
  - `Recall = 56.11%`
  - `F1 = 49.20%`
  - matriz de confusión
- Mensaje clave:
  - `Este backtest se interpreta como validación operativa del MVP en un escenario más separable.`
- Frase oral sugerida:
  - `No lo presentamos como una mejora milagrosa frente a Sprint 3, sino como una prueba final de integración y uso.`

### Diapositiva 11 - Impacto de negocio

- Evidencia visual sugerida:
  - `reports/figures/sprint_04_holdout_evaluation_roi.png`
- Debe incluir:
  - ROI campaña masiva: `-89.57%`
  - ROI campaña optimizada: `+250.40%`
  - ahorro estimado: `BRL 1,417,365.00`
- Mensaje clave:
  - `El valor del MVP no es solo clasificar, sino hacer viable una campaña mucho más eficiente.`

### Diapositiva 12 - Riesgos, gobernanza y cierre

- Debe incluir:
  - sesgo logístico/geográfico
  - sesgo por método de pago
  - supervisión humana
  - mensaje final
- Mensaje clave:
  - `El modelo se propone como apoyo a decisión comercial, no como automatización ciega.`
- Cierre sugerido:
  - `Sprint 3 valida el modelo. Sprint 4 demuestra que ese modelo puede operar como un MVP explicable y útil para negocio.`

## Prompt listo para pedir la presentación completa a NotebookLM

```text
Usa el material del proyecto para construir una presentación de 12 diapositivas sobre el Demo Day del Sprint 4.

Quiero que la narrativa siga exactamente esta lógica:

1. Carátula
2. Problema de negocio
3. Objetivo del proyecto
4. Recapitulación Sprint 1 a 3
5. Qué aporta Sprint 4
6. Arquitectura del MVP
7. Demo del dashboard
8. Explicabilidad del modelo
9. Validación metodológica Sprint 3
10. Backtest operativo Sprint 4
11. Impacto de negocio
12. Riesgos, gobernanza y cierre

Reglas obligatorias:
- tratar Sprint 3 como referencia metodológica principal del modelo
- tratar Sprint 4 como validación operativa del MVP
- no vender el ROC-AUC backtest como mejora milagrosa
- conectar siempre problema -> solución -> evidencia -> impacto -> gobernanza
- usar tono académico, claro y defendible
- proponer transición verbal entre diapositivas
- indicar qué visual del proyecto usar en cada slide cuando aplique

Para cada diapositiva entrégame:
- título
- objetivo
- mensaje clave
- texto sugerido breve
- visual recomendado
- frase oral sugerida
```
