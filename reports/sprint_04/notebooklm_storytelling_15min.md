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

## Prompts sugeridos para generar storytelling de 15 minutos

### Prompt 1 - estructura completa con carátula

```text
Actúa como asesor de storytelling para una defensa académica de 15 minutos sobre un proyecto de machine learning aplicado a negocio.

Quiero un guion y estructura de diapositivas en español para una presentación de 15 minutos con carátula incluida.

El proyecto trata sobre identificación de clientes premium en Olist mediante un pipeline reproducible, un modelo LightGBM final, validación holdout, dashboard Streamlit, API mínima, explicabilidad SHAP y análisis de ROI.

Necesito que la presentación:
1. incluya carátula
2. haga una recapitulación rápida y clara de los Sprint 1, 2 y 3
3. explique que el foco principal es el Sprint 4
4. conecte siempre problema -> solución -> evidencia -> impacto -> gobernanza
5. reserve un bloque para demo
6. cierre con conclusiones y siguientes pasos

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
- por qué Sprint 4 es la integración final
- cómo funciona el MVP
- qué significa el impacto de ROI
- cómo defendemos la explicabilidad SHAP
- por qué el ROC-AUC holdout alto no debe venderse de forma ingenua
- qué riesgos éticos y de gobernanza reconocemos

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
- incluir explicación del dashboard, API, SHAP beeswarm, ROI y gobernanza
- incluir una forma defendible de explicar por qué el ROC-AUC holdout es 0.9872 sin decir algo engañoso
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
Ayúdame a redactar una explicación académicamente responsable de por qué el ROC-AUC holdout de 0.9872 no debe venderse como una mejora milagrosa automática respecto al ROC-AUC de validación de Sprint 3.

Quiero una explicación breve, sólida y defendible que mencione:
- diferencia entre sobreajuste y escenario holdout más fácil
- relación entre la definición de la etiqueta premium y las features transaccionales
- por qué aun así el resultado es útil para negocio
```
