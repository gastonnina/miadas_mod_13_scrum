# Guía de Diapositivas para Defender Resultados - Sprint 4

## Mensaje central

El Sprint 4 no debe presentarse como “el modelo mejoró mágicamente”, sino como:

- integración reproducible del artefacto final;
- validación operativa sobre holdout;
- demostración de explicabilidad;
- prueba de impacto de negocio mediante ROI;
- cierre con gobernanza y límites metodológicos.

Frase eje para repetir en varias láminas:

> Sprint 3 valida el modelo. Sprint 4 demuestra que ese modelo puede convertirse en un MVP funcional, explicable y útil para negocio.

## Orden recomendado de diapositivas

1. Portada y objetivo del proyecto.
2. Recapitulación rápida de sprints 1 a 3.
3. Qué entrega Sprint 4 y por qué era necesario.
4. Arquitectura del MVP: pipeline, dashboard y API.
5. Demo o capturas del flujo de scoring.
6. Métricas holdout con interpretación correcta.
7. SHAP global tipo beeswarm y explicación breve.
8. ROI y ahorro frente a campaña masiva.
9. Riesgos, ética y gobernanza.
10. Cierre ejecutivo.

## Qué sí mostrar

### 1. Recapitulación breve de sprints 1 a 3

- Sprint 1: baseline y definición del problema
- Sprint 2: pipeline reproducible y features
- Sprint 3: selección y tuning del modelo final

### 2. Validación técnica del Sprint 4

Mostrar:

- `ROC-AUC holdout = 0.9872`
- `Gini = 0.9745`
- `Precision = 43.10%`
- `Recall = 57.06%`
- `F1 = 49.11%`

Pero acompañar siempre con la siguiente idea:

> Este resultado refleja un escenario holdout altamente separable y útil para validar el MVP; no lo interpretamos como comparación directa y simple contra la validación temporal del Sprint 3.

Complemento recomendado para decirlo en voz alta:

> El objetivo del Sprint 4 no era volver a competir entre modelos, sino demostrar integración, explicabilidad y utilidad comercial con el artefacto final ya elegido.

### 3. Explicabilidad

Usar:

- `reports/figures/sprint_04_shap_beeswarm.png`

Mensaje:

- el modelo se apoya principalmente en señales transaccionales y de valor comercial;
- no parece una caja negra arbitraria;
- las variables logísticas y de pago también influyen, por lo que requieren vigilancia ética.

### 4. Impacto de negocio

Usar:

- `reports/figures/sprint_04_holdout_evaluation_roi.png`

Mensaje:

- campaña masiva: ROI negativo;
- campaña focalizada por modelo: ROI positivo;
- reducción fuerte del costo comercial.

### 5. Arquitectura final del MVP

Usar:

- `reports/figures/sprint_04_mvp_architecture.png`

Mensaje:

- el pipeline no desaparece; prepara el dataset de scoring del holdout;
- dashboard y API corren en contenedores separados;
- ambos reutilizan el mismo `modelo_final.pkl` y el mismo esquema de features;
- esto demuestra trazabilidad e integración real, no solo analisis en notebook.

## Qué NO mostrar sin aclaración

### 1. Gráficas de recompra o actividad con septiembre y octubre 2018

No usarlas como evidencia de tendencia de crecimiento.

Razón:

- el holdout técnico cubre agosto-octubre 2018;
- pero el volumen real de órdenes está casi totalmente concentrado en agosto;
- septiembre y octubre tienen muy pocos registros, por lo que los porcentajes se distorsionan.

Si se muestra una gráfica mensual, cortar en `2018-08` o agregar una nota visible:

> Septiembre y octubre presentan cola truncada del dataset y no se interpretan como tendencia real.

## Frases recomendadas para defensa

### Sobre el AUC alto

> No interpretamos el ROC-AUC holdout como una mejora milagrosa del modelo. Lo interpretamos como una validación final en un escenario mucho más separable, útil para demostrar integración técnica, explicabilidad y valor de negocio del MVP.

### Sobre posible overfitting

> No vemos evidencia clásica de overfitting solo porque el holdout tenga AUC más alto. Más bien observamos un holdout estructuralmente más fácil y una etiqueta premium fuertemente conectada con señales transaccionales.

### Sobre la comparabilidad metodológica

> La comparación metodológica principal del modelo sigue siendo la validación temporal del Sprint 3. El holdout del Sprint 4 complementa esa validación como prueba de integración y demo de negocio.

### Sobre el valor real del sprint

> El valor de Sprint 4 no está en “inflar” una métrica, sino en cerrar el ciclo completo: datos alineados, scoring reproducible, visualización, explicación del score, impacto de negocio y criterios de uso responsable.

## Archivos visuales sugeridos para la presentación

- `reports/figures/sprint_03_feature_importance.png`
- `reports/figures/sprint_04_mvp_architecture.png`
- `reports/figures/sprint_04_shap_beeswarm.png`
- `reports/figures/sprint_04_holdout_evaluation_roi.png`
- capturas del dashboard y de la API si quieren respaldo estático

## Recomendación final

Si van a defender resultados con rigor, no conviene presentar septiembre-octubre 2018 como evidencia de evolución temporal. Si quieren una validación temporal más estricta y limpia, el siguiente paso sería redefinir el holdout a una ventana realmente completa y recalcular métricas oficiales.
