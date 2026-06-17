# Pitch Demo Day - Sprint 4

## Título

Identificación de Clientes Premium para Optimizar Campañas Comerciales en Olist

## Guion de 5 minutos

### Idea fuerza que debe repetirse

Sprint 3 validó metodológicamente el modelo. Sprint 4 demuestra que ese modelo ya puede operar como un MVP reproducible, explicable y útil para negocio.

### 1. Problema

Hoy una campaña masiva trata igual a todos los clientes. Eso implica alto costo comercial y bajo retorno, porque el segmento premium real es pequeño y está mezclado dentro de una base amplia.

### 2. Solución

Construimos un clasificador basado en LightGBM que reutiliza el pipeline del proyecto, transforma el holdout mensual y asigna a cada cliente una probabilidad de pertenecer al segmento premium. El modelo final se entrega como `modelo_final.pkl` y se consume desde un MVP en Streamlit y una API mínima.

Apoyo visual recomendado:

- mostrar `reports/figures/sprint_04_mvp_architecture.png` para explicar en 20-30 segundos cómo se separan pipeline, dashboard y API en la arquitectura final.

### 3. Qué ve el usuario en la demo

- Selección de un cliente desde una muestra curada de casos demo
- Probabilidad de premium y clasificación final con umbral `0.55`
- Variables que más empujan el score en ese caso individual
- Métricas globales del modelo y comparativa de ROI

### 4. Evidencia técnica

- ROC-AUC holdout: `0.9872`
- Gini holdout: `0.9745`
- Precision: `43.10%`
- Recall: `57.06%`
- F1-Score: `49.11%`

Además, generamos un SHAP Summary Plot tipo beeswarm para mostrar que la señal principal del modelo proviene de variables transaccionales y de valor comercial como `delivered_orders`, `max_payment_installments`, `total_items` y `top_category_is_high_value`.

Frase sugerida para defensa:

> Este ROC-AUC holdout no lo vendemos como una mejora milagrosa del modelo frente al Sprint 3. Lo leemos como un escenario final más separable, útil para validar integración, scoring, explicabilidad y ROI del MVP.

Frase complementaria:

> La validación metodológica principal del modelo quedó en Sprint 3. En Sprint 4 el foco es demostrar que ese modelo puede ejecutarse de punta a punta, explicarse con SHAP y sostener una decisión comercial defendible.

### 5. Impacto de negocio

Escenario tradicional:

- Campaña masiva sobre `96,096` clientes
- ROI estimado: `-89.57%`

Escenario con modelo:

- Campaña focalizada sobre `1,659` clientes
- ROI estimado: `+244.79%`
- Ahorro en costo de marketing: `BRL 1,416,555.00`

En otras palabras, pasamos de una estrategia con pérdida estructural a una campaña optimizada con utilidad positiva y foco mucho más eficiente.

### 6. Riesgos y gobernanza

El modelo también usa señales logísticas y de pago, por lo que no proponemos automatización ciega. Recomendamos supervisión humana, versionado explícito del artefacto y monitoreo periódico de sesgos geográficos o financieros.

También aclaramos una limitación metodológica: el holdout técnico abarca agosto a octubre de 2018, pero el volumen real se concentra casi por completo en agosto. Por eso no usamos septiembre y octubre para argumentar tendencias de crecimiento del negocio.

### 7. Cierre

Nuestro MVP demuestra que es posible transformar datos operativos en una herramienta accionable para negocio: identificar mejor a los clientes premium, reducir desperdicio comercial y respaldar decisiones con evidencia explicable.

Lo más importante del Sprint 4 no es decir que “subió el AUC”, sino demostrar que el clasificador final ya está integrado en dashboard/API, puede explicarse en vivo y genera una recomendación comercial con sentido económico.

## Reparto sugerido por integrante

- Persona 1 (`DE`): presentar arquitectura breve, dashboard y API mínima.
- Persona 2 (`DS`): presentar modelo, SHAP global/local y métricas técnicas.
- Persona 3 (`DPO`): presentar problema de negocio, ROI, riesgos éticos y cierre ejecutivo.

## Orden sugerido de exposición

- Minuto `0:00 - 1:00`: problema y oportunidad de negocio.
- Minuto `1:00 - 2:00`: recapitulación de sprints 1 a 3 y transición hacia el MVP.
- Minuto `2:00 - 3:00`: solución técnica y artefactos del MVP.
- Minuto `3:00 - 4:00`: demo en vivo con un caso premium y explicación SHAP.
- Minuto `4:00 - 4:40`: impacto económico y comparativa de ROI.
- Minuto `4:40 - 5:00`: riesgos, gobernanza y cierre.

## Preguntas esperadas y respuestas cortas

### ¿Por qué usar un umbral de `0.55` y no `0.50`?

Porque `0.55` fue el umbral que mejor balanceó precision y recall en validación y mejoró el F1 respecto al umbral por defecto.

### ¿Qué explica que un cliente sea marcado como premium?

Principalmente señales de intensidad transaccional y valor comercial: más órdenes entregadas, más ítems, categorías de mayor valor y mayor uso de cuotas.

### ¿Qué ganamos frente a una campaña masiva?

Pasamos de una estrategia con ROI estimado negativo (`-89.57%`) a una estrategia focalizada con ROI positivo (`+244.79%`) y reducción muy fuerte del costo comercial.

### ¿Por qué el ROC-AUC holdout es tan alto?

Porque el escenario holdout quedó mucho más separable que la validación del Sprint 3 y la etiqueta premium está fuertemente asociada a señales transaccionales. Lo defendemos como validación operativa del MVP, no como comparación directa uno a uno contra la validación temporal previa.

### ¿Entonces cuál es el verdadero aporte del Sprint 4?

El aporte real es convertir el clasificador final en un producto demostrable: dataset de scoring reproducible, dashboard, API, SHAP global y local, simulación de ROI y marco de gobernanza para usar el score con criterio.

### ¿Qué riesgos tiene el modelo?

Puede absorber señales logísticas y de pago que reflejen desigualdades regionales o de bancarización. Por eso recomendamos supervisión humana y monitoreo periódico.

## Mensaje final para diapositiva de cierre

Segmentar mejor no solo mejora métricas técnicas; cambia el resultado económico de la campaña.
