# Informe Ético y de Gobernanza - Sprint 4

## Propósito del MVP

El MVP de Sprint 4 identifica clientes con alta probabilidad de pertenecer al segmento premium para focalizar campañas comerciales y evitar el gasto ineficiente de estrategias masivas. El modelo no toma decisiones automáticas irreversibles sobre crédito, exclusión de servicio ni precio individual; su uso previsto es priorización comercial y apoyo a la toma de decisiones humanas.

## Riesgos éticos identificados

### 1. Sesgo logístico y geográfico

Variables como `avg_delivery_days` y `avg_estimated_delivery_days` aparecen entre las señales relevantes del modelo. Esto implica un riesgo de castigar indirectamente a clientes ubicados en regiones con peores condiciones logísticas o infraestructura menos desarrollada. Si la operación utiliza estas predicciones sin supervisión, podría reforzar desigualdades regionales en la asignación de beneficios.

### 2. Sesgo por bancarización y método de pago

Variables como `max_payment_installments`, `credit_card_flag` y `main_payment_type_boleto` muestran capacidad predictiva. Estas variables pueden estar reflejando no solo comportamiento comercial, sino también acceso diferencial a instrumentos financieros. El riesgo es favorecer clientes con mayor bancarización y dejar subrepresentados perfiles con gasto valioso pero menor acceso a cuotas o tarjeta.

### 3. Riesgo de sobrerrepresentación de intensidad transaccional

El SHAP global muestra que `delivered_orders`, `total_items` y `top_category_is_high_value` concentran gran parte de la señal del modelo. Esto es coherente con el negocio, pero también puede sesgar la estrategia hacia clientes históricamente más visibles en la data, reduciendo la capacidad de detectar clientes emergentes con potencial premium.

## Controles de gobernanza propuestos

### 1. Human-in-the-loop

La predicción no debe ejecutarse como criterio automático único. Recomendamos que el score del modelo se utilice como priorización inicial y que las campañas finales sean revisadas por negocio cuando impliquen presupuestos altos, segmentos sensibles o cambios relevantes en la cobertura geográfica.

### 2. Versionado explícito del modelo

El artefacto oficial del sprint es `models/final/modelo_final.pkl`. Recomendamos mantener versionado manual con:

- nombre del artefacto
- fecha de generación
- entorno Python y librerías críticas
- umbral de decisión utilizado
- dataset de entrenamiento y holdout asociado

### 3. Monitoreo periódico de drift y fairness operativo

En cada nueva corrida del pipeline se deben revisar al menos:

- tasa predicha de clientes premium
- distribución por estado o región
- importancia o SHAP global de variables logísticas
- ROI observado frente a campaña masiva

Si se detecta desplazamiento fuerte en estas métricas, el modelo debe ser reentrenado o recalibrado antes de escalar el MVP.

### 4. Trazabilidad de decisiones

La app y la API deben conservar la posibilidad de explicar cada score mediante contribuciones locales de variables. Esto permite auditar por qué un cliente fue priorizado y facilita la defensa técnica ante jurado o stakeholders.

## Límites del MVP

- El modelo se entrenó sobre comportamiento histórico y no mide causalidad.
- El score no reemplaza criterio comercial.
- La simulación financiera es un escenario de negocio, no una garantía contractual de retorno.
- La explicabilidad disponible es suficiente para demo y monitoreo inicial, pero no sustituye una auditoría formal de fairness en producción.

## Recomendación final

El modelo puede usarse como herramienta de apoyo comercial siempre que su despliegue se mantenga dentro de un marco de supervisión humana, versionado explícito y monitoreo de variables logísticas y de pago. La prioridad para una siguiente iteración debe ser robustecer gobernanza y compatibilidad reproducible del artefacto antes de cualquier uso operativo más amplio.
