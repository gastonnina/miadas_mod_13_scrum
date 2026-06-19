# Informe Ético y de Gobernanza - Sprint 4

## Propósito del MVP

El MVP de Sprint 4 identifica clientes con alta probabilidad de pertenecer al segmento premium para focalizar campañas comerciales y evitar el gasto ineficiente de estrategias masivas. El modelo no toma decisiones automáticas irreversibles sobre crédito, exclusión de servicio ni precio individual; su uso previsto es priorización comercial y apoyo a la toma de decisiones humanas.

## Privacidad y tratamiento de datos

### 1. Datos que no se usan para entrenar el modelo

El modelo final no entrena con datos personales directos ni con identificadores operativos de alta sensibilidad. En la selección final de variables quedaron excluidos:

- `customer_unique_id`
- `customer_city`
- `customer_zip_code_prefix`
- variables de gasto directo como `total_spent` y `avg_ticket`

Esto permite sostener que el score no se apoya en nombre, correo, teléfono, dirección exacta, documento personal ni medios de pago completos. Tampoco utiliza números de tarjeta, cuentas bancarias ni credenciales de pago; el dataset solo contiene señales agregadas como tipo principal de pago o cantidad de cuotas.

### 2. Datos que sí aparecen en capas intermedias

En la construcción de la master table se usa `customer_unique_id` como llave técnica de consolidación y trazabilidad. Sin embargo, esa columna queda fuera del set final de entrenamiento y no participa como predictor del modelo.

También se conservan variables de contexto como `customer_state` y `main_payment_type`. Estas no son datos sensibles directos, pero sí pueden funcionar como proxies de condiciones regionales o de bancarización, por lo que deben tratarse como variables de riesgo ético y no como atributos neutrales.

### 3. Criterio de protección esperado para un despliegue real

En este MVP académico no se evidencia un componente formal de cifrado en reposo o en tránsito dentro del repositorio. Por eso la defensa correcta no es afirmar que "todo está encriptado", sino precisar lo siguiente:

- el modelo no fue entrenado con datos personales sensibles directos;
- los identificadores técnicos usados para integrar tablas deben minimizarse en exposición;
- si una versión productiva almacenara identificadores de cliente, payloads de scoring o historiales de consulta, esos artefactos deberían viajar por canales cifrados y persistirse con controles de acceso y cifrado en reposo.

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

### 5. Criterio de reentrenamiento del modelo

El modelo no debe reentrenarse solo por pasar cierto tiempo calendario, sino cuando la evidencia muestre que el comportamiento histórico dejó de representar bien al mercado actual.

Se recomienda disparar revisión o reentrenamiento cuando ocurra alguna de estas condiciones sobre ventanas nuevas no usadas en entrenamiento:

- caída sostenida de métricas de discriminación como `Gini`, `ROC-AUC`, `precision` o `recall`
- deterioro del ROI de campaña focalizada frente al escenario esperado
- cambio fuerte en la tasa predicha de clientes premium
- drift visible en variables clave como `delivered_orders`, `total_items`, `max_payment_installments`, `customer_state` o `top_category_is_high_value`

Como criterio operativo inicial para el MVP:

- alerta amarilla si el `Gini` cae más de `20%` respecto al baseline de referencia
- alerta roja si el `Gini` cae por debajo de un umbral mínimo acordado por negocio y al mismo tiempo se deteriora el ROI o cambia de forma brusca la tasa premium predicha

La base correcta para reentrenar no debe ser cualquier mes histórico mezclado, sino una nueva ventana temporal completa y posterior al período usado para entrenar el artefacto vigente. Esto evita confundir backtesting sobre historia conocida con verdadera adaptación a condiciones nuevas del mercado.

## Límites del MVP

- El modelo se entrenó sobre comportamiento histórico y no mide causalidad.
- El score no reemplaza criterio comercial.
- La simulación financiera es un escenario de negocio, no una garantía contractual de retorno.
- La explicabilidad disponible es suficiente para demo y monitoreo inicial, pero no sustituye una auditoría formal de fairness en producción.

## Recomendación final

El modelo puede usarse como herramienta de apoyo comercial siempre que su despliegue se mantenga dentro de un marco de supervisión humana, versionado explícito, minimización de identificadores y monitoreo de variables logísticas y de pago. La prioridad para una siguiente iteración debe ser robustecer gobernanza, controles de privacidad y compatibilidad reproducible del artefacto antes de cualquier uso operativo más amplio.
