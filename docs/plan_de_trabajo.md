# Plan de Trabajo: Módulo de Implementación de Soluciones de Inteligencia Artificial

> [!IMPORTANT]
> **CASO DE ESTUDIO ASIGNADO AL EQUIPO:**
> **Caso 3: Identificación de clientes premium**
> * **Variable Objetivo (Target):** `is_premium` (Clasificación Supervisada)
> * **Funcionalidad:** Detectar clientes con alto valor de vida (LTV) o gasto promedio elevado de manera mensual.
> * **Métricas Principales:** Precision, Recall y Coeficiente de Gini.
> 
> *Este documento ha sido adaptado y personalizado para destacar las actividades, entregables e hitos correspondientes al **Caso 3**, manteniendo el resto de los casos como referencia secundaria.*

---

## Información General
* **Módulo:** Implementación de Soluciones de Inteligencia Artificial Aplicada a los Negocios (Versión 1 – Paralelo 2)
* **Duración:** 12 clases (3 horas cada una) $\rightarrow$ 4 semanas
* **Dataset Base:** [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
* **Equipos:** 10 grupos de 3 integrantes:
  * 1 Ingeniero de Datos
  * 1 Científico de Datos
  * 1 Data Product Owner (DPO)
* **Producto Final:** MVP funcional de IA desplegable y monitoreable enfocado en la clasificación de clientes premium.

---

## 1. Objetivo General
Desarrollar en los estudiantes la capacidad de diseñar, implementar y evaluar soluciones de Inteligencia Artificial (IA) orientadas a la generación de valor empresarial, usando metodologías ágiles y considerando aspectos técnicos, éticos y de gobernanza, aplicados al perfilamiento y detección de clientes de alto valor.

---

## 2. Presentación del Dataset
El **Olist Brazilian E-Commerce Dataset** contiene información real de transacciones de comercio electrónico de la plataforma Olist en Brasil entre 2016 y 2018.

### 2.1. Estructura General
El dataset se compone de las siguientes tablas:

| Tabla | Descripción | Clave Principal |
| :--- | :--- | :--- |
| `olist_orders_dataset.csv` | Registra cada pedido: fechas, estatus, cliente y vendedor. | `order_id` |
| `olist_customers_dataset.csv` | Información de los clientes: ciudad, estado, geolocalización. | `customer_id` |
| `olist_order_items_dataset.csv` | Detalle de productos en cada pedido (SKU, precios, flete). | `order_item_id` |
| `olist_products_dataset.csv` | Descripción de productos, categorías y dimensiones. | `product_id` |
| `olist_sellers_dataset.csv` | Información de los vendedores asociados a Olist. | `seller_id` |
| `olist_geolocation_dataset.csv` | Coordenadas de ubicaciones por ciudad y estado. | `geolocation_zip_code_prefix` |
| `olist_order_reviews_dataset.csv` | Calificaciones, comentarios y sentimiento de los compradores. | `review_id` |
| `olist_order_payments_dataset.csv` | Métodos de pago y valores de las transacciones. | `order_id` |
| `product_category_name_translation.csv` | Traducción de categorías de producto de portugués a inglés. | `product_category_name` |

### 2.2. Características y Potencial Analítico (Caso de Estudio 3)
* **Volumen:** Más de 100,000 registros distribuidos en múltiples tablas relacionales y temporales.
* **Periodo:** Pedidos entre 2016 y 2018.
* **Foco Analítico del Equipo:** Estimar el valor del cliente en el tiempo (LTV) y segmentar de forma supervisada a aquellos con un volumen transaccional de alto impacto (clientes Premium), permitiendo estrategias personalizadas de marketing y retención.

---

## 3. Trabajo en Grupos
El equipo (célula de desarrollo ágil de IA) diseñará, implementará y presentará un MVP para resolver la **Identificación de clientes premium (Caso de Estudio 3)**.

### 3.1. Requisitos del Trabajo
1. Delimitar el problema de negocio específico (pérdida de clientes de alto valor, incentivos premium).
2. Desarrollo de un pipeline de IA completo (ingeniería de datos hasta despliegue).
3. Trabajo en sprints semanales con entregas iterativas.
4. Aplicación de métricas de negocio (gasto promedio de segmento, retención) y técnicas (Precision, Recall).
5. Creación de un Dashboard (para visualización del perfil premium) y una API funcional para scoring en tiempo real.
6. Evaluación de consideraciones éticas (sesgos en la clasificación socioeconómica o geográfica).

### 3.2. Entregables Finales
* **MVP de IA:** Modelo de clasificación entrenado (archivo `.pkl`) + pipeline reproducible mensualmente.
* **Dashboard e Interfaz:** Dashboard (ej. Streamlit) y API funcional de scoring.
* **Informes:** Informe ejecutivo y ético sobre gobernanza y discriminación algorítmica.
* **Presentación:** Pitch final tipo demo empresarial.

---

## 4. Roles y Responsabilidades (Enfocados en el Caso 3)

| Rol | Descripción | Responsabilidades Específicas del Caso 3 |
| :--- | :--- | :--- |
| **Ingeniero de Datos** | Especialista en gestión, limpieza, integración y automatización de flujos de datos. | - Integrar tablas de órdenes, pagos y clientes para consolidar el perfil de facturación por cliente (`customer_id`).<br>- Calcular agregados históricos de compras (monto total, frecuencia, recencia).<br>- Automatizar el pipeline ETL para incorporar nuevos datos mensuales de facturación.<br>- Desplegar la API que recibe un ID de cliente y retorna si es Premium. |
| **Científico de Datos** | Responsable del análisis exploratorio, modelado y evaluación técnica de soluciones de IA. | - Analizar la distribución de gasto de los clientes para establecer el umbral de etiqueta (`is_premium`).<br>- Entrenar modelos de clasificación (Random Forest, XGBoost, etc.).<br>- Optimizar la relación Precision-Recall para minimizar falsos positivos/negativos.<br>- Documentar la importancia de las variables (ej. frecuencia de compra vs. valor del flete). |
| **Data Product Owner (DPO)** | Conecta los resultados técnicos con los objetivos de negocio. Lidera la comunicación y planificación. | - Definir el criterio de negocio de "Cliente Premium" y las métricas de impacto (ej. incremento de ticket promedio).<br>- Coordinar sprints y priorizar tareas de desarrollo.<br>- Analizar riesgos éticos (evitar sesgos geográficos o discriminación en ofertas).<br>- Preparar el storytelling de negocio de cómo la identificación de clientes premium maximiza el ROI. |

---

## 5. Casos de Estudio y Categorías de Negocio

### 5.1. Tabla de Casos de Estudio

| Nº | Tema | Target / Variable Objetivo | Tipo de Modelo | Funcionalidad del Modelo | Temporalidad | Métricas Sugeridas |
| :-: | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Predicción de satisfacción | `review_score` o `customer_satisfaction` | Regresión / Clasificación | Estimar el nivel de satisfacción del cliente según tiempos, valor y calidad del pedido. | Mensual | RMSE, F1, $R^2$, Gini |
| 2 | Detección de retrasos en la entrega | `is_late_delivery` | Clasificación | Identificar pedidos con alto riesgo de retraso antes de la entrega. | Diario / Semanal | Precision, Recall, ROC-AUC |
| **⭐ 3** | **Identificación de clientes premium (ASIGNADO)** | **`is_premium`** | **Supervisado, Clasificación** | **Detectar clientes con alto valor de vida (LTV) o gasto promedio elevado.** | **Mensual** | **Precision, Recall, Gini (Foco principal)** |
| 4 | Predicción de cancelaciones | `is_canceled` | Clasificación | Anticipar qué pedidos serán cancelados para tomar acciones preventivas. | Mensual | Recall, F1-score, ROC-AUC, Gini |
| 5 | Forecast de demanda por categoría | `demand_next_month` | Regresión / Series temporales | Predecir las ventas futuras por categoría de producto. | Mensual | RMSE, MAPE, MAE |
| 6 | Optimización de tiempos de entrega | `expected_delivery_days` | Regresión | Predecir la duración óptima del envío según ubicación, proveedor y método. | Diario / Semanal | RMSE, MAE |
| 7 | Análisis de churn | `is_churn` | Clasificación | Detectar clientes inactivos o con riesgo de abandono. | Mensual | Recall, F1, AUC |
| 8 | Recomendación personalizada | `product_id_recommended` | Sistema de recomendación | Sugerir productos basados en historial y similitud de compras. | Continuo | Precision@k, Recall@k |
| 9 | Predicción de método de pago | `payment_type` | Clasificación multiclase | Predecir el método de pago preferido por el cliente o para un pedido. | Mensual | Accuracy, Macro F1 |
| 10 | Forecast de ingresos mensuales | `monthly_revenue` | Series temporales | Predecir los ingresos globales o por categoría del e-commerce. | Mensual | RMSE, MAPE |

### 5.2. Categorías Temáticas del Equipo (Foco en Caso 3)
* **Scoring / Riesgo / Valor del Cliente (ENFOQUE PRINCIPAL):** Asigna un puntaje o etiqueta a clientes (`is_premium`) según su valor transaccional o potencial de gasto futuro.
* **LTV / Valor Futuro (ENFOQUE COMPLEMENTARIO):** Estima el valor económico total que un cliente premium aportará a la empresa, permitiendo calcular el límite presupuestario para campañas de fidelización (CAC vs LTV).
* *Otras Categorías (Referenciales):* Propensión de negocio, Churn, Recomendación, Forecasting, Optimización Operativa.

---

## 6. Plan General de Clases (12 Sesiones)

| Semana | Clase | Enfoque | Actividades Principales |
| :-: | :-: | :--- | :--- |
| **1** | 1 | Introducción al Área, Metodología Ágil y Storytelling | Explicar dinámica de Sprints, roles y objetivos. Introducción al dataset Olist. Taller de storytelling de datos. |
| **1** | 2 | **Sprint 1 – Diseño y EDA inicial** | Definir problema, hipótesis, variables. EDA inicial y métricas base. |
| **1** | 3 | Exposición Sprint 1 | Presentación de avances por grupo. Análisis EDA preliminar y posible target/temporalidad. |
| **2** | 4 | **Sprint 2 – Ingeniería de datos y pipeline** | Limpieza, feature engineering y preparación de pipeline reproducible. |
| **2** | 5 | Avances Sprint 2 | Revisión técnica de pipeline y validación de métricas. Exposición de 4 grupos. |
| **2** | 6 | Exposición Sprint 2 | Presentación de modelos preliminares y defensa del target final. |
| **3** | 7 | **Sprint 3 – Hiperparametrización y modelo final** | Tuning, validación cruzada y exportación del modelo final (pickle). |
| **3** | 8 | Avances Sprint 3 | Ejecución de comparación de modelos. Exposición de 3 grupos. |
| **3** | 9 | Exposición Sprint 3 | Presentaciones de performance final del modelo. |
| **4** | 10 | **Sprint 4 – Integración y gobernanza** | Integración con dashboard/API, MLflow, ética y gobernanza. |
| **4** | 11 | Preparación del Demo Day | Práctica de storytelling y revisión de métricas de negocio. Exposición de 3 grupos. |
| **4** | 12 | Demo Day (Presentaciones finales) | Presentación, ejecución en vivo y defensa del MVP1. |

---

## 7. Lógica y Seguimiento de Sprints (Checklist de Control - Caso 3)

### 📋 Sprint 1: Definición del Problema y EDA Inicial (Semana 1)
**Objetivo:** Formular el problema de negocio de clientes premium, validar hipótesis con datos y establecer métricas base.

* [x] **Actividades Clave:**
  * [x] **Definir el problema de negocio:** Establecer qué constituye un cliente de alto valor y cómo identificarlos ayuda a la retención y venta cruzada.
  * [x] **Formular hipótesis de negocio:** (Ej. *Los clientes premium compran con mayor frecuencia, realizan pagos de alto valor o eligen fletes express*). *(Entregable: Excel, Notebook)*
  * [x] **Establecer Master Table:** Seleccionar variables clave de órdenes (`olist_orders_dataset`), pagos (`olist_order_payments_dataset`) y clientes (`olist_customers_dataset`) integrando a nivel de `customer_unique_id`.
  * [x] **Definir variable objetivo (`is_premium`):** Determinar técnicamente el umbral (percentil de gasto, número mínimo de transacciones) para etiquetar a los clientes.
  * [x] **Realizar EDA:** Analizar la concentración del gasto (Curva de Pareto / Regla del 80-20), distribuciones de frecuencia de compra y valores de flete.
  * [x] **Calcular Baseline Inicial:**
    * [x] Métricas técnicas base: Precision, Recall y Coeficiente de Gini usando un modelo clasificador muy simple (heurística de gasto acumulado).
    * [x] Métricas de negocio base: Gasto promedio mensual del segmento clasificado como premium y porcentaje del total de la facturación que representan.
* [x] **Entregables del Sprint:**
  * [x] Notebook de Jupyter con el EDA, análisis de Pareto y cálculo del baseline de clasificación.
  * [x] Documento con la definición formal del umbral para `is_premium` y métricas meta.
  * [x] Presentación ejecutiva del Sprint con hipótesis de comportamiento del cliente premium y definición del target preliminar.
* [x] **Cumplimiento de Roles:**
  * [x] **Ingeniero de Datos:** Consolidar datos de múltiples tablas y unificar registros a nivel de cliente para el análisis del gasto.
  * [x] **Científico de Datos:** Analizar cuantitativamente los umbrales de gasto, estructurar el dataset de modelado y programar el baseline.
  * [x] **Data Product Owner:** Validar si el umbral del gasto del "Cliente Premium" hace sentido desde una perspectiva comercial y de marketing.

---

### 📋 Sprint 2: Ingeniería de Datos y Pipeline Reproducible (Semana 2)
**Objetivo:** Construir un pipeline de datos robusto y escalable enfocado en features transaccionales y de perfilamiento de clientes.

* [ ] **Actividades Clave:**
  * [ ] **Limpieza y transformación:** Manejar nulos en valor de transacciones, normalizar distribuciones de gasto sesgadas.
  * [ ] **Feature Engineering (Foco en Clientes):**
    * [ ] Crear métricas RFM (Recency: días desde la última compra; Frequency: cantidad de compras; Monetary: monto total gastado).
    * [ ] Calcular categorías de producto preferidas, cantidad de métodos de pago utilizados y promedio de cuotas de pago.
  * [ ] **Pipeline reproducible:** Encapsular transformaciones de datos en un objeto reproducible (`sklearn.pipeline` o scripts modulares).
  * [ ] **Simulación mensual:** Simular la llegada de nuevos datos transaccionales mensuales para probar que el pipeline recalcula el estado Premium sin fallos.
  * [ ] **Medición final:** Medir métricas de clasificación base con el dataset transformado completo.
* [ ] **Entregables del Sprint:**
  * [ ] Script modularizado de Python (`.py`) o notebook con el pipeline transaccional de extremo a extremo.
  * [ ] Documentación técnica de las transformaciones y del cálculo de las variables RFM.
  * [ ] Reporte de métricas técnicas (Precision y Recall) obtenidas en validación con el pipeline consolidado.
  * [ ] Defensa final de la definición de la variable target `is_premium`.
* [ ] **Cumplimiento de Roles:**
  * [ ] **Ingeniero de Datos:** Automatizar el cálculo de agregados de compras (RFM) y modularizar el pipeline en código limpio.
  * [ ] **Científico de Datos:** Validar la correlación de las nuevas variables transaccionales creadas con la etiqueta de cliente premium.
  * [ ] **Data Product Owner:** Analizar que los datos simulados mensuales representen de forma realista la dinámica de facturación real.

---

### 📋 Sprint 3: Hiperparametrización y Modelo Final (Semana 3)
**Objetivo:** Optimizar el modelo clasificador de clientes premium, evaluar detalladamente su rendimiento y exportar el artefacto.

* [ ] **Actividades Clave:**
  * [ ] **Entrenamiento de modelos candidatos:** Entrenar clasificadores (Logistic Regression, Decision Trees, Random Forest, XGBoost).
  * [ ] **Optimización de hiperparámetros (Optuna / GridSearchCV):** Optimizar hiperparámetros enfocados en maximizar **Recall** (no perder clientes de alto valor) y controlar **Precision** (evitar dar beneficios premium a clientes comunes).
  * [ ] **Validación Cruzada:** Implementar validación cruzada estratificada para evitar sobreajuste ante clases minoritarias (los clientes premium suelen ser un porcentaje bajo de la base).
  * [ ] **Serialización:** Exportar el modelo final entrenado en formato `.pkl` (pickle) o formato equivalente.
  * [ ] **Documentar retraining:** Crear un script de reentrenamiento periódico para adaptar el modelo a cambios estacionales de compra.
* [ ] **Entregables del Sprint:**
  * [ ] Archivo binario serializado del clasificador premium (`.pkl`).
  * [ ] Notebook con la comparativa de algoritmos, curvas de ROC, matrices de confusión y justificación del modelo seleccionado.
  * [ ] Gráficas de importancia de variables (*feature importance*) y análisis de impacto de variables RFM.
* [ ] **Cumplimiento de Roles:**
  * [ ] **Ingeniero de Datos:** Diseñar el repositorio y control de versiones del modelo binario final (MLflow o Git).
  * [ ] **Científico de Datos:** Ejecutar los experimentos de búsqueda de hiperparámetros, validar el modelo final y generar gráficos técnicos.
  * [ ] **Data Product Owner:** Evaluar el coste financiero de clasificar erróneamente a clientes (falsos positivos/negativos) en la estrategia comercial.

---

### 📋 Sprint 4: Integración, Despliegue y Gobernanza (Semana 4)
**Objetivo:** Integrar el clasificador de clientes premium en una API y Dashboard, evaluar la gobernanza y considerar los aspectos éticos del perfilamiento.

* [ ] **Actividades Clave:**
  * [ ] **Dashboard interactivo:** Desarrollar una interfaz (ej. Streamlit) que muestre la lista de clientes premium y un panel de análisis de comportamiento de compra por segmento.
  * [ ] **API de Scoring:** Implementar un endpoint API (FastAPI) que reciba los datos de un cliente y retorne la predicción instantánea de su estatus Premium (`is_premium: true/false`).
  * [ ] **Métricas de impacto de negocio:** Traducir la mejora en Precision/Recall a impacto financiero (ej. *ahorro en campañas al dirigir promociones exclusivas únicamente a clientes con alta probabilidad de ser premium*).
  * [ ] **Gobernanza y Ética:** Evaluar si el modelo discrimina indirectamente por geolocalización o métodos de pago (sesgos geográficos o de exclusión financiera).
  * [ ] **Storytelling:** Estructurar la presentación final explicando el problema, el modelo de clientes premium y su viabilidad de negocio.
* [ ] **Entregables del Sprint:**
  * [ ] MVP funcional: API web fastapi/flask y Dashboard Streamlit operativo conectado al modelo `.pkl`.
  * [ ] Informe ético y de gobernanza (2 a 3 páginas) abordando la privacidad de los datos financieros del cliente.
  * [ ] Diapositivas finales de la demo empresarial y pitch técnico/comercial ensayado.
* [ ] **Cumplimiento de Roles:**
  * [ ] **Ingeniero de Datos:** Desplegar y probar la API y Dashboard, e implementar el control del modelo en MLflow.
  * [ ] **Científico de Datos:** Validar el comportamiento de la API en producción y redactar la sección técnica de gobernanza.
  * [ ] **Data Product Owner:** Diseñar la estrategia de marketing basada en el Dashboard y liderar el pitch comercial del Demo Day.

---

## 8. Criterios de Evaluación

### 8.1. Ponderación por Entregables

| Criterio / Sprint | Peso | Evidencias Esperadas |
| :--- | :---: | :--- |
| **Sprint 1 – Definición y EDA** | 15% | Problema de clientes premium definido, hipótesis de comportamiento validadas, baseline. |
| **Sprint 2 – Pipeline e Ingeniería** | 25% | Pipeline de datos modular, variables de RFM y transformaciones listas, simulación de datos mensuales. |
| **Sprint 3 – Modelo Final** | 25% | Clasificador optimizado, archivo serializado (`.pkl`), métricas meta logradas (`Precision`, `Recall`, `Gini`). |
| **Sprint 4 – Integración y Gobernanza** | 25% | Dashboard de clientes premium/API FastAPI operativa, informe de gobernanza/ética, pitch, Demo Day. |
| **Exposición Caso Paralelo** | 10% | Exposición individual/grupal (Viernes) sobre un modelo seleccionado usando un **dataset diferente al Olist**: <br>- Informe de investigación en formato IEEE (min. 6 páginas + 10 referencias).<br>- Presentación de diapositivas (PPTX).<br>- Compartir código fuente documentado en GitHub.<br>- Exposición y defensa técnica de 30 minutos. |

> [!WARNING]
> **Penalizaciones:** Se aplicará una reducción del **5% de la nota final** por cada retraso o incumplimiento en los plazos de entrega de tareas.
