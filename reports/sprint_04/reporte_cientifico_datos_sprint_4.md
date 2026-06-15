# Reporte de Transición — Sprint 4: Entregables del Científico de Datos

**Proyecto:** Identificación de Clientes Premium — Olist Brazilian E-Commerce  
**Rol Emisor:** Científico de Datos (Gerick Toro)  
**Destinatarios:** Ingeniero de Datos (Gaston Nina) y Data Product Owner (Marcelo De la Quintana)  
**Sprint:** 4 de 4  
**Fecha:** 15 de junio de 2026  

---

## 1. Introducción y Contexto

Estimado equipo:

He concluido la fase de **ciencia de datos y analítica** del Sprint 4. Para mantener la separación de roles y la dinámica ágil de Scrum, he dejado listos y ejecutados todos los insumos de datos, cálculos matemáticos, explicabilidad y simulaciones financieras en un único notebook integrador. 

A partir de este punto, el **Ingeniero de Datos** puede iniciar la construcción de la aplicación Streamlit, la API y el empaquetado Docker, mientras que el **Product Owner** puede redactar los informes finales y el pitch de presentación, utilizando los números y métricas oficiales validados aquí.

---

## 2. Insumos Analíticos y Datos Generados (Listos para uso)

El notebook [notebooks/sprint_04_integration/04_demo_validation.ipynb](file:///home/gerick/Documents/MIADAS/miadas_mod_13_scrum/notebooks/sprint_04_integration/04_demo_validation.ipynb) ha sido ejecutado completamente sobre la ventana de holdout (`holdout_3m`). Este proceso ha generado los siguientes archivos en la carpeta `data/processed/`:

### A. Dataset de Inferencia Holdout
* **Archivo:** `data/processed/holdout_features_selected.parquet`  
* **Descripción:** Contiene las características del holdout mensual de 96,096 clientes. Se encuentra **filtrado y ordenado** con las 28 columnas exactas requeridas por el preprocesador del modelo final.
* **Uso:** Sirve como la base de datos de scoring para el dashboard o la API.

### B. Muestra de Casos Ejemplo para la Demo
* **Archivos:** `data/processed/demo_sample_scoring.parquet` y `data/processed/demo_cases.csv`  
* **Descripción:** Una muestra controlada de 4 clientes reales del holdout para la demo interactiva:
  * 2 clientes clasificados de forma inequívoca como **Premium** (alta probabilidad).
  * 2 clientes clasificados de forma inequívoca como **Regulares** (baja probabilidad).
* **Uso:** El dashboard Streamlit puede cargar este archivo ligero directamente para el selector de clientes en la presentación, evitando buscar entre los 96,000 registros.

### C. Gráfico Comparativo del ROI y Evaluación
* **Archivo:** `reports/figures/sprint_04_holdout_evaluation_roi.png`  
* **Descripción:** Imagen que contiene la matriz de confusión del modelo sobre el holdout y la gráfica de barras de la comparativa de ROI.
* **Uso:** Listo para insertar directamente en las diapositivas de la presentación ejecutiva.

---

## 3. Resultados Analíticos y Métricas de Negocio Oficiales

Para la redacción del informe de negocio y el pitch, estos son nuestros números oficiales de holdout:

* **Tasa real de premium en holdout (mensual):** **1.30%** (1,253 de 96,096 clientes) bajo el umbral inmutable de **BRL 197.01** (P80 de la población de desarrollo).
* **Regla de decisión del modelo:** Clasificar como premium si la probabilidad estimada es **$\ge$ 0.55** (umbral óptimo obtenido por Optuna).
* **Tasa de premium predicha en holdout:** **1.73%** (1,659 clientes).
* **Métricas técnicas en holdout:** ROC-AUC de `0.9872` | Gini de `0.9745` | Precision de `43.10%` | Recall de `57.06%` | F1-Score de `49.11%`.

### Captura de Gasto y Simulación de Campaña (BRL 15 costo / BRL 120 retorno)
1. **Facturación total en holdout:** BRL 985,414.28.
2. **Facturación capturada:** El segmento premium representa el 52.87% del gasto total. Nuestro modelo, impactando a solo el **1.73% de clientes**, captura el **64.11% de toda la facturación premium** (BRL 333,999.80).
3. **Estrategia Masiva (All):** Costo BRL 1.44M, Retorno BRL 150K $\rightarrow$ **Pérdida neta de BRL -1.29 millones (ROI: -89.57%)**.
4. **Estrategia del Modelo (Umbral 0.55):** Costo BRL 24,885, Retorno BRL 85.8K $\rightarrow$ **Utilidad neta de BRL +60,915.00 (ROI: +244.79%)**.
5. **Ahorro generado:** Reducción del **98.27% del costo de marketing** (ahorro de **BRL 1,416,555.00**).

---

## 4. Tareas Pendientes del Equipo (Transición)

A partir de este punto, cada rol tiene asignadas las siguientes actividades para finalizar el MVP del Sprint 4:

### 🛠️ Para el Ingeniero de Datos (Gaston)

#### 1. Implementar la interfaz en `app/dashboard/app.py`
Debe construirse el MVP de visualización utilizando **Streamlit**.
* **Entradas:** Cargar el modelo `models/final/modelo_final.pkl` y leer la muestra demo `data/processed/demo_sample_scoring.parquet` (o la base completa en `holdout_features_selected.parquet`).
* **Componentes visuales sugeridos:**
  * Selector de cliente (usando el ID único).
  * Panel de predicción con badges distintivas (Premium vs. Regular).
  * Barra de progreso para la probabilidad e indicador del umbral (0.55).
  * Explicabilidad local mediante las contribuciones SHAP nativas de LightGBM (ejemplo de código incluido en el notebook y en la sección 5 de este reporte).
  * Panel de métricas globales (ROI, ahorro y total spent) para que el jurado vea el impacto macro.

#### 2. Completar Dockerización de la App (`docker/Dockerfile.dashboard`)
* Empaquetar la aplicación Streamlit en el puerto `8501`.
* Asegurar que se instalen las dependencias correctas definidas en `pyproject.toml` (especialmente `lightgbm`, `scikit-learn`, `pandas` y `streamlit`).
* Integrar el servicio en `docker-compose.yml` para levantar la aplicación con un solo comando.

#### 3. API de Scoring como Stretch Goal (`app/api/main.py`)
* Si el dashboard queda estable, se puede implementar un endpoint mínimo `/predict` (FastAPI) que reciba los features de un cliente y retorne la probabilidad y la clasificación final.

---

### 📋 Para el Data Product Owner (Marcelo)

#### 1. Redactar el Informe Ético y de Gobernanza (`reports/sprint_04/informe_etico_gobernanza.md`)
Utilizar los hallazgos del modelo para documentar los riesgos éticos:
* **Sesgo logístico/geográfico:** El modelo da un peso del 10.6% a `avg_estimated_delivery_days` y 8.5% a `avg_delivery_days`. Dado que el sudeste de Brasil tiene mejor infraestructura que el norte/noreste, existe riesgo de discriminación indirecta por código postal.
* **Sesgo por método de pago:** Explicar cómo el uso de variables de cuotas de pago (`max_payment_installments`) favorece a clientes bancarizados, por lo que se deben sugerir métodos de pago inclusivos.
* **Gobernanza:** Proponer versionado del `.pkl` e implementar supervisión humana (Human-in-the-loop) para decisiones comerciales de alto impacto.

#### 2. Preparar Storytelling y pitch del Demo Day (`reports/sprint_04/pitch_demo_day.md`)
* Construir el guión de presentación de 5 minutos centrado en la estructura:
  * **Problema:** Desperdicio de dinero en campañas de marketing masivas.
  * **Solución:** Clasificador LightGBM basado en Pareto y transacciones operativas.
  * **Demo:** Mostrar cómo se scorea un cliente en vivo y se explican sus variables con SHAP.
  * **Impacto:** Pasar de perder BRL 1.29M (Campaña Masiva) a ganar BRL 61K (Campaña del Modelo) con un ROI de +244%.

---

## 5. Nota Técnica: Cálculo de SHAP Nativo para el Dashboard

Para simplificar el despliegue del dashboard y evitar instalar la librería externa `shap` en el contenedor, se implementó en el notebook de validación un método directo de LightGBM que calcula los SHAP values en un par de líneas sobre las features preprocesadas por el ColumnTransformer. 

Gaston, puedes utilizar esta lógica en el archivo `app.py`:

```python
# 1. Extraer los componentes del pipeline
preprocessor = pipeline.named_steps['preprocessor']
lgb_model = pipeline.named_steps['model']

# 2. Transformar el DataFrame del cliente seleccionado (1 fila)
# X_cust debe tener las 28 variables en el orden de selected_features
X_proc = preprocessor.transform(X_cust)
feature_names_out = preprocessor.get_feature_names_out()

# 3. Calcular contribuciones locales (SHAP nativo de LightGBM)
# Devuelve array de tamaño (n_features_out + 1)
contribs = lgb_model.predict(X_proc, pred_contrib=True)[0]
shap_values = contribs[:-1]
base_value = contribs[-1]

# 4. Crear DataFrame para graficar
cleaned_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names_out]
shap_df = pd.DataFrame({
    'Variable': cleaned_names,
    'SHAP': shap_values
}).sort_values('SHAP', key=abs, ascending=False)
```

¡Quedo a disposición de cualquier consulta para apoyarlos en el desarrollo final!
