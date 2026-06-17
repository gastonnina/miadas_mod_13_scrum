from __future__ import annotations

import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
notebook_path = project_root / 'notebooks' / 'sprint_04_integration' / '04_demo_validation.ipynb'

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Sprint 4 — Integración, Validación de MVP y Métricas de Negocio\n",
    "\n",
    "**Módulo:** Módulo 13 - Scrum e IA Aplicada  \n",
    "**Caso de Estudio:** Caso 3 — Identificación de Clientes Premium  \n",
    "**Entregables del Sprint 4:**\n",
    "1. **S4-DE-01 | Dockerizar y operar pipeline reutilizado**: Flujo batch sobre holdout y alineación de features.\n",
    "2. **S4-DS-01 | Insumos analíticos del MVP**: Definición oficial del criterio de decisión y muestra demo.\n",
    "3. **S4-DS-02 | Explicabilidad y SHAP**: Explicabilidad a nivel de feature importance global y local (SHAP nativo).\n",
    "4. **S4-DS-03 | Consolidación de métricas de negocio**: Conversión a KPIs de negocio y simulación de ROI de campaña.\n",
    "\n",
    "Este notebook consolida y ejecuta estas actividades sobre el conjunto de datos de holdout mensual (`holdout_3m`)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from __future__ import annotations\n",
    "\n",
    "import json\n",
    "import pickle\n",
    "from pathlib import Path\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import (\n",
    "    ConfusionMatrixDisplay,\n",
    "    RocCurveDisplay,\n",
    "    PrecisionRecallDisplay,\n",
    "    classification_report,\n",
    "    confusion_matrix,\n",
    "    f1_score,\n",
    "    precision_score,\n",
    "    recall_score,\n",
    "    roc_auc_score,\n",
    ")\n",
    "\n",
    "sns.set_theme(style='whitegrid')\n",
    "\n",
    "# ── Configuración de Rutas ──────────────────────────────────────────────────\n",
    "project_root = Path.cwd().resolve()\n",
    "if not (project_root / 'data').exists():\n",
    "    for candidate in [project_root, *project_root.parents]:\n",
    "        if (candidate / 'data').exists() and (candidate / 'notebooks').exists():\n",
    "            project_root = candidate\n",
    "            break\n",
    "\n",
    "HOLDOUT_RFM_PATH    = project_root / 'data' / 'processed' / 'holdout_features_rfm.parquet'\n",
    "HOLDOUT_CLEAN_PATH  = project_root / 'data' / 'processed' / '03_master_table_clean_holdout.parquet'\n",
    "METADATA_PATH       = project_root / 'data' / 'processed' / '06_features_selected_metadata.json'\n",
    "MODEL_PATH          = project_root / 'models' / 'final' / 'modelo_final.pkl'\n",
    "SELECTED_OUT_PATH   = project_root / 'data' / 'processed' / 'holdout_features_selected.parquet'\n",
    "DEMO_SAMPLE_PATH    = project_root / 'data' / 'processed' / 'demo_sample_scoring.parquet'\n",
    "FIGURES_DIR         = project_root / 'reports' / 'figures'\n",
    "\n",
    "FIGURES_DIR.mkdir(parents=True, exist_ok=True)\n",
    "\n",
    "print(f'Proyecto: {project_root}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. S4-DE-01 | Alineación de Features y Preparación de `holdout_features_selected.parquet`\n",
    "\n",
    "Para garantizar que el pipeline sea compatible con el modelo entrenado, leemos las variables RFM calculadas en holdout y las alineamos (filtramos y ordenamos) según las columnas seleccionadas oficialmente en el Sprint 3 (registradas en `06_features_selected_metadata.json`)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cargar variables RFM de holdout y la metadata del modelo\n",
    "rfm_df = pd.read_parquet(HOLDOUT_RFM_PATH)\n",
    "with open(METADATA_PATH) as f:\n",
    "    metadata = json.load(f)\n",
    "\n",
    "selected_features = metadata['selected_model_columns']\n",
    "print(f'Holdout RFM crudo: {rfm_df.shape}')\n",
    "print(f'Features seleccionadas por el modelo: {len(selected_features)}')\n",
    "\n",
    "# Comprobar que no falte ninguna feature en holdout\n",
    "missing_features = [col for col in selected_features if col not in rfm_df.columns]\n",
    "print(f'Features faltantes en holdout: {missing_features}')\n",
    "assert len(missing_features) == 0, '¡Error! Faltan features requeridas en el dataset de holdout.'\n",
    "\n",
    "# Alinear columnas: conservamos identificador, target y las features seleccionadas en el orden correcto\n",
    "aligned_columns = ['customer_unique_id', 'is_premium'] + selected_features\n",
    "holdout_selected_df = rfm_df[aligned_columns].copy()\n",
    "print(f'Dataset alineado holdout_features_selected: {holdout_selected_df.shape}')\n",
    "\n",
    "# Guardar el artefacto final\n",
    "holdout_selected_df.to_parquet(SELECTED_OUT_PATH, index=False)\n",
    "print(f'Artefacto guardado exitosamente en: {SELECTED_OUT_PATH}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. S4-DS-01 | Carga del Modelo Oficial y Validación de Incompatibilidades\n",
    "\n",
    "Cargamos el pipeline serializado en `modelo_final.pkl` y ejecutamos la inferencia sobre el dataset de holdout para validar la compatibilidad operativa de las columnas y del entorno."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cargar el modelo final serializado (.pkl)\n",
    "with open(MODEL_PATH, 'rb') as f:\n",
    "    pipeline = pickle.load(f)\n",
    "\n",
    "print('Modelo final cargado exitosamente.')\n",
    "print(pipeline)\n",
    "\n",
    "# Separar features y target para evaluación\n",
    "X_holdout = holdout_selected_df[selected_features]\n",
    "y_holdout = holdout_selected_df['is_premium']\n",
    "\n",
    "# Ejecutar la predicción de probabilidades y clases con el umbral por defecto (0.50)\n",
    "probs_holdout = pipeline.predict_proba(X_holdout)[:, 1]\n",
    "preds_holdout_50 = (probs_holdout >= 0.50).astype(int)\n",
    "\n",
    "print('\\nInferencia ejecutada con éxito.')\n",
    "print(f'Clientes evaluados en holdout: {len(X_holdout)}')\n",
    "print(f'Tasa de clientes Premium predicha (umbral 0.50): {preds_holdout_50.mean():.2%}')\n",
    "print(f'Tasa de clientes Premium real: {y_holdout.mean():.2%}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. S4-DS-01 | Preparación de Insumos Analíticos para la Demo (Muestra Demo)\n",
    "\n",
    "Definimos el criterio oficial de decisión (`predict_proba() >= 0.55`) y extraemos una muestra controlada con casos reales representativos del segmento Premium y Regular para la demostración en vivo (MVP Dashboard)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_threshold = 0.55\n",
    "preds_holdout_55 = (probs_holdout >= best_threshold).astype(int)\n",
    "\n",
    "# Unimos las probabilidades y predicciones al DataFrame\n",
    "demo_df = holdout_selected_df.copy()\n",
    "demo_df['premium_probability'] = probs_holdout\n",
    "demo_df['predicted_premium'] = preds_holdout_55\n",
    "\n",
    "# Identificamos casos:\n",
    "# - True Premium (TP): Real 1, Predicho 1 (alta confianza)\n",
    "# - True Regular (TN): Real 0, Predicho 0 (alta confianza)\n",
    "tp_cases = demo_df[(demo_df['is_premium'] == 1) & (demo_df['predicted_premium'] == 1)].sort_values('premium_probability', ascending=False).head(2)\n",
    "tn_cases = demo_df[(demo_df['is_premium'] == 0) & (demo_df['predicted_premium'] == 0)].sort_values('premium_probability', ascending=True).head(2)\n",
    "\n",
    "# Combinamos en una muestra demo de 4 casos representativos\n",
    "demo_sample = pd.concat([tp_cases, tn_cases])\n",
    "print(f'Casos Demo seleccionados (2 True Premium, 2 True Regular):')\n",
    "display_cols = ['customer_unique_id', 'is_premium', 'predicted_premium', 'premium_probability', 'total_orders', 'recency_days', 'customer_lifetime_days']\n",
    "print(demo_sample[display_cols].to_string(index=False))\n",
    "\n",
    "# Guardamos la muestra demo\n",
    "demo_sample.to_parquet(DEMO_SAMPLE_PATH, index=False)\n",
    "demo_sample[display_cols].to_csv(project_root / 'data' / 'processed' / 'demo_cases.csv', index=False)\n",
    "print(f'Muestra demo guardada en: {DEMO_SAMPLE_PATH}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. S4-DS-02 | Explicabilidad y SHAP Nativos de LightGBM\n",
    "\n",
    "Dado que no disponemos de la librería `shap` instalada, aprovechamos la capacidad nativa de LightGBM para calcular las contribuciones locales de variables (SHAP values) sobre el espacio preprocesado (una vez aplicado ColumnTransformer) de los casos demo."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "preprocessor = pipeline.named_steps['preprocessor']\n",
    "lgb_model = pipeline.named_steps['model']\n",
    "\n",
    "# Preprocesamos las características de la muestra demo\n",
    "X_demo = demo_sample[selected_features]\n",
    "X_demo_proc = preprocessor.transform(X_demo)\n",
    "feature_names_out = preprocessor.get_feature_names_out()\n",
    "X_demo_proc_df = pd.DataFrame(X_demo_proc, columns=feature_names_out, index=X_demo.index)\n",
    "\n",
    "# Calculamos las contribuciones locales (pred_contrib=True)\n",
    "# Devuelve una matriz de shape (4, n_features_out + 1)\n",
    "contribs = lgb_model.predict(X_demo_proc_df, pred_contrib=True)\n",
    "shap_values = contribs[:, :-1]\n",
    "base_value = contribs[0, -1]\n",
    "\n",
    "print(f'Valor base de decisión (log-odds): {base_value:.4f}')\n",
    "print(f'Contribuciones locales calculadas. Features preprocesadas: {len(feature_names_out)}')\n",
    "\n",
    "# Explicamos uno de los casos premium detectados\n",
    "premium_idx = 0  # Primer cliente True Premium en la muestra\n",
    "cust_id = demo_sample.iloc[premium_idx]['customer_unique_id']\n",
    "prob_cust = demo_sample.iloc[premium_idx]['premium_probability']\n",
    "\n",
    "print(f'\\nExplicación local para Cliente Premium: {cust_id} (Probabilidad: {prob_cust:.2%})')\n",
    "cust_contribs = pd.Series(shap_values[premium_idx], index=feature_names_out)\n",
    "\n",
    "# Limpiamos nombres de variables para mejor visualización\n",
    "cust_contribs.index = [col.replace('num__', '').replace('cat__', '') for col in cust_contribs.index]\n",
    "\n",
    "print('\\nPrincipales variables de empuje positivo (aportan a ser Premium):')\n",
    "print(cust_contribs.sort_values(ascending=False).head(5))\n",
    "print('\\nPrincipales variables de empuje negativo (reducen score de Premium):')\n",
    "print(cust_contribs.sort_values(ascending=True).head(5))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.1 SHAP Summary Plot (Beeswarm) Global\n",
    "\n",
    "Además de la explicación local, generamos una visual global tipo panal de abeja sobre una muestra del holdout para mostrar qué variables empujan más el score del modelo y en qué dirección. Como no usamos la librería `shap`, construimos la visual directamente con `pred_contrib=True` de LightGBM."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "sample_size = min(len(holdout_selected_df), 3000)\n",
    "holdout_sample = holdout_selected_df.sample(n=sample_size, random_state=42)\n",
    "X_sample = holdout_sample[selected_features]\n",
    "X_sample_proc = preprocessor.transform(X_sample)\n",
    "X_sample_proc_df = pd.DataFrame(X_sample_proc, columns=feature_names_out, index=X_sample.index)\n",
    "\n",
    "summary_contribs = lgb_model.predict(X_sample_proc_df, pred_contrib=True)\n",
    "summary_shap = np.asarray(summary_contribs[:, :-1])\n",
    "X_sample_proc = X_sample_proc_df.to_numpy()\n",
    "\n",
    "mean_abs_shap = np.abs(summary_shap).mean(axis=0)\n",
    "top_idx = np.argsort(mean_abs_shap)[-12:][::-1]\n",
    "clean_feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names_out]\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12, 7))\n",
    "cmap = plt.get_cmap('RdYlBu_r')\n",
    "rng = np.random.default_rng(42)\n",
    "\n",
    "for rank, idx in enumerate(top_idx):\n",
    "    shap_col = summary_shap[:, idx]\n",
    "    val_col = X_sample_proc[:, idx]\n",
    "    vmin, vmax = np.nanmin(val_col), np.nanmax(val_col)\n",
    "    if np.isclose(vmin, vmax):\n",
    "        color_values = np.full_like(val_col, 0.5, dtype=float)\n",
    "    else:\n",
    "        color_values = (val_col - vmin) / (vmax - vmin)\n",
    "    y = np.full_like(shap_col, rank, dtype=float) + rng.uniform(-0.28, 0.28, size=len(shap_col))\n",
    "    ax.scatter(shap_col, y, c=color_values, cmap=cmap, s=14, alpha=0.65, linewidths=0, rasterized=True)\n",
    "\n",
    "ax.axvline(0, color='#333333', linewidth=0.9, linestyle='--')\n",
    "ax.set_yticks(range(len(top_idx)))\n",
    "ax.set_yticklabels([clean_feature_names[idx] for idx in top_idx])\n",
    "ax.invert_yaxis()\n",
    "ax.set_xlabel('Contribución SHAP al score (log-odds)')\n",
    "ax.set_ylabel('Variables más influyentes')\n",
    "ax.set_title('SHAP Summary Plot (Beeswarm) - Holdout Sprint 4')\n",
    "ax.grid(axis='x', linestyle=':', alpha=0.25)\n",
    "\n",
    "sm = plt.cm.ScalarMappable(cmap=cmap)\n",
    "sm.set_array([0, 1])\n",
    "cbar = fig.colorbar(sm, ax=ax, pad=0.02)\n",
    "cbar.set_label('Valor relativo de la variable')\n",
    "cbar.set_ticks([0, 1])\n",
    "cbar.set_ticklabels(['Bajo', 'Alto'])\n",
    "\n",
    "plt.tight_layout()\n",
    "shap_fig_path = FIGURES_DIR / 'sprint_04_shap_beeswarm.png'\n",
    "plt.savefig(shap_fig_path, dpi=180, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(f'Gráfico SHAP beeswarm guardado en: {shap_fig_path}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4.2 Interpretación breve del SHAP Summary Plot\n",
    "\n",
    "Observamos que nuestro modelo concentra la mayor parte de su señal en variables transaccionales y de comportamiento de compra. En particular, `delivered_orders`, `max_payment_installments`, `total_items` y `top_category_is_high_value` son las variables que más empujan la predicción hacia el segmento premium cuando toman valores altos.\n",
    "\n",
    "Esta lectura es consistente con la lógica de negocio del caso: los clientes premium tienden a comprar más, completar más órdenes y participar en categorías de mayor valor. Al mismo tiempo, variables logísticas como `avg_delivery_days` y `avg_estimated_delivery_days` también muestran influencia, por lo que conviene monitorear su efecto para evitar sesgos geográficos o de infraestructura en la interpretación del modelo."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. S4-DS-03 | Consolidación de Métricas Técnicas e Impacto de Negocio\n",
    "\n",
    "Evaluamos formalmente el rendimiento técnico en holdout y traducimos estas métricas a impacto financiero acumulado, comparando una campaña de marketing masiva frente a una campaña focalizada mediante nuestro modelo."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Unimos df_eval con total_spent de la master table de holdout para calcular gasto capturado\n",
    "master_holdout = pd.read_parquet(HOLDOUT_CLEAN_PATH)\n",
    "df_eval = pd.DataFrame({\n",
    "    'customer_unique_id': holdout_selected_df['customer_unique_id'],\n",
    "    'is_premium': y_holdout,\n",
    "    'prob': probs_holdout,\n",
    "    'pred_55': preds_holdout_55\n",
    "}).merge(master_holdout[['customer_unique_id', 'total_spent']], on='customer_unique_id', how='left')\n",
    "\n",
    "# 5.1 Métricas Técnicas en Holdout (Umbral 0.55)\n",
    "auc_val = roc_auc_score(y_holdout, probs_holdout)\n",
    "gini_val = 2 * auc_val - 1\n",
    "prec_val = precision_score(y_holdout, preds_holdout_55)\n",
    "rec_val = recall_score(y_holdout, preds_holdout_55)\n",
    "f1_val = f1_score(y_holdout, preds_holdout_55)\n",
    "\n",
    "print('--- Evaluación Técnica en Holdout ---')\n",
    "print(f'ROC-AUC: {auc_val:.4f}')\n",
    "print(f'Gini: {gini_val:.4f}')\n",
    "print(f'Precision: {prec_val:.4f}')\n",
    "print(f'Recall: {rec_val:.4f}')\n",
    "print(f'F1-Score: {f1_val:.4f}')\n",
    "\n",
    "# 5.2 Distribución de la Facturación (Gasto Capturado)\n",
    "total_spend = df_eval['total_spent'].sum()\n",
    "premium_spend = df_eval[df_eval['is_premium'] == 1]['total_spent'].sum()\n",
    "predicted_premium_spend = df_eval[df_eval['pred_55'] == 1]['total_spent'].sum()\n",
    "tp_spend = df_eval[(df_eval['pred_55'] == 1) & (df_eval['is_premium'] == 1)]['total_spent'].sum()\n",
    "\n",
    "print('\\n--- Análisis de Facturación Capturada ---')\n",
    "print(f'Facturación total en Holdout: BRL {total_spend:,.2f}')\n",
    "print(f'Facturación del segmento Premium real: BRL {premium_spend:,.2f} ({premium_spend/total_spend:.2%} del total)')\n",
    "print(f'Facturación del segmento predicho como Premium: BRL {predicted_premium_spend:,.2f} ({predicted_premium_spend/total_spend:.2%} del total)')\n",
    "print(f'Facturación de Clientes Premium correctamente detectados (TP): BRL {tp_spend:,.2f} ({tp_spend/premium_spend:.2%} del gasto premium, {tp_spend/total_spend:.2%} del total)')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5.3 Simulación Financiera del Retorno de Inversión (ROI)\n",
    "\n",
    "**Hipótesis de Negocio:**\n",
    "- Costo de envío de una oferta o beneficio Premium: **BRL 15 por cliente** (campaña física/digital con kit de bienvenida).\n",
    "- Retorno neto estimado por cada cliente Premium que convierte y se fideliza: **BRL 120**.\n",
    "- Retorno neto de clientes regulares: **BRL 0**.\n",
    "\n",
    "Comparamos la política histórica de enviar campañas masivas a toda la base frente a la política optimizada con el modelo (`prob >= 0.55`)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cost_per_cust = 15\n",
    "rev_per_premium = 120\n",
    "\n",
    "n_total = len(df_eval)\n",
    "n_premium = df_eval['is_premium'].sum()\n",
    "n_pred_premium = df_eval['pred_55'].sum()\n",
    "tp_count = df_eval[(df_eval['pred_55'] == 1) & (df_eval['is_premium'] == 1)].shape[0]\n",
    "\n",
    "# Escenario A: Campaña Masiva (Target All)\n",
    "cost_all = n_total * cost_per_cust\n",
    "rev_all = n_premium * rev_per_premium\n",
    "profit_all = rev_all - cost_all\n",
    "roi_all = (profit_all / cost_all) * 100 if cost_all > 0 else 0\n",
    "\n",
    "# Escenario B: Campaña Optimizada (Target Model)\n",
    "cost_pred = n_pred_premium * cost_per_cust\n",
    "rev_pred = tp_count * rev_per_premium\n",
    "profit_pred = rev_pred - cost_pred\n",
    "roi_pred = (profit_pred / cost_pred) * 100 if cost_pred > 0 else 0\n",
    "savings = cost_all - cost_pred\n",
    "\n",
    "print('=== COMPARATIVA DE ESTRATEGIAS COMERCIALES ===')\n",
    "print(f'{\"Métrica\":<35} | {\"Masiva (All)\":<15} | {\"Optimizada (Modelo)\":<20}')\n",
    "print('-' * 77)\n",
    "print(f'{\"Clientes impactados\":<35} | {n_total:<15,} | {n_pred_premium:<20,}')\n",
    "print(f'{\"Costo total de campaña\":<35} | BRL {cost_all:<11,.2f} | BRL {cost_pred:<16,.2f}')\n",
    "print(f'{\"Clientes Premium detectados (TP)\":<35} | {n_premium:<15,} | {tp_count:<20,}')\n",
    "print(f'{\"Ingreso por conversiones\":<35} | BRL {rev_all:<11,.2f} | BRL {rev_pred:<16,.2f}')\n",
    "print(f'{\"Utilidad neta de campaña\":<35} | BRL {profit_all:<11,.2f} | BRL {profit_pred:<16,.2f}')\n",
    "print(f'{\"Retorno de Inversión (ROI)\":<35} | {roi_all:<14.2f}% | {roi_pred:<19.2f}%')\n",
    "print('-' * 77)\n",
    "print(f'Ahorro en Costo de Marketing: BRL {savings:,.2f} (Reducción del {1 - n_pred_premium/n_total:.2%})')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. S4-DS-03 | Visualizaciones para Pitch Final (Matriz de Confusión y ROI)\n",
    "\n",
    "Generamos las visualizaciones oficiales que formarán parte de la presentación final y el Dashboard para el Demo Day."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# 1. Matriz de Confusión en Holdout\n",
    "ConfusionMatrixDisplay.from_predictions(\n",
    "    y_holdout, \n",
    "    preds_holdout_55, \n",
    "    display_labels=['Regular', 'Premium'], \n",
    "    cmap='Blues', \n",
    "    ax=axes[0]\n",
    ")\n",
    "axes[0].set_title('Matriz de Confusión (Holdout - Umbral 0.55)')\n",
    "\n",
    "# 2. Comparación de ROI por Estrategia\n",
    "strategies = ['Campaña Masiva (All)', 'Campaña Modelo (Umbral 0.55)']\n",
    "rois = [roi_all, roi_pred]\n",
    "colors = ['#ff9999', '#66b3ff']\n",
    "\n",
    "bars = axes[1].bar(strategies, rois, color=colors, edgecolor='grey', width=0.5)\n",
    "axes[1].set_ylabel('ROI (%)')\n",
    "axes[1].set_title('Comparativa del Retorno de Inversión (ROI)')\n",
    "\n",
    "# Añadir etiquetas en las barras\n",
    "for bar in bars:\n",
    "    yval = bar.get_height()\n",
    "    axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + (5 if yval >= 0 else -15), f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')\n",
    "\n",
    "plt.tight_layout()\n",
    "fig_path = FIGURES_DIR / 'sprint_04_holdout_evaluation_roi.png'\n",
    "plt.savefig(fig_path, dpi=150, bbox_inches='tight')\n",
    "plt.show()\n",
    "print(f'Gráfico guardado en: {fig_path}')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook generated successfully at {notebook_path}!")
