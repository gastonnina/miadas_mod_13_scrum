# Tuning de Hiperparámetros — Sprint 3

**Método:** Optuna TPE — 50 trials por modelo
**Tareas:** S3-DS-03 (LightGBM) + S3-DS-04 (XGBoost)

---

## 1. Resultados de Optimización

| Modelo | ROC-AUC val | Gini val | F1 val | Recall val | Gap overfit |
|---|---|---|---|---|---|
| LightGBM baseline | 0.8035 | 0.6070 | 0.5565 | 0.6653 | 0.0370 |
| **LightGBM tuneado** | **0.8081** | **0.6162** | **0.5576** | **0.6617** | 0.0385 |
| XGBoost baseline | 0.7982 | 0.5964 | 0.5504 | 0.6906 | 0.0290 |
| **XGBoost tuneado** | **0.8070** | **0.6140** | **0.5560** | **0.6371** | 0.0525 |

## 2. Mejoras Obtenidas

- LightGBM: ROC-AUC +0.0046 | F1 +0.0012
- XGBoost:  ROC-AUC +0.0088  | F1 +0.0056

## 3. Modelo Final: LightGBM (tuneado)

**Archivo:** `C:\Users\marce\Documents\Code\Maestria\m13\miadas_mod_13_scrum\models\final\modelo_final.pkl`  
**ROC-AUC val:** 0.8081  
**Gap overfit:** 0.0385

### Hiperparámetros óptimos

```json
{
  "num_leaves": 59,
  "max_depth": 8,
  "learning_rate": 0.054038945566405934,
  "n_estimators": 255,
  "min_child_samples": 68,
  "subsample": 0.9059228577785707,
  "colsample_bytree": 0.855428530082444,
  "reg_alpha": 6.735979730583029e-08,
  "reg_lambda": 0.0005641632739363156
}
```

## 4. Umbral Óptimo

Umbral con mayor F1 = **0.55**
- Precision: 0.5225
- Recall:    0.6041
- F1:        0.5603

## 5. Figuras generadas

- `sprint_03_tuning_comparison.png` — Baseline vs. tuneado por métrica
- `sprint_03_optimization_history.png` — Historia de trials Optuna
- `sprint_03_roc_pr_curves.png` — Curvas ROC y PR
- `sprint_03_feature_importance.png` — Top 20 features por modelo
- `sprint_03_threshold_analysis.png` — Análisis de umbral

---

*Generado desde `notebooks/sprint_03_modeling/02_hyperparameter_tuning.ipynb`*