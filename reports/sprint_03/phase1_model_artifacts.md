# Fase 1 - Modelos e Importancias

- Cutoff validacion: `2018-07-01`
- Filas train: `83589`
- Filas validation: `6230`
- Features seleccionadas: `28`

| Modelo | ROC-AUC val | Gini val | F1 val | Recall val | Gap overfit |
| --- | ---: | ---: | ---: | ---: | ---: |
| lightgbm_baseline | 0.802465 | 0.604930 | 0.556888 | 0.669480 | 0.0381 |
| lightgbm_tuned | 0.808117 | 0.616235 | 0.557630 | 0.661744 | 0.0385 |
| xgboost_baseline | 0.798919 | 0.597838 | 0.554953 | 0.683544 | 0.0281 |
| xgboost_tuned | 0.806694 | 0.613388 | 0.559758 | 0.650492 | 0.0535 |

## Artefactos

- `models/baseline/lightgbm_baseline.pkl`
- `models/baseline/xgboost_baseline.pkl`
- `models/final/lightgbm_tuned.pkl`
- `models/final/xgboost_tuned.pkl`
- `data/processed/13_feature_audit_lightgbm.parquet`
- `data/processed/13_feature_audit_xgboost.parquet`
