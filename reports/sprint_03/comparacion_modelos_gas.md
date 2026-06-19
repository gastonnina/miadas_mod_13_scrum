# Comparacion de Modelos

- Cutoff temporal: `2018-07-01`
- Filas train: `83589`
- Filas validation: `6230`
- Features usadas: `28`

| Modelo | ROC-AUC Val | Gini Val | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `xgboost` | 0.7978 | 0.5956 | 0.4709 | 0.6835 | 0.5577 |
| `random_forest` | 0.7929 | 0.5858 | 0.4766 | 0.6294 | 0.5424 |
| `logistic_regression` | 0.7884 | 0.5768 | 0.4528 | 0.6442 | 0.5318 |
| `svm_linear` | 0.7872 | 0.5744 | 0.6790 | 0.2841 | 0.4006 |
| `decision_tree` | 0.7784 | 0.5568 | 0.4449 | 0.6871 | 0.5401 |
| `extra_trees` | 0.7708 | 0.5417 | 0.4429 | 0.6195 | 0.5166 |

## Mejor modelo por ROC-AUC

- Modelo: `xgboost`
- ROC-AUC validation: `0.7978`
- Matriz de confusion: `[[3716, 1092], [450, 972]]`
