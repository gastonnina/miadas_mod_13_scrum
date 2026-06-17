# Checklist Final de Cierre - Sprint 4

## Consistencia funcional

- [x] El pipeline genera `data/processed/holdout_features_rfm.parquet`
- [x] El pipeline genera `data/processed/holdout_features_selected.parquet`
- [x] El dashboard consume `modelo_final.pkl` y `holdout_features_selected.parquet`
- [x] La API mínima expone `/health` y `/predict`
- [x] Docker del dashboard construye correctamente
- [x] Docker de la API construye correctamente
- [x] `modelo_final.pkl` carga dentro de ambos contenedores

## Consistencia analítica

- [x] Existe muestra demo reutilizable en `data/processed/demo_sample_scoring.parquet`
- [x] Existe tabla corta de casos demo en `data/processed/demo_cases.csv`
- [x] Existe gráfico de ROI en `reports/figures/sprint_04_holdout_evaluation_roi.png`
- [x] Existe SHAP summary plot global en `reports/figures/sprint_04_shap_beeswarm.png`
- [x] El notebook `notebooks/sprint_04_integration/04_demo_validation.ipynb` contiene interpretación del SHAP
- [x] Las métricas del reporte y la app usan el mismo escenario holdout con umbral `0.55`

## Consistencia documental

- [x] Informe ético y de gobernanza: `reports/sprint_04/informe_etico_gobernanza.md`
- [x] Pitch Demo Day: `reports/sprint_04/pitch_demo_day.md`
- [x] Guía de ejecución MVP: `reports/sprint_04/guia_ejecucion_mvp.md`
- [x] Reporte científico de transición: `reports/sprint_04/reporte_cientifico_datos_sprint_4.md`

## Mensajes clave alineados

- [x] Problema: campañas masivas generan alto costo y bajo retorno
- [x] Solución: clasificación premium con `LightGBM` sobre pipeline reproducible
- [x] Evidencia: ROC-AUC `0.9872`, Gini `0.9745`, SHAP local y global
- [x] Impacto: ROI masivo `-89.57%` vs ROI modelo `+244.79%`
- [x] Gobernanza: supervisión humana, versionado y monitoreo de sesgos

## Pendientes de último momento

- [ ] Ejecutar ensayo final entre los 3 integrantes
- [ ] Confirmar quién presenta cada bloque en la defensa real
- [ ] Preparar capturas de respaldo por si falla la demo en vivo

## Dictamen final

No se detectan contradicciones de fondo entre pipeline, app, métricas, explicabilidad, informe ético y pitch. El sprint queda listo para defensa, sujeto únicamente al ensayo final del equipo y a la preparación de respaldo visual para la demo.
