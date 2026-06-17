# Guia de ejecucion del MVP

## Requisitos

- Docker y Docker Compose instalados
- Estar parado en la raiz del repositorio
- Para reproducir el artefacto `modelo_final.pkl` se usa runtime Python `3.12` en los contenedores del MVP.

## Generar artefactos del Sprint 4

```bash
bash scripts/build_features.sh
```

Este comando deja listos los artefactos de scoring del holdout:

- `data/processed/holdout_features_rfm.parquet`
- `data/processed/holdout_features_selected.parquet`
- `data/processed/demo_sample_scoring.parquet`
- `data/processed/demo_cases.csv`

## Levantar el dashboard

```bash
docker compose up dashboard --build
```

O usando el script del sprint:

```bash
bash scripts/run_app.sh dashboard
```

Abrir en el navegador: `http://localhost:8501`

## Levantar dashboard y API juntos

```bash
docker compose up dashboard api --build
```

O usando el script del sprint:

```bash
bash scripts/run_app.sh all
```

| Servicio  | URL                          |
|-----------|------------------------------|
| Dashboard | http://localhost:8501        |
| API docs  | http://localhost:8000/docs   |

Si el puerto `8000` ya esta ocupado en tu maquina, levanta la API con otro puerto:

```bash
API_PORT=8001 docker compose up api --build
```

La documentacion quedara disponible en `http://localhost:8001/docs`.

## Nota sobre perfiles de Docker Compose

`pipeline` y `pipeline-cron` quedaron bajo el perfil `pipeline`, para que `docker compose up` levante por defecto solo el MVP (`dashboard` y `api`).

Si tambien quieres levantar los servicios del pipeline:

```bash
docker compose --profile pipeline up --build
```

## Detener

```bash
docker compose down
```

---

## Ejemplo de uso de la API

### Health check

```bash
curl http://localhost:8000/health
```

### Prediccion de cliente

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "total_orders": 2.0,
    "total_items": 3.0,
    "total_products": 2.0,
    "avg_review_score": 4.5,
    "avg_delivery_days": 12.0,
    "avg_estimated_delivery_days": 18.0,
    "delivered_orders": 2.0,
    "late_deliveries": 0.0,
    "payment_methods_count": 1.0,
    "max_payment_installments": 6.0,
    "recency_days": 65.0,
    "customer_lifetime_days": 0.0,
    "cancellation_rate": 0.0,
    "products_per_order": 1.5,
    "max_to_avg_price_ratio": 1.2,
    "installments_gt_1_flag": 1.0,
    "installments_gt_6_flag": 0.0,
    "credit_card_flag": 1.0,
    "voucher_flag": 0.0,
    "delivery_gap": 5.0,
    "reviews_per_order": 1.0,
    "far_region_flag": 0.0,
    "top_category_is_high_value": 1.0,
    "customer_state": "SP",
    "main_payment_type": "credit_card",
    "top_category": "informatica_acessorios",
    "region_group": "southeast",
    "top_category_group": "tech"
  }'
```

### Respuesta esperada

```json
{
  "customer_classification": "PREMIUM",
  "is_premium": true,
  "premium_probability": 0.992563,
  "threshold_used": 0.55,
  "model_version": "modelo_final.pkl"
}
```
