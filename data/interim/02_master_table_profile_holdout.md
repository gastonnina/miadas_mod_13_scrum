# Perfil de Master Table Raw Sprint 2

## Resumen

- Filas: `96096`
- Columnas: `32`
- Duplicados por `customer_unique_id`: `0`
- Columnas numericas: `25`
- Columnas categoricas: `0`
- Columnas datetime: `2`

## Columnas por tipo

- Numericas: `customer_zip_code_prefix, total_orders, total_items, total_products, avg_review_score, total_reviews, avg_delivery_days, avg_estimated_delivery_days, delivered_orders, canceled_orders, late_deliveries, payment_methods_count, max_payment_installments, avg_payment_installments, total_spent, avg_ticket, avg_order_price, avg_freight_value, avg_freight_ratio, recency_days, customer_lifetime_days, cancellation_rate, late_delivery_rate, max_item_price, avg_item_price`
- Categoricas: ``
- Datetime: `first_purchase, last_purchase`

## Columnas con mas nulos

| Columna | Tipo | Nulos | % Nulos | Unicos |
| --- | --- | ---: | ---: | ---: |
| `avg_delivery_days` | `float64` | 89786 | 93.43% | 49 |
| `avg_freight_ratio` | `float64` | 89786 | 93.43% | 4240 |
| `avg_freight_value` | `float64` | 89786 | 93.43% | 2328 |
| `avg_order_price` | `float64` | 89786 | 93.43% | 1658 |
| `avg_ticket` | `float64` | 89786 | 93.43% | 4109 |
| `total_spent` | `float64` | 89786 | 93.43% | 4113 |
| `top_category` | `str` | 89713 | 93.36% | 65 |
| `avg_item_price` | `float64` | 89685 | 93.33% | 1549 |
| `max_item_price` | `float64` | 89685 | 93.33% | 1424 |
| `avg_review_score` | `float64` | 89656 | 93.30% | 9 |
| `avg_estimated_delivery_days` | `float64` | 89628 | 93.27% | 80 |
| `avg_payment_installments` | `float64` | 89628 | 93.27% | 25 |
| `canceled_orders` | `float64` | 89628 | 93.27% | 4 |
| `cancellation_rate` | `float64` | 89628 | 93.27% | 4 |
| `customer_lifetime_days` | `float64` | 89628 | 93.27% | 24 |

## Riesgos observados

- Los nulos se concentran en clientes sin pedidos o sin entrega efectiva en la ventana `dev`.
- No hay columnas con variabilidad nula.
- Las fechas y columnas de monto requieren control de leakage antes de modelado.
- Watchlist de leakage inicial: `total_spent, avg_ticket, avg_order_price, avg_item_price, max_item_price`.

## Recomendacion para Fase 4

- Mantener limpieza minima para conteos y sumas con imputacion a `0`.
- No imputar ciegamente columnas de catalogo o comportamiento de pago sin justificacion.
- Excluir del modelado directo las variables pegadas a gasto total hasta cerrar la revision metodologica.
