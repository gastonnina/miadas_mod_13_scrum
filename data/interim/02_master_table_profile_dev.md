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
| `avg_delivery_days` | `float64` | 8884 | 9.24% | 245 |
| `avg_freight_ratio` | `float64` | 8882 | 9.24% | 41587 |
| `avg_freight_value` | `float64` | 8882 | 9.24% | 8664 |
| `avg_order_price` | `float64` | 8882 | 9.24% | 8369 |
| `avg_ticket` | `float64` | 8883 | 9.24% | 27317 |
| `total_spent` | `float64` | 8882 | 9.24% | 27163 |
| `top_category` | `str` | 8205 | 8.54% | 73 |
| `avg_review_score` | `float64` | 6966 | 7.25% | 35 |
| `avg_item_price` | `float64` | 6914 | 7.19% | 7795 |
| `max_item_price` | `float64` | 6914 | 7.19% | 5528 |
| `avg_estimated_delivery_days` | `float64` | 6277 | 6.53% | 209 |
| `avg_payment_installments` | `float64` | 6278 | 6.53% | 69 |
| `canceled_orders` | `float64` | 6277 | 6.53% | 2 |
| `cancellation_rate` | `float64` | 6277 | 6.53% | 4 |
| `customer_lifetime_days` | `float64` | 6277 | 6.53% | 401 |

## Riesgos observados

- Los nulos se concentran en clientes sin pedidos o sin entrega efectiva en la ventana `dev`.
- No hay columnas con variabilidad nula.
- Las fechas y columnas de monto requieren control de leakage antes de modelado.
- Watchlist de leakage inicial: `total_spent, avg_ticket, avg_order_price, avg_item_price, max_item_price`.

## Recomendacion para Fase 4

- Mantener limpieza minima para conteos y sumas con imputacion a `0`.
- No imputar ciegamente columnas de catalogo o comportamiento de pago sin justificacion.
- Excluir del modelado directo las variables pegadas a gasto total hasta cerrar la revision metodologica.
