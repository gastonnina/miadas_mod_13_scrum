# Experimentos de Seleccion de Features

- Mejor experimento seleccionado: `corr_le_0.85`

| Experimento | Metodo | Threshold | # Features | AUC Train | AUC Val | Gini Train | Gini Val |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `initial` | `initial` |  | 37 | 0.8170 | 0.7938 | 0.6340 | 0.5875 |
| `missing_le_0.10` | `missing` | 0.1 | 37 | 0.8170 | 0.7938 | 0.6340 | 0.5875 |
| `corr_le_0.85` | `correlation` | 0.85 | 28 | 0.8163 | 0.7920 | 0.6326 | 0.5840 |
| `corr_le_0.90` | `correlation` | 0.9 | 29 | 0.8161 | 0.7916 | 0.6322 | 0.5832 |
| `corr_le_0.95` | `correlation` | 0.95 | 32 | 0.8173 | 0.7915 | 0.6346 | 0.5830 |
| `univariate_top_30pct` | `univariate` | 0.3 | 12 | 0.7956 | 0.7760 | 0.5912 | 0.5521 |
| `univariate_top_20pct` | `univariate` | 0.2 | 8 | 0.7869 | 0.7714 | 0.5738 | 0.5429 |
| `univariate_top_10pct` | `univariate` | 0.1 | 4 | 0.7452 | 0.7430 | 0.4904 | 0.4859 |

## Detalle de features por experimento

### `initial`

- Metodo: `initial`
- Threshold: ``
- # Features: `37`
- AUC Train: `0.8170`
- AUC Val: `0.7938`
- Gini Train: `0.6340`
- Gini Val: `0.5875`
- Features:
  - `customer_state`
  - `total_orders`
  - `total_items`
  - `total_products`
  - `avg_review_score`
  - `total_reviews`
  - `avg_delivery_days`
  - `avg_estimated_delivery_days`
  - `delivered_orders`
  - `canceled_orders`
  - `late_deliveries`
  - `payment_methods_count`
  - `max_payment_installments`
  - `avg_payment_installments`
  - `main_payment_type`
  - `recency_days`
  - `customer_lifetime_days`
  - `cancellation_rate`
  - `late_delivery_rate`
  - `top_category`
  - `items_per_order`
  - `products_per_order`
  - `max_to_avg_price_ratio`
  - `installments_gt_1_flag`
  - `installments_gt_6_flag`
  - `credit_card_flag`
  - `boleto_flag`
  - `voucher_flag`
  - `payment_complexity_flag`
  - `has_late_delivery`
  - `has_cancellation`
  - `delivery_gap`
  - `reviews_per_order`
  - `region_group`
  - `far_region_flag`
  - `top_category_group`
  - `top_category_is_high_value`

### `missing_le_0.10`

- Metodo: `missing`
- Threshold: `0.1`
- # Features: `37`
- AUC Train: `0.8170`
- AUC Val: `0.7938`
- Gini Train: `0.6340`
- Gini Val: `0.5875`
- Features:
  - `customer_state`
  - `total_orders`
  - `total_items`
  - `total_products`
  - `avg_review_score`
  - `total_reviews`
  - `avg_delivery_days`
  - `avg_estimated_delivery_days`
  - `delivered_orders`
  - `canceled_orders`
  - `late_deliveries`
  - `payment_methods_count`
  - `max_payment_installments`
  - `avg_payment_installments`
  - `main_payment_type`
  - `recency_days`
  - `customer_lifetime_days`
  - `cancellation_rate`
  - `late_delivery_rate`
  - `top_category`
  - `items_per_order`
  - `products_per_order`
  - `max_to_avg_price_ratio`
  - `installments_gt_1_flag`
  - `installments_gt_6_flag`
  - `credit_card_flag`
  - `boleto_flag`
  - `voucher_flag`
  - `payment_complexity_flag`
  - `has_late_delivery`
  - `has_cancellation`
  - `delivery_gap`
  - `reviews_per_order`
  - `region_group`
  - `far_region_flag`
  - `top_category_group`
  - `top_category_is_high_value`

### `corr_le_0.85`

- Metodo: `correlation`
- Threshold: `0.85`
- # Features: `28`
- AUC Train: `0.8163`
- AUC Val: `0.7920`
- Gini Train: `0.6326`
- Gini Val: `0.5840`
- Features:
  - `total_orders`
  - `total_items`
  - `total_products`
  - `avg_review_score`
  - `avg_delivery_days`
  - `avg_estimated_delivery_days`
  - `delivered_orders`
  - `late_deliveries`
  - `payment_methods_count`
  - `max_payment_installments`
  - `recency_days`
  - `customer_lifetime_days`
  - `cancellation_rate`
  - `products_per_order`
  - `max_to_avg_price_ratio`
  - `installments_gt_1_flag`
  - `installments_gt_6_flag`
  - `credit_card_flag`
  - `voucher_flag`
  - `delivery_gap`
  - `reviews_per_order`
  - `far_region_flag`
  - `top_category_is_high_value`
  - `customer_state`
  - `main_payment_type`
  - `top_category`
  - `region_group`
  - `top_category_group`

### `corr_le_0.90`

- Metodo: `correlation`
- Threshold: `0.9`
- # Features: `29`
- AUC Train: `0.8161`
- AUC Val: `0.7916`
- Gini Train: `0.6322`
- Gini Val: `0.5832`
- Features:
  - `total_orders`
  - `total_items`
  - `total_products`
  - `avg_review_score`
  - `total_reviews`
  - `avg_delivery_days`
  - `avg_estimated_delivery_days`
  - `delivered_orders`
  - `late_deliveries`
  - `payment_methods_count`
  - `max_payment_installments`
  - `recency_days`
  - `customer_lifetime_days`
  - `cancellation_rate`
  - `products_per_order`
  - `max_to_avg_price_ratio`
  - `installments_gt_1_flag`
  - `installments_gt_6_flag`
  - `credit_card_flag`
  - `voucher_flag`
  - `delivery_gap`
  - `reviews_per_order`
  - `far_region_flag`
  - `top_category_is_high_value`
  - `customer_state`
  - `main_payment_type`
  - `top_category`
  - `region_group`
  - `top_category_group`

### `corr_le_0.95`

- Metodo: `correlation`
- Threshold: `0.95`
- # Features: `32`
- AUC Train: `0.8173`
- AUC Val: `0.7915`
- Gini Train: `0.6346`
- Gini Val: `0.5830`
- Features:
  - `total_orders`
  - `total_items`
  - `total_products`
  - `avg_review_score`
  - `total_reviews`
  - `avg_delivery_days`
  - `avg_estimated_delivery_days`
  - `delivered_orders`
  - `late_deliveries`
  - `payment_methods_count`
  - `max_payment_installments`
  - `recency_days`
  - `customer_lifetime_days`
  - `cancellation_rate`
  - `items_per_order`
  - `products_per_order`
  - `max_to_avg_price_ratio`
  - `installments_gt_1_flag`
  - `installments_gt_6_flag`
  - `credit_card_flag`
  - `boleto_flag`
  - `voucher_flag`
  - `payment_complexity_flag`
  - `delivery_gap`
  - `reviews_per_order`
  - `far_region_flag`
  - `top_category_is_high_value`
  - `customer_state`
  - `main_payment_type`
  - `top_category`
  - `region_group`
  - `top_category_group`

### `univariate_top_30pct`

- Metodo: `univariate`
- Threshold: `0.3`
- # Features: `12`
- AUC Train: `0.7956`
- AUC Val: `0.7760`
- Gini Train: `0.5912`
- Gini Val: `0.5521`
- Features:
  - `max_payment_installments`
  - `avg_payment_installments`
  - `installments_gt_6_flag`
  - `total_items`
  - `payment_complexity_flag`
  - `top_category`
  - `installments_gt_1_flag`
  - `total_products`
  - `delivered_orders`
  - `items_per_order`
  - `max_to_avg_price_ratio`
  - `total_orders`

### `univariate_top_20pct`

- Metodo: `univariate`
- Threshold: `0.2`
- # Features: `8`
- AUC Train: `0.7869`
- AUC Val: `0.7714`
- Gini Train: `0.5738`
- Gini Val: `0.5429`
- Features:
  - `max_payment_installments`
  - `avg_payment_installments`
  - `installments_gt_6_flag`
  - `total_items`
  - `payment_complexity_flag`
  - `top_category`
  - `installments_gt_1_flag`
  - `total_products`

### `univariate_top_10pct`

- Metodo: `univariate`
- Threshold: `0.1`
- # Features: `4`
- AUC Train: `0.7452`
- AUC Val: `0.7430`
- Gini Train: `0.4904`
- Gini Val: `0.4859`
- Features:
  - `max_payment_installments`
  - `avg_payment_installments`
  - `installments_gt_6_flag`
  - `total_items`

