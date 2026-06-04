# Experimentos de Seleccion de Features

- Mejor experimento seleccionado: `corr_le_0.85`

| Experimento | Metodo | Threshold | # Features | AUC Train | AUC Val | Gini Train | Gini Val |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `initial` | `initial` |  | 37 | 0.8172 | 0.7939 | 0.6344 | 0.5878 |
| `missing_le_0.10` | `missing` | 0.1 | 37 | 0.8172 | 0.7939 | 0.6344 | 0.5878 |
| `corr_le_0.90` | `correlation` | 0.9 | 29 | 0.8152 | 0.7923 | 0.6304 | 0.5846 |
| `corr_le_0.95` | `correlation` | 0.95 | 32 | 0.8170 | 0.7922 | 0.6340 | 0.5845 |
| `corr_le_0.85` | `correlation` | 0.85 | 28 | 0.8172 | 0.7921 | 0.6344 | 0.5843 |
| `univariate_top_30pct` | `univariate` | 0.3 | 12 | 0.7941 | 0.7752 | 0.5882 | 0.5503 |
| `univariate_top_20pct` | `univariate` | 0.2 | 8 | 0.7872 | 0.7686 | 0.5743 | 0.5372 |
| `univariate_top_10pct` | `univariate` | 0.1 | 4 | 0.7452 | 0.7430 | 0.4904 | 0.4860 |

## Detalle de features por experimento

### `initial`

- Metodo: `initial`
- Threshold: ``
- # Features: `37`
- AUC Train: `0.8172`
- AUC Val: `0.7939`
- Gini Train: `0.6344`
- Gini Val: `0.5878`
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
- AUC Train: `0.8172`
- AUC Val: `0.7939`
- Gini Train: `0.6344`
- Gini Val: `0.5878`
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

### `corr_le_0.90`

- Metodo: `correlation`
- Threshold: `0.9`
- # Features: `29`
- AUC Train: `0.8152`
- AUC Val: `0.7923`
- Gini Train: `0.6304`
- Gini Val: `0.5846`
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
- AUC Train: `0.8170`
- AUC Val: `0.7922`
- Gini Train: `0.6340`
- Gini Val: `0.5845`
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

### `corr_le_0.85`

- Metodo: `correlation`
- Threshold: `0.85`
- # Features: `28`
- AUC Train: `0.8172`
- AUC Val: `0.7921`
- Gini Train: `0.6344`
- Gini Val: `0.5843`
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

### `univariate_top_30pct`

- Metodo: `univariate`
- Threshold: `0.3`
- # Features: `12`
- AUC Train: `0.7941`
- AUC Val: `0.7752`
- Gini Train: `0.5882`
- Gini Val: `0.5503`
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
- AUC Train: `0.7872`
- AUC Val: `0.7686`
- Gini Train: `0.5743`
- Gini Val: `0.5372`
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
- Gini Val: `0.4860`
- Features:
  - `max_payment_installments`
  - `avg_payment_installments`
  - `installments_gt_6_flag`
  - `total_items`

