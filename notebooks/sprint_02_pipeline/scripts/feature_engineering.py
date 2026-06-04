"""
Core feature engineering transforms: order-level aggregations → customer-level master table.

All functions are pure (no side effects) and accept/return DataFrames so they can be
unit-tested independently and reused in a Spark context (replace pandas groupby with
pyspark equivalents on the same logic).
"""

import numpy as np
import pandas as pd

_DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

_ZERO_FILL_COLS = [
    "total_spent",
    "total_orders",
    "total_items",
    "total_products",
    "total_reviews",
    "delivered_orders",
    "canceled_orders",
    "late_deliveries",
    "cancellation_rate",
    "late_delivery_rate",
]


def parse_order_dates(orders: pd.DataFrame) -> pd.DataFrame:
    df = orders.copy()
    for col in _DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def compute_order_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate payment metrics to one row per order_id."""
    return payments.groupby("order_id", as_index=False).agg(
        order_payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        payment_methods_count=("payment_type", "nunique"),
        main_payment_type=(
            "payment_type",
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        ),
    )


def compute_order_item_features(order_items: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item/price/freight metrics to one row per order_id."""
    df = order_items.groupby("order_id", as_index=False).agg(
        order_items_count=("order_item_id", "count"),
        order_products_count=("product_id", "nunique"),
        order_price_total=("price", "sum"),
        order_freight_total=("freight_value", "sum"),
        sellers_count=("seller_id", "nunique"),
    )
    df["freight_ratio"] = (
        df["order_freight_total"] / (df["order_price_total"] + df["order_freight_total"])
    ).replace([np.inf, -np.inf], np.nan)
    return df


def compute_order_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Aggregate review score to one row per order_id."""
    return reviews.groupby("order_id", as_index=False).agg(
        review_score=("review_score", "mean"),
        review_count=("review_id", "nunique"),
    )


def enrich_orders(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    order_payments: pd.DataFrame,
    order_item_features: pd.DataFrame,
    order_reviews: pd.DataFrame,
) -> pd.DataFrame:
    """Join order-level aggregations and derive delivery/status flags."""
    df = (
        orders.merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="inner")
        .merge(order_payments, on="order_id", how="left")
        .merge(order_item_features, on="order_id", how="left")
        .merge(order_reviews, on="order_id", how="left")
    )
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["is_delivered"] = np.where(df["order_status"].eq("delivered"), 1, 0)
    df["is_canceled"] = np.where(df["order_status"].eq("canceled"), 1, 0)
    df["is_late_delivery"] = np.where(
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"], 1, 0
    )
    return df


def aggregate_customers(orders_enriched: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order-level data to one row per customer_unique_id."""
    reference_date = orders_enriched["order_purchase_timestamp"].max()

    # Financial metrics: only delivered orders reflect real net revenue
    customer_financials = (
        orders_enriched[orders_enriched["order_status"].eq("delivered")]
        .groupby("customer_unique_id", as_index=False)
        .agg(
            total_spent=("order_payment_value", "sum"),
            avg_ticket=("order_payment_value", "mean"),
            avg_order_price=("order_price_total", "mean"),
            avg_freight_value=("order_freight_total", "mean"),
            avg_freight_ratio=("freight_ratio", "mean"),
        )
    )

    # Operational/volume metrics: all orders including canceled
    customer_general = orders_enriched.groupby("customer_unique_id", as_index=False).agg(
        total_orders=("order_id", "nunique"),
        total_items=("order_items_count", "sum"),
        total_products=("order_products_count", "sum"),
        avg_review_score=("review_score", "mean"),
        total_reviews=("review_count", "sum"),
        first_purchase=("order_purchase_timestamp", "min"),
        last_purchase=("order_purchase_timestamp", "max"),
        avg_delivery_days=("delivery_days", "mean"),
        avg_estimated_delivery_days=("estimated_delivery_days", "mean"),
        delivered_orders=("is_delivered", "sum"),
        canceled_orders=("is_canceled", "sum"),
        late_deliveries=("is_late_delivery", "sum"),
        payment_methods_count=("payment_methods_count", "max"),
        main_payment_type=(
            "main_payment_type",
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
        ),
    )

    features = customer_general.merge(customer_financials, on="customer_unique_id", how="left")
    features["recency_days"] = (reference_date - features["last_purchase"]).dt.days
    features["customer_lifetime_days"] = (
        features["last_purchase"] - features["first_purchase"]
    ).dt.days
    features["cancellation_rate"] = features["canceled_orders"] / features["total_orders"]
    features["late_delivery_rate"] = features["late_deliveries"] / features["total_orders"]
    return features


def integrate_geo(customers: pd.DataFrame, customer_features: pd.DataFrame) -> pd.DataFrame:
    """Left-join geographic columns and zero-fill count/sum columns for customers with no orders."""
    customer_geo = customers[
        ["customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"]
    ].drop_duplicates(subset=["customer_unique_id"], keep="last")

    master = customer_geo.merge(customer_features, on="customer_unique_id", how="left")
    master[_ZERO_FILL_COLS] = master[_ZERO_FILL_COLS].fillna(0)
    return master


def add_premium_target(
    master_table: pd.DataFrame, quantile: float = 0.80
) -> tuple[pd.DataFrame, float]:
    """Label customers in the top-`quantile` of total_spent as premium (1)."""
    threshold = master_table["total_spent"].quantile(quantile)
    df = master_table.copy()
    df["is_premium"] = np.where(df["total_spent"] >= threshold, 1, 0)
    return df, threshold


def build_master_table(
    datasets: dict[str, pd.DataFrame], premium_quantile: float = 0.80
) -> tuple[pd.DataFrame, float]:
    """
    Full pipeline: raw datasets → customer-level master table with is_premium label.

    Parameters
    ----------
    datasets : dict with keys customers, orders, payments, reviews, order_items
    premium_quantile : percentile threshold for the is_premium label (default 0.80)

    Returns
    -------
    (master_table DataFrame, premium_threshold float)
    """
    orders = parse_order_dates(datasets["orders"])
    order_payments = compute_order_payments(datasets["payments"])
    item_features = compute_order_item_features(datasets["order_items"])
    order_reviews = compute_order_reviews(datasets["reviews"])
    orders_enriched = enrich_orders(
        orders, datasets["customers"], order_payments, item_features, order_reviews
    )
    customer_features = aggregate_customers(orders_enriched)
    master = integrate_geo(datasets["customers"], customer_features)
    master, threshold = add_premium_target(master, quantile=premium_quantile)
    return master, threshold
