"""
Data quality checks for raw Olist datasets.
All functions return plain data structures (DataFrames or dicts) — display is left to callers.
"""

import pandas as pd


def summarize_datasets(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a summary DataFrame with shape, duplicate, and null counts per dataset."""
    rows = []
    for name, df in datasets.items():
        rows.append(
            {
                "dataset": name,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "duplicated_rows": int(df.duplicated().sum()),
                "total_nulls": int(df.isna().sum().sum()),
            }
        )
    return pd.DataFrame(rows)


def get_null_report(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Return a dict of null-count DataFrames, one entry per dataset that has nulls.
    Datasets with no nulls are omitted.
    """
    report = {}
    for name, df in datasets.items():
        null_counts = df.isnull().sum()
        null_cols = null_counts[null_counts > 0].sort_values(ascending=False)
        if not null_cols.empty:
            report[name] = pd.DataFrame(
                {"Null Count": null_cols, "Proportion": null_cols / len(df)}
            )
    return report


def check_key_uniqueness(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
    order_items: pd.DataFrame,
) -> dict:
    """Verify primary key uniqueness across the main tables."""
    return {
        "customers_customer_id_unique": bool(customers["customer_id"].is_unique),
        "orders_order_id_unique": bool(orders["order_id"].is_unique),
        "products_product_id_unique": bool(products["product_id"].is_unique),
        "sellers_seller_id_unique": bool(sellers["seller_id"].is_unique),
        "order_items_composite_unique": bool(
            not order_items.duplicated(subset=["order_id", "order_item_id"]).any()
        ),
        "customers_unique_id_count": int(customers["customer_unique_id"].nunique()),
    }


def check_temporal_consistency(orders: pd.DataFrame) -> dict[str, int]:
    """
    Return counts of logical date-order violations in the orders table.
    Converts date columns to datetime internally; does not mutate the input.
    """
    df = orders.copy()
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return {
        "carrier_before_purchase": int(
            (df["order_delivered_carrier_date"] < df["order_purchase_timestamp"]).sum()
        ),
        "carrier_before_approved": int(
            (df["order_delivered_carrier_date"] < df["order_approved_at"]).sum()
        ),
        "delivered_before_carrier": int(
            (df["order_delivered_customer_date"] < df["order_delivered_carrier_date"]).sum()
        ),
        "delivered_before_approved": int(
            (df["order_delivered_customer_date"] < df["order_approved_at"]).sum()
        ),
    }
