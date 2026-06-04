from pathlib import Path

import pandas as pd

from .config import RAW_DIR

_DATASET_STEMS = {
    "customers": "olist_customers_dataset",
    "orders": "olist_orders_dataset",
    "payments": "olist_order_payments_dataset",
    "reviews": "olist_order_reviews_dataset",
    "order_items": "olist_order_items_dataset",
    "products": "olist_products_dataset",
    "sellers": "olist_sellers_dataset",
}


def load_dataset(file_stem: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load a single dataset from Parquet if available, otherwise CSV."""
    parquet_path = raw_dir / f"{file_stem}.parquet"
    csv_path = raw_dir / f"{file_stem}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Dataset not found: {file_stem} (tried {raw_dir})")


def load_raw_datasets(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load all 7 Olist raw datasets and return as a named dict."""
    return {name: load_dataset(stem, raw_dir) for name, stem in _DATASET_STEMS.items()}
