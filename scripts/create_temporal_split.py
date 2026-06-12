#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create temporal dev/holdout split from raw Olist parquet files.")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/splits/temporal_2018q4")
    parser.add_argument("--holdout-start", default="2018-08-01")
    parser.add_argument("--holdout-end", default="2018-10-31")
    parser.add_argument("--dev-end", default="2018-07-31")
    parser.add_argument("--exclude-holdout-customers-from-dev", default="false")
    return parser.parse_args()


def as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    holdout_start = pd.Timestamp(f"{args.holdout_start} 00:00:00")
    holdout_end = pd.Timestamp(f"{args.holdout_end} 23:59:59")
    dev_end = pd.Timestamp(f"{args.dev_end} 23:59:59")
    exclude_holdout_customers = as_bool(args.exclude_holdout_customers_from_dev)

    dev_dir = output_dir / "dev"
    holdout_dir = output_dir / "holdout_3m"
    dev_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    orders = pd.read_parquet(input_dir / "olist_orders_dataset.parquet")
    items = pd.read_parquet(input_dir / "olist_order_items_dataset.parquet")
    payments = pd.read_parquet(input_dir / "olist_order_payments_dataset.parquet")
    reviews = pd.read_parquet(input_dir / "olist_order_reviews_dataset.parquet")
    customers = pd.read_parquet(input_dir / "olist_customers_dataset.parquet")

    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

    # Primer corte temporal del proyecto:
    # dev = historico hasta julio 2018, holdout = agosto a octubre 2018.
    orders_dev = orders[orders["order_purchase_timestamp"] <= dev_end].copy()
    orders_holdout = orders[
        (orders["order_purchase_timestamp"] >= holdout_start)
        & (orders["order_purchase_timestamp"] <= holdout_end)
    ].copy()

    holdout_order_ids = set(orders_holdout["order_id"])
    dev_order_ids = set(orders_dev["order_id"])

    items_dev = items[items["order_id"].isin(dev_order_ids)].copy()
    items_holdout = items[items["order_id"].isin(holdout_order_ids)].copy()

    payments_dev = payments[payments["order_id"].isin(dev_order_ids)].copy()
    payments_holdout = payments[payments["order_id"].isin(holdout_order_ids)].copy()

    reviews_dev = reviews[reviews["order_id"].isin(dev_order_ids)].copy()
    reviews_holdout = reviews[reviews["order_id"].isin(holdout_order_ids)].copy()

    orders_holdout_customers = orders_holdout[["order_id", "customer_id"]].merge(
        customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left",
    )
    holdout_customer_ids = (
        orders_holdout_customers["customer_unique_id"].dropna().drop_duplicates().sort_values()
    )

    if exclude_holdout_customers:
        # Modo mas estricto: si un cliente aparece en holdout, se elimina por
        # completo de dev para evitar compartir identidad entre ventanas.
        holdout_customer_id_set = set(holdout_customer_ids)
        orders_dev = orders_dev.merge(
            customers[["customer_id", "customer_unique_id"]],
            on="customer_id",
            how="left",
        )
        orders_dev = orders_dev[~orders_dev["customer_unique_id"].isin(holdout_customer_id_set)].copy()
        orders_dev = orders_dev.drop(columns=["customer_unique_id"])
        dev_order_ids = set(orders_dev["order_id"])
        items_dev = items[items["order_id"].isin(dev_order_ids)].copy()
        payments_dev = payments[payments["order_id"].isin(dev_order_ids)].copy()
        reviews_dev = reviews[reviews["order_id"].isin(dev_order_ids)].copy()

    orders_dev.to_parquet(dev_dir / "orders.parquet", index=False)
    orders_holdout.to_parquet(holdout_dir / "orders.parquet", index=False)
    items_dev.to_parquet(dev_dir / "order_items.parquet", index=False)
    items_holdout.to_parquet(holdout_dir / "order_items.parquet", index=False)
    payments_dev.to_parquet(dev_dir / "order_payments.parquet", index=False)
    payments_holdout.to_parquet(holdout_dir / "order_payments.parquet", index=False)
    reviews_dev.to_parquet(dev_dir / "order_reviews.parquet", index=False)
    reviews_holdout.to_parquet(holdout_dir / "order_reviews.parquet", index=False)

    holdout_customer_ids.to_frame(name="customer_unique_id").to_parquet(
        output_dir / "holdout_3m_ids.parquet",
        index=False,
    )

    metadata = {
        "run_timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "cutoff_start_holdout": args.holdout_start,
        "cutoff_end_holdout": args.holdout_end,
        "dev_end_date": args.dev_end,
        "exclude_holdout_customers_from_dev": exclude_holdout_customers,
        "rows_orders_dev": int(len(orders_dev)),
        "rows_orders_holdout": int(len(orders_holdout)),
        "rows_items_dev": int(len(items_dev)),
        "rows_items_holdout": int(len(items_holdout)),
        "rows_payments_dev": int(len(payments_dev)),
        "rows_payments_holdout": int(len(payments_holdout)),
        "rows_reviews_dev": int(len(reviews_dev)),
        "rows_reviews_holdout": int(len(reviews_holdout)),
        "customers_holdout": int(len(holdout_customer_ids)),
    }

    (output_dir / "holdout_3m_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Temporal split created:")
    print(f"- output_dir: {output_dir}")
    print(f"- orders_dev: {len(orders_dev)}")
    print(f"- orders_holdout: {len(orders_holdout)}")
    print(f"- holdout_customers: {len(holdout_customer_ids)}")


if __name__ == "__main__":
    main()
