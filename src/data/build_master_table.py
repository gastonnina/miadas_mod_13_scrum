from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

SPLIT_NAME_MAP = {
    "olist_orders_dataset": "orders",
    "olist_order_items_dataset": "order_items",
    "olist_order_payments_dataset": "order_payments",
    "olist_order_reviews_dataset": "order_reviews",
}

FILL_ZERO_COLUMNS = [
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


@dataclass(frozen=True)
class BuildConfig:
    project_root: Path
    profile_source: str = "dev"

    @property
    def raw_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def split_dev_dir(self) -> Path:
        return self.project_root / "data" / "splits" / "temporal_2018q4" / "dev"

    @property
    def processed_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def interim_dir(self) -> Path:
        return self.project_root / "data" / "interim"

    def validate(self) -> None:
        if self.profile_source not in {"raw", "dev"}:
            raise ValueError("profile_source debe ser 'raw' o 'dev'")


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_dataset(config: BuildConfig, file_stem: str, force_raw: bool = False) -> pd.DataFrame:
    source_dir = config.raw_dir if force_raw else (
        config.raw_dir if config.profile_source == "raw" else config.split_dev_dir
    )

    effective_stem = file_stem
    if not force_raw and config.profile_source == "dev":
        effective_stem = SPLIT_NAME_MAP.get(file_stem, file_stem)

    parquet_path = source_dir / f"{effective_stem}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    raise FileNotFoundError(
        f"Dataset not found: {file_stem} (effective: {effective_stem}) in {source_dir}"
    )


def load_all_datasets(config: BuildConfig) -> dict[str, pd.DataFrame]:
    return {
        "customers": load_dataset(config, "olist_customers_dataset", force_raw=True),
        "orders": load_dataset(config, "olist_orders_dataset"),
        "payments": load_dataset(config, "olist_order_payments_dataset"),
        "reviews": load_dataset(config, "olist_order_reviews_dataset"),
        "order_items": load_dataset(config, "olist_order_items_dataset"),
        "products": load_dataset(config, "olist_products_dataset", force_raw=True),
        "sellers": load_dataset(config, "olist_sellers_dataset", force_raw=True),
    }


def _mode_or_nan(values: pd.Series) -> object:
    mode = values.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else np.nan


def _prepare_orders(orders: pd.DataFrame) -> pd.DataFrame:
    prepared = orders.copy()
    for column in DATE_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(prepared[column], errors="coerce")
    return prepared


def build_orders_enriched(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    customers = datasets["customers"]
    orders = _prepare_orders(datasets["orders"])
    payments = datasets["payments"]
    reviews = datasets["reviews"]
    order_items = datasets["order_items"]

    order_payments = (
        payments.groupby("order_id", as_index=False)
        .agg(
            order_payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_methods_count=("payment_type", "nunique"),
            main_payment_type=("payment_type", _mode_or_nan),
        )
    )

    order_item_features = (
        order_items.groupby("order_id", as_index=False)
        .agg(
            order_items_count=("order_item_id", "count"),
            order_products_count=("product_id", "nunique"),
            order_price_total=("price", "sum"),
            order_freight_total=("freight_value", "sum"),
            sellers_count=("seller_id", "nunique"),
        )
    )
    order_item_features["freight_ratio"] = (
        order_item_features["order_freight_total"]
        / (order_item_features["order_price_total"] + order_item_features["order_freight_total"])
    ).replace([np.inf, -np.inf], np.nan)

    order_reviews = (
        reviews.groupby("order_id", as_index=False)
        .agg(
            review_score=("review_score", "mean"),
            review_count=("review_id", "nunique"),
        )
    )

    orders_enriched = (
        orders.merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="inner")
        .merge(order_payments, on="order_id", how="left")
        .merge(order_item_features, on="order_id", how="left")
        .merge(order_reviews, on="order_id", how="left")
    )

    orders_enriched["delivery_days"] = (
        orders_enriched["order_delivered_customer_date"] - orders_enriched["order_purchase_timestamp"]
    ).dt.days
    orders_enriched["estimated_delivery_days"] = (
        orders_enriched["order_estimated_delivery_date"] - orders_enriched["order_purchase_timestamp"]
    ).dt.days
    orders_enriched["is_delivered"] = np.where(orders_enriched["order_status"].eq("delivered"), 1, 0)
    orders_enriched["is_canceled"] = np.where(orders_enriched["order_status"].eq("canceled"), 1, 0)
    orders_enriched["is_late_delivery"] = np.where(
        orders_enriched["order_delivered_customer_date"] > orders_enriched["order_estimated_delivery_date"],
        1,
        0,
    )

    return orders_enriched


def build_customer_features(orders_enriched: pd.DataFrame) -> pd.DataFrame:
    reference_date = orders_enriched["order_purchase_timestamp"].max()

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

    customer_general = (
        orders_enriched.groupby("customer_unique_id", as_index=False)
        .agg(
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
            max_payment_installments=("payment_installments", "max"),
            avg_payment_installments=("payment_installments", "mean"),
            main_payment_type=("main_payment_type", _mode_or_nan),
        )
    )

    customer_features = customer_general.merge(customer_financials, on="customer_unique_id", how="left")
    customer_features["recency_days"] = (
        reference_date - customer_features["last_purchase"]
    ).dt.days
    customer_features["customer_lifetime_days"] = (
        customer_features["last_purchase"] - customer_features["first_purchase"]
    ).dt.days
    customer_features["cancellation_rate"] = (
        customer_features["canceled_orders"] / customer_features["total_orders"]
    )
    customer_features["late_delivery_rate"] = (
        customer_features["late_deliveries"] / customer_features["total_orders"]
    )

    return customer_features


def build_customer_catalog(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    order_items = datasets["order_items"]
    products = datasets["products"]
    orders = datasets["orders"]
    customers = datasets["customers"]

    return (
        order_items.merge(products[["product_id", "product_category_name"]], on="product_id", how="left")
        .merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
        .merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left")
        .groupby("customer_unique_id", as_index=False)
        .agg(
            max_item_price=("price", "max"),
            avg_item_price=("price", "mean"),
            top_category=("product_category_name", _mode_or_nan),
        )
    )


def build_customer_geo(customers: pd.DataFrame) -> pd.DataFrame:
    return customers[
        [
            "customer_unique_id",
            "customer_zip_code_prefix",
            "customer_city",
            "customer_state",
        ]
    ].drop_duplicates(subset=["customer_unique_id"], keep="last")


def add_target(master_table: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    threshold = float(master_table["total_spent"].quantile(0.80))
    result = master_table.copy()
    result["is_premium"] = np.where(result["total_spent"] >= threshold, 1, 0)
    return result, threshold


def apply_fixed_target(master_table: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = master_table.copy()
    result["is_premium"] = np.where(result["total_spent"] >= threshold, 1, 0)
    return result


def build_master_table(
    config: BuildConfig,
    fixed_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    config.validate()
    datasets = load_all_datasets(config)
    orders_enriched = build_orders_enriched(datasets)
    customer_features = build_customer_features(orders_enriched)
    customer_catalog = build_customer_catalog(datasets)
    customer_geo = build_customer_geo(datasets["customers"])

    master_table_raw = (
        customer_geo.merge(customer_features, on="customer_unique_id", how="left")
        .merge(customer_catalog, on="customer_unique_id", how="left")
    )

    master_table_clean = master_table_raw.copy()
    master_table_clean[FILL_ZERO_COLUMNS] = master_table_clean[FILL_ZERO_COLUMNS].fillna(0)
    if fixed_threshold is None:
        master_table_clean, threshold = add_target(master_table_clean)
    else:
        threshold = float(fixed_threshold)
        master_table_clean = apply_fixed_target(master_table_clean, threshold)

    return master_table_raw, master_table_clean, threshold


def validate_master_table(master_table: pd.DataFrame) -> dict[str, float | int]:
    return {
        "rows": int(master_table.shape[0]),
        "columns": int(master_table.shape[1]),
        "duplicated_customer_unique_id": int(master_table["customer_unique_id"].duplicated().sum()),
        "null_customer_unique_id": int(master_table["customer_unique_id"].isna().sum()),
        "null_total_spent": int(master_table["total_spent"].isna().sum()),
        "null_total_orders": int(master_table["total_orders"].isna().sum()),
        "premium_rate": float(master_table["is_premium"].mean()) if "is_premium" in master_table else np.nan,
    }


def build_profile_markdown(master_table_raw: pd.DataFrame) -> str:
    profile = pd.DataFrame(
        {
            "column": master_table_raw.columns,
            "dtype": master_table_raw.dtypes.astype(str).values,
            "nulls": master_table_raw.isna().sum().values,
            "null_pct": (master_table_raw.isna().mean() * 100).round(2).values,
            "n_unique": master_table_raw.nunique(dropna=True).values,
        }
    ).sort_values(["null_pct", "column"], ascending=[False, True])

    numeric_columns = [
        col for col in master_table_raw.columns if pd.api.types.is_numeric_dtype(master_table_raw[col])
    ]
    categorical_columns = [
        col for col in master_table_raw.columns if pd.api.types.is_object_dtype(master_table_raw[col])
    ]
    datetime_columns = [
        col for col in master_table_raw.columns if pd.api.types.is_datetime64_any_dtype(master_table_raw[col])
    ]
    low_variability_columns = profile.loc[profile["n_unique"] <= 1, "column"].tolist()

    leakage_watchlist = [
        "total_spent",
        "avg_ticket",
        "avg_order_price",
        "avg_item_price",
        "max_item_price",
    ]

    lines = [
        "# Perfil de Master Table Raw Sprint 2",
        "",
        "## Resumen",
        "",
        f"- Filas: `{master_table_raw.shape[0]}`",
        f"- Columnas: `{master_table_raw.shape[1]}`",
        f"- Duplicados por `customer_unique_id`: `{int(master_table_raw['customer_unique_id'].duplicated().sum())}`",
        f"- Columnas numericas: `{len(numeric_columns)}`",
        f"- Columnas categoricas: `{len(categorical_columns)}`",
        f"- Columnas datetime: `{len(datetime_columns)}`",
        "",
        "## Columnas por tipo",
        "",
        f"- Numericas: `{', '.join(numeric_columns)}`",
        f"- Categoricas: `{', '.join(categorical_columns)}`",
        f"- Datetime: `{', '.join(datetime_columns)}`",
        "",
        "## Columnas con mas nulos",
        "",
        "| Columna | Tipo | Nulos | % Nulos | Unicos |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for row in profile.head(15).itertuples(index=False):
        lines.append(
            f"| `{row.column}` | `{row.dtype}` | {int(row.nulls)} | {row.null_pct:.2f}% | {int(row.n_unique)} |"
        )

    lines.extend(
        [
            "",
            "## Riesgos observados",
            "",
            "- Los nulos se concentran en clientes sin pedidos o sin entrega efectiva en la ventana `dev`.",
            "- No hay columnas con variabilidad nula.",
            "- Las fechas y columnas de monto requieren control de leakage antes de modelado.",
            f"- Watchlist de leakage inicial: `{', '.join(leakage_watchlist)}`.",
            "",
            "## Recomendacion para Fase 4",
            "",
            "- Mantener limpieza minima para conteos y sumas con imputacion a `0`.",
            "- No imputar ciegamente columnas de catalogo o comportamiento de pago sin justificacion.",
            "- Excluir del modelado directo las variables pegadas a gasto total hasta cerrar la revision metodologica.",
        ]
    )

    if low_variability_columns:
        lines.extend(["", f"- Columnas con variabilidad casi nula: `{', '.join(low_variability_columns)}`"])

    return "\n".join(lines) + "\n"


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)


def save_text(text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def save_threshold_metadata(
    threshold: float,
    output_path: Path,
    *,
    profile_source: str,
    reference_population: str,
    mode: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile_source": profile_source,
        "premium_threshold": round(float(threshold), 2),
        "threshold_rule": "P80",
        "threshold_reference_population": reference_population,
        "threshold_mode": mode,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_threshold_metadata(metadata_path: Path) -> dict[str, object]:
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye la master table del proyecto.")
    parser.add_argument("--profile-source", default="dev", choices=["raw", "dev"])
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--raw-output-path",
        type=Path,
        default=None,
        help="Ruta opcional para guardar la master table sucia del Sprint 2.",
    )
    parser.add_argument(
        "--profile-output-path",
        type=Path,
        default=None,
        help="Ruta opcional para guardar el perfil de la master table sucia.",
    )
    parser.add_argument(
        "--threshold-metadata-path",
        type=Path,
        default=None,
        help="Ruta del JSON con el umbral premium fijo.",
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["fit", "apply", "auto"],
        default="auto",
        help="fit calcula y guarda umbral, apply reutiliza uno existente, auto aplica si existe o calcula si no existe.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildConfig(project_root=resolve_project_root(), profile_source=args.profile_source)

    raw_output_path = args.raw_output_path or (config.interim_dir / "01_master_table_raw_sprint2.parquet")
    clean_output_path = args.output_path or (
        config.processed_dir
        / ("03_master_table_clean.parquet" if config.profile_source == "dev" else "03_master_table_clean_raw.parquet")
    )
    compatibility_output_path = config.processed_dir / (
        "master_table.parquet" if config.profile_source == "raw" else "master_table_dev.parquet"
    )
    profile_output_path = args.profile_output_path or (config.interim_dir / "02_master_table_profile.md")
    threshold_metadata_path = args.threshold_metadata_path or (
        config.processed_dir / "premium_threshold_dev.json"
    )

    fixed_threshold = None
    threshold_mode_used = args.threshold_mode
    if args.threshold_mode in {"apply", "auto"} and threshold_metadata_path.exists():
        metadata = load_threshold_metadata(threshold_metadata_path)
        fixed_threshold = float(metadata["premium_threshold"])
        if args.threshold_mode == "auto":
            threshold_mode_used = "apply"
    elif args.threshold_mode == "apply":
        raise FileNotFoundError(
            f"No existe metadata de umbral en {threshold_metadata_path} para usar threshold-mode=apply"
        )
    elif args.threshold_mode == "auto":
        threshold_mode_used = "fit"

    master_table_raw, master_table_clean, threshold = build_master_table(
        config,
        fixed_threshold=fixed_threshold,
    )
    raw_validation = validate_master_table(master_table_raw.assign(is_premium=0))
    clean_validation = validate_master_table(master_table_clean)
    profile_markdown = build_profile_markdown(master_table_raw)

    save_dataframe(master_table_raw, raw_output_path)
    save_dataframe(master_table_clean, clean_output_path)
    save_dataframe(master_table_clean, compatibility_output_path)
    save_text(profile_markdown, profile_output_path)
    save_threshold_metadata(
        threshold,
        threshold_metadata_path,
        profile_source=config.profile_source,
        reference_population="data/splits/temporal_2018q4/dev" if config.profile_source == "dev" else "data/raw",
        mode=threshold_mode_used,
    )

    print(f"Profile source: {config.profile_source}")
    print(f"Premium threshold: {threshold:.2f}")
    print(f"Threshold mode used: {threshold_mode_used}")
    print(f"Raw master table saved to: {raw_output_path}")
    print(f"Clean master table saved to: {clean_output_path}")
    print(f"Compatibility master table saved to: {compatibility_output_path}")
    print(f"Profile markdown saved to: {profile_output_path}")
    print(f"Threshold metadata saved to: {threshold_metadata_path}")
    print("Raw validation:", raw_validation)
    print("Clean validation:", clean_validation)


if __name__ == "__main__":
    main()
