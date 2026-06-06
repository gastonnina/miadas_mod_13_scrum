from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_DESCRIPTIONS = {
    "items_per_order": "Promedio de items por pedido.",
    "products_per_order": "Promedio de productos distintos por pedido.",
    "max_to_avg_price_ratio": "Relacion entre precio maximo y precio promedio del item.",
    "freight_to_item_ratio": "Relacion entre flete promedio y precio promedio del item.",
    "installments_gt_1_flag": "Indica si el cliente uso cuotas mayores a 1.",
    "installments_gt_6_flag": "Indica si el cliente uso cuotas altas, mayores a 6.",
    "credit_card_flag": "Indica si el metodo de pago principal es tarjeta de credito.",
    "boleto_flag": "Indica si el metodo de pago principal es boleto.",
    "voucher_flag": "Indica si el metodo de pago principal es voucher.",
    "payment_complexity_flag": "Indica multiples metodos o cuotas altas.",
    "has_late_delivery": "Indica si el cliente tuvo al menos una entrega tardia.",
    "has_cancellation": "Indica si el cliente tuvo al menos una cancelacion.",
    "delivery_gap": "Diferencia entre tiempo estimado y tiempo real promedio de entrega.",
    "reviews_per_order": "Promedio de reseñas por pedido.",
    "region_group": "Agrupacion geografica por region de Brasil.",
    "far_region_flag": "Indica region alejada con mayor costo logistico potencial.",
    "top_category_group": "Agrupacion de categoria principal del cliente.",
    "top_category_is_high_value": "Indica si la categoria principal pertenece al cuartil superior de precio promedio por categoria.",
}

REGION_BY_STATE = {
    "AC": "north",
    "AM": "north",
    "AP": "north",
    "PA": "north",
    "RO": "north",
    "RR": "north",
    "TO": "north",
    "AL": "northeast",
    "BA": "northeast",
    "CE": "northeast",
    "MA": "northeast",
    "PB": "northeast",
    "PE": "northeast",
    "PI": "northeast",
    "RN": "northeast",
    "SE": "northeast",
    "DF": "center_west",
    "GO": "center_west",
    "MS": "center_west",
    "MT": "center_west",
    "ES": "southeast",
    "MG": "southeast",
    "RJ": "southeast",
    "SP": "southeast",
    "PR": "south",
    "RS": "south",
    "SC": "south",
}


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.divide(denominator.replace(0, np.nan))
    return result.replace([np.inf, -np.inf], np.nan)


def group_top_categories(df: pd.DataFrame, min_count: int = 500) -> pd.Series:
    counts = df["top_category"].value_counts(dropna=True)
    frequent_categories = set(counts[counts >= min_count].index.tolist())
    grouped = df["top_category"].where(df["top_category"].isin(frequent_categories), other="other")
    return grouped.fillna("unknown")


def high_value_categories(df: pd.DataFrame, min_count: int = 100) -> set[str]:
    category_stats = (
        df.dropna(subset=["top_category"])
        .groupby("top_category")
        .agg(
            customers=("customer_unique_id", "count"),
            category_avg_item_price=("avg_item_price", "mean"),
        )
    )
    eligible = category_stats[category_stats["customers"] >= min_count]
    if eligible.empty:
        return set()
    threshold = eligible["category_avg_item_price"].quantile(0.75)
    return set(eligible[eligible["category_avg_item_price"] >= threshold].index.tolist())


def build_features(master_table: pd.DataFrame) -> pd.DataFrame:
    df = master_table.copy()

    df["items_per_order"] = safe_divide(df["total_items"], df["total_orders"]).fillna(0)
    df["products_per_order"] = safe_divide(df["total_products"], df["total_orders"]).fillna(0)
    df["max_to_avg_price_ratio"] = safe_divide(df["max_item_price"], df["avg_item_price"]).fillna(0)
    df["freight_to_item_ratio"] = safe_divide(df["avg_freight_value"], df["avg_item_price"]).fillna(0)

    df["installments_gt_1_flag"] = df["max_payment_installments"].fillna(0).gt(1).astype(int)
    df["installments_gt_6_flag"] = df["max_payment_installments"].fillna(0).gt(6).astype(int)
    payment_type = df["main_payment_type"].fillna("unknown")
    df["credit_card_flag"] = payment_type.eq("credit_card").astype(int)
    df["boleto_flag"] = payment_type.eq("boleto").astype(int)
    df["voucher_flag"] = payment_type.eq("voucher").astype(int)
    df["payment_complexity_flag"] = (
        df["payment_methods_count"].fillna(0).gt(1) | df["max_payment_installments"].fillna(0).gt(6)
    ).astype(int)

    df["has_late_delivery"] = df["late_deliveries"].fillna(0).gt(0).astype(int)
    df["has_cancellation"] = df["canceled_orders"].fillna(0).gt(0).astype(int)
    df["delivery_gap"] = (
        df["avg_estimated_delivery_days"].fillna(0) - df["avg_delivery_days"].fillna(0)
    )
    df["reviews_per_order"] = safe_divide(df["total_reviews"], df["total_orders"]).fillna(0)

    df["region_group"] = df["customer_state"].map(REGION_BY_STATE).fillna("unknown")
    df["far_region_flag"] = df["region_group"].isin(["north", "northeast"]).astype(int)

    df["top_category_group"] = group_top_categories(df)
    premium_categories = high_value_categories(df)
    df["top_category_is_high_value"] = df["top_category"].isin(premium_categories).fillna(False).astype(int)

    return df


def feature_metadata(df: pd.DataFrame) -> dict[str, object]:
    feature_columns = list(FEATURE_DESCRIPTIONS.keys())
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "target_column": "is_premium",
        "target_source": "Inherited from 03_master_table_clean.parquet with fixed dev threshold",
        "descriptions": FEATURE_DESCRIPTIONS,
    }


def save_metadata(metadata: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def validate_features(df: pd.DataFrame) -> dict[str, object]:
    expected = list(FEATURE_DESCRIPTIONS.keys())
    missing = [column for column in expected if column not in df.columns]
    key_nulls = df[["customer_unique_id", "is_premium"]].isna().sum().to_dict()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicated_customer_unique_id": int(df["customer_unique_id"].duplicated().sum()),
        "missing_expected_features": missing,
        "key_nulls": key_nulls,
        "premium_rate": float(df["is_premium"].mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye features del Sprint 2.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Ruta de entrada de la master table limpia.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Ruta de salida para las features.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Ruta de salida para metadata de features.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()
    input_path = args.input_path or (project_root / "data" / "processed" / "03_master_table_clean.parquet")
    output_path = args.output_path or (project_root / "data" / "processed" / "05_features_rfm.parquet")
    metadata_path = args.metadata_path or (project_root / "data" / "processed" / "05_features_rfm_metadata.json")

    master_table = pd.read_parquet(input_path)
    features = build_features(master_table)
    metadata = feature_metadata(features)
    validation = validate_features(features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    save_metadata(metadata, metadata_path)

    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Metadata path: {metadata_path}")
    print("Validation:", validation)


if __name__ == "__main__":
    main()
