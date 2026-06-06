from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


IDENTIFIER_COLUMNS = [
    "customer_unique_id",
]

TARGET_COLUMNS = [
    "is_premium",
]

LEAKAGE_COLUMNS = [
    "total_spent",
    "avg_ticket",
    "avg_order_price",
]

ANALYSIS_ONLY_COLUMNS = [
    "first_purchase",
    "last_purchase",
    "customer_city",
    "customer_zip_code_prefix",
    "customer_state",
    "main_payment_type",
    "top_category",
    "avg_item_price",
    "max_item_price",
    "total_items",
    "total_reviews",
    "delivered_orders",
    "canceled_orders",
    "late_deliveries",
    "avg_payment_installments",
    "payment_complexity_flag",
    "has_late_delivery",
    "has_cancellation",
]

SELECTED_MODEL_COLUMNS = [
    "total_orders",
    "total_products",
    "avg_review_score",
    "avg_delivery_days",
    "avg_estimated_delivery_days",
    "payment_methods_count",
    "max_payment_installments",
    "avg_freight_value",
    "avg_freight_ratio",
    "recency_days",
    "customer_lifetime_days",
    "cancellation_rate",
    "late_delivery_rate",
    "items_per_order",
    "products_per_order",
    "max_to_avg_price_ratio",
    "freight_to_item_ratio",
    "installments_gt_1_flag",
    "installments_gt_6_flag",
    "credit_card_flag",
    "boleto_flag",
    "voucher_flag",
    "delivery_gap",
    "reviews_per_order",
    "region_group",
    "far_region_flag",
    "top_category_group",
    "top_category_is_high_value",
]


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_selection_metadata(df: pd.DataFrame) -> dict[str, object]:
    excluded = sorted(set(df.columns) - set(IDENTIFIER_COLUMNS) - set(TARGET_COLUMNS) - set(SELECTED_MODEL_COLUMNS))
    return {
        "rows": int(df.shape[0]),
        "selected_feature_count": len(SELECTED_MODEL_COLUMNS),
        "identifier_columns": IDENTIFIER_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "selected_model_columns": SELECTED_MODEL_COLUMNS,
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
        "analysis_only_columns": ANALYSIS_ONLY_COLUMNS,
        "other_excluded_columns": [col for col in excluded if col not in LEAKAGE_COLUMNS and col not in ANALYSIS_ONLY_COLUMNS],
        "selection_rationale": {
            "model": "Variables operativas, logisticas, de cuotas, composicion y geografia agregada con señal util y menor riesgo de leakage directo.",
            "analysis_only": "Variables utiles para EDA, interpretacion o futuras discusiones, pero no incluidas en esta primera seleccion del modelo.",
            "leakage": "Variables demasiado cercanas a la definicion del target por gasto acumulado o ticket.",
        },
    }


def validate_selection(source_df: pd.DataFrame, selected_df: pd.DataFrame) -> dict[str, object]:
    missing_selected = [col for col in SELECTED_MODEL_COLUMNS if col not in source_df.columns]
    leakage_included = [col for col in LEAKAGE_COLUMNS if col in selected_df.columns]
    analysis_only_included = [col for col in ANALYSIS_ONLY_COLUMNS if col in selected_df.columns]
    return {
        "rows": int(selected_df.shape[0]),
        "columns": int(selected_df.shape[1]),
        "duplicated_customer_unique_id": int(selected_df["customer_unique_id"].duplicated().sum()),
        "missing_selected_columns": missing_selected,
        "leakage_included": leakage_included,
        "analysis_only_included": analysis_only_included,
        "target_rate": float(selected_df["is_premium"].mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seleccion inicial de features para Sprint 2.")
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument(
        "--allow-manual-selection",
        action="store_true",
        help="Permite ejecutar la seleccion manual historica. El flujo oficial usa feature_selection_experiments.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_manual_selection:
        raise SystemExit(
            "select_features.py esta deprecado para el flujo final del Sprint 2. "
            "Usa src/features/feature_selection_experiments.py o agrega --allow-manual-selection si necesitas reproducir la seleccion manual historica."
        )
    project_root = resolve_project_root()
    input_path = args.input_path or (project_root / "data" / "processed" / "05_features_rfm.parquet")
    output_path = args.output_path or (project_root / "data" / "processed" / "06_features_selected.parquet")
    metadata_path = args.metadata_path or (project_root / "data" / "processed" / "06_features_selected_metadata.json")

    source_df = pd.read_parquet(input_path)
    selected_columns = IDENTIFIER_COLUMNS + TARGET_COLUMNS + SELECTED_MODEL_COLUMNS
    selected_df = source_df[selected_columns].copy()
    metadata = build_selection_metadata(source_df)
    validation = validate_selection(source_df, selected_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Metadata path: {metadata_path}")
    print("Validation:", validation)


if __name__ == "__main__":
    main()
