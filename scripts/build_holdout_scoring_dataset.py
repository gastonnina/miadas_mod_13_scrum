from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "06_features_selected_metadata.json"
MODEL_PATH = PROJECT_ROOT / "models" / "final" / "modelo_final.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alinea features de scoring a las columnas del modelo final.")
    parser.add_argument("--profile-name", default="holdout", choices=["holdout", "backtest"])
    parser.add_argument("--rfm-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_rfm = PROJECT_ROOT / "data" / "processed" / f"{args.profile_name}_features_rfm.parquet"
    default_output = PROJECT_ROOT / "data" / "processed" / f"{args.profile_name}_features_selected.parquet"
    rfm_path = args.rfm_path or default_rfm
    output_path = args.output_path or default_output

    rfm_df = pd.read_parquet(rfm_path)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    selected_cols = metadata["selected_model_columns"]
    missing_cols = [col for col in selected_cols if col not in rfm_df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en {rfm_path.name}: {missing_cols}")

    aligned_cols = ["customer_unique_id", "is_premium", *selected_cols]
    selected_df = rfm_df[aligned_cols].copy()
    selected_df.to_parquet(output_path, index=False)

    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    x_scoring = selected_df[selected_cols]
    preprocessor = pipeline.named_steps["preprocessor"]
    lgb_model = pipeline.named_steps["model"]
    x_proc = preprocessor.transform(x_scoring)
    feature_names = preprocessor.get_feature_names_out()
    x_proc_df = pd.DataFrame(x_proc, columns=feature_names, index=x_scoring.index)
    probs = lgb_model.predict_proba(x_proc_df)[:, 1]
    preds = (probs >= 0.55).astype(int)

    print(f"Archivo generado: {output_path}")
    print(f"Shape: {selected_df.shape}")
    print(
        "Validacion de scoring OK: "
        f"premium_pred={preds.mean():.4f} | premium_real={selected_df['is_premium'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
