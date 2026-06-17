from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RFM_PATH = PROJECT_ROOT / "data" / "processed" / "holdout_features_rfm.parquet"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "06_features_selected_metadata.json"
MODEL_PATH = PROJECT_ROOT / "models" / "final" / "modelo_final.pkl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "holdout_features_selected.parquet"


def main() -> None:
    rfm_df = pd.read_parquet(RFM_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    selected_cols = metadata["selected_model_columns"]
    missing_cols = [col for col in selected_cols if col not in rfm_df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en holdout_features_rfm: {missing_cols}")

    aligned_cols = ["customer_unique_id", "is_premium", *selected_cols]
    holdout_selected_df = rfm_df[aligned_cols].copy()
    holdout_selected_df.to_parquet(OUTPUT_PATH, index=False)

    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    x_holdout = holdout_selected_df[selected_cols]
    preprocessor = pipeline.named_steps["preprocessor"]
    lgb_model = pipeline.named_steps["model"]
    x_proc = preprocessor.transform(x_holdout)
    feature_names = preprocessor.get_feature_names_out()
    x_proc_df = pd.DataFrame(x_proc, columns=feature_names, index=x_holdout.index)
    probs = lgb_model.predict_proba(x_proc_df)[:, 1]
    preds = (probs >= 0.55).astype(int)

    print(f"Archivo generado: {OUTPUT_PATH}")
    print(f"Shape: {holdout_selected_df.shape}")
    print(
        "Validacion de scoring OK: "
        f"premium_pred={preds.mean():.4f} | premium_real={holdout_selected_df['is_premium'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
