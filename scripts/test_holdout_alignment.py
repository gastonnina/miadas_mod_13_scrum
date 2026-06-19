from __future__ import annotations

import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parents[1]
rfm_path = project_root / 'data' / 'processed' / 'holdout_features_rfm.parquet'
metadata_path = project_root / 'data' / 'processed' / '06_features_selected_metadata.json'
model_path = project_root / 'models' / 'final' / 'modelo_final.pkl'

# 1. Load files
rfm_df = pd.read_parquet(rfm_path)
with open(metadata_path) as f:
    metadata = json.load(f)

selected_cols = metadata['selected_model_columns']
print(f"Loaded RFM holdout features: {rfm_df.shape}")
print(f"Selected model columns count: {len(selected_cols)}")

# 2. Align columns
# Check if any columns are missing in holdout RFM features
missing_cols = [col for col in selected_cols if col not in rfm_df.columns]
print(f"Missing columns in holdout features: {missing_cols}")

# Select columns and align
aligned_cols = ['customer_unique_id', 'is_premium'] + selected_cols
holdout_selected_df = rfm_df[aligned_cols].copy()
print(f"Aligned holdout selected shape: {holdout_selected_df.shape}")

# Save
out_path = project_root / 'data' / 'processed' / 'holdout_features_selected.parquet'
holdout_selected_df.to_parquet(out_path, index=False)
print(f"Saved holdout selected to {out_path}")

# 3. Load model and test predict
with open(model_path, 'rb') as f:
    pipeline = pickle.load(f)

X_holdout = holdout_selected_df[selected_cols]
y_holdout = holdout_selected_df['is_premium']

# Predict probabilities
probs = pipeline.predict_proba(X_holdout)[:, 1]
preds = (probs >= 0.55).astype(int)

print(f"Model predictions: premium rate: {preds.mean():.4f}, actual premium rate: {y_holdout.mean():.4f}")
print("Test completed successfully!")
