from __future__ import annotations

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
model_path = project_root / 'models' / 'final' / 'modelo_final.pkl'
data_path = project_root / 'data' / 'processed' / 'holdout_features_selected.parquet'

# Load
with open(model_path, 'rb') as f:
    pipeline = pickle.load(f)
df = pd.read_parquet(data_path)

# Extract preprocessor and model
preprocessor = pipeline.named_steps['preprocessor']
lgb_model = pipeline.named_steps['model']

# Get test sample (1 row)
X = df.drop(columns=['customer_unique_id', 'is_premium'])
X_sample = X.head(1)

# Preprocess
X_proc = preprocessor.transform(X_sample)
feature_names = preprocessor.get_feature_names_out()

# Predict contributions
contribs = lgb_model.predict(X_proc, pred_contrib=True)
print(f"Contributions shape: {contribs.shape}")
print(f"Num features: {len(feature_names)}")

# Check that the sum of contributions equals the margin output of the model
margin = lgb_model.predict(X_proc, raw_score=True)
sum_contrib = contribs.sum(axis=1)
print(f"Margin: {margin}")
print(f"Sum of contributions: {sum_contrib}")
print(f"Difference: {margin - sum_contrib}")

# Map feature names
df_contribs = pd.DataFrame(contribs[:, :-1], columns=feature_names)
print("\nTop 5 positive contributions:")
print(df_contribs.T.sort_values(by=0, ascending=False).head(5))

print("\nTop 5 negative contributions:")
print(df_contribs.T.sort_values(by=0, ascending=True).head(5))
