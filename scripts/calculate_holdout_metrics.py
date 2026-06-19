from __future__ import annotations

import json
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score

project_root = Path(__file__).resolve().parents[1]
selected_path = project_root / 'data' / 'processed' / 'holdout_features_selected.parquet'
master_path = project_root / 'data' / 'processed' / '03_master_table_clean_holdout.parquet'
model_path = project_root / 'models' / 'final' / 'modelo_final.pkl'
metadata_path = project_root / 'data' / 'processed' / '06_features_selected_metadata.json'

# Load files
selected_df = pd.read_parquet(selected_path)
master_df = pd.read_parquet(master_path)
with open(model_path, 'rb') as f:
    pipeline = pickle.load(f)
with open(metadata_path) as f:
    metadata = json.load(f)

selected_cols = metadata['selected_model_columns']

X = selected_df[selected_cols]
y_true = selected_df['is_premium']

# Ingress predictions
probs = pipeline.predict_proba(X)[:, 1]
preds_50 = (probs >= 0.50).astype(int)
preds_55 = (probs >= 0.55).astype(int)

# Merging with total_spent from master table
# master_df has customer_unique_id and total_spent
df_eval = pd.DataFrame({
    'customer_unique_id': selected_df['customer_unique_id'],
    'is_premium': y_true,
    'prob': probs,
    'pred_50': preds_50,
    'pred_55': preds_55
})
df_eval = df_eval.merge(master_df[['customer_unique_id', 'total_spent']], on='customer_unique_id', how='left')

# Print overall info
n_total = len(df_eval)
n_premium = y_true.sum()
premium_rate = y_true.mean()

print(f"Total customers: {n_total}")
print(f"Actual premium customers: {n_premium} ({premium_rate:.2%})")

# Evaluate at threshold 0.55
auc = roc_auc_score(y_true, probs)
gini = 2 * auc - 1
f1 = f1_score(y_true, preds_55)
prec = precision_score(y_true, preds_55)
rec = recall_score(y_true, preds_55)

print("\n--- Technical Metrics on Holdout (Threshold 0.55) ---")
print(f"ROC-AUC: {auc:.4f}")
print(f"Gini: {gini:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")

cm = confusion_matrix(y_true, preds_55)
print(f"Confusion Matrix:\n{cm}")

# Business Metrics
total_spend = df_eval['total_spent'].sum()
premium_spend = df_eval[df_eval['is_premium'] == 1]['total_spent'].sum()
predicted_premium_spend = df_eval[df_eval['pred_55'] == 1]['total_spent'].sum()
tp_spend = df_eval[(df_eval['pred_55'] == 1) & (df_eval['is_premium'] == 1)]['total_spent'].sum()

print("\n--- Spend Metrics ---")
print(f"Total spend in holdout: BRL {total_spend:,.2f}")
print(f"Actual premium segment spend: BRL {premium_spend:,.2f} ({premium_spend/total_spend:.2%} of total)")
print(f"Predicted premium segment spend: BRL {predicted_premium_spend:,.2f} ({predicted_premium_spend/total_spend:.2%} of total)")
print(f"True positive (correctly detected premium) spend: BRL {tp_spend:,.2f} ({tp_spend/premium_spend:.2%} of premium spend, {tp_spend/total_spend:.2%} of total)")

# Campaign Simulation
# Let's say we have a budget/cost of campaign:
# Cost of sending a campaign = BRL 15 per customer
# Revenue return from a premium customer reached = BRL 120 (net contribution)
# Revenue return from a regular customer reached = BRL 0

cost_per_cust = 15
rev_per_premium = 120

# Case 1: Target ALL customers
cost_all = n_total * cost_per_cust
rev_all = n_premium * rev_per_premium
profit_all = rev_all - cost_all
roi_all = profit_all / cost_all if cost_all > 0 else 0

# Case 2: Target ONLY predicted premium (Threshold 0.55)
n_pred_premium = preds_55.sum()
tp_count = ((preds_55 == 1) & (y_true == 1)).sum()
cost_pred = n_pred_premium * cost_per_cust
rev_pred = tp_count * rev_per_premium
profit_pred = rev_pred - cost_pred
roi_pred = profit_pred / cost_pred if cost_pred > 0 else 0
savings = cost_all - cost_pred

print("\n--- Marketing Campaign Simulation ---")
print(f"Campaign cost per customer: BRL {cost_per_cust}")
print(f"Net profit from premium customer: BRL {rev_per_premium}")
print(f"\nScenario A: Target ALL Customers")
print(f"  Customers targeted: {n_total}")
print(f"  Total Cost: BRL {cost_all:,.2f}")
print(f"  Total Revenue: BRL {rev_all:,.2f}")
print(f"  Net Profit: BRL {profit_all:,.2f}")
print(f"  ROI: {roi_all:.2%}")
print(f"\nScenario B: Target ONLY Predicted Premium (Threshold 0.55)")
print(f"  Customers targeted: {n_pred_premium} (reduction of {1 - n_pred_premium/n_total:.2%})")
print(f"  Total Cost: BRL {cost_pred:,.2f} (Savings of BRL {savings:,.2f})")
print(f"  True Premium detected in target: {tp_count} (out of {n_premium})")
print(f"  Total Revenue: BRL {rev_pred:,.2f}")
print(f"  Net Profit: BRL {profit_pred:,.2f}")
print(f"  ROI: {roi_pred:.2%}")
print(f"  ROI Improvement factor: {roi_pred/roi_all:.2f}x")
