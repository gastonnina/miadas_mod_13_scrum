"""
Baseline logistic regression model for premium-customer classification.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASELINE_FEATURES = [
    "total_orders",
    "recency_days",
    "customer_lifetime_days",
    "avg_review_score",
    "review_is_missing",
    "late_delivery_rate",
    "payment_methods_count",
]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the review_is_missing indicator and return the feature matrix."""
    out = df.copy()
    out["review_is_missing"] = out["avg_review_score"].isna().astype(int)
    return out[BASELINE_FEATURES]


def train_baseline(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict:
    """
    Train a logistic regression baseline with leak-free imputation.

    Imputation statistics are derived only from the training split and then
    applied to the test split to avoid data leakage.

    Returns
    -------
    dict with keys: model, scaler, X_test, y_test, y_pred, y_proba,
                    train_mean_review, train_medians
    """
    X = prepare_features(df)
    y = df["is_premium"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Impute avg_review_score with train mean
    train_mean_review = X_train["avg_review_score"].mean()
    train_medians = {
        col: X_train[col].median()
        for col in BASELINE_FEATURES
        if col != "avg_review_score"
    }

    def _impute(X_split):
        out = X_split.copy()
        out["avg_review_score"] = out["avg_review_score"].fillna(train_mean_review)
        for col, median in train_medians.items():
            out[col] = out[col].fillna(median)
        return out

    X_train_imp = _impute(X_train)
    X_test_imp = _impute(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    model = LogisticRegression(class_weight="balanced", random_state=random_state)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    return {
        "model": model,
        "scaler": scaler,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "train_mean_review": train_mean_review,
        "train_medians": train_medians,
    }
