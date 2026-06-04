from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "is_premium"
ID_COLUMN = "customer_unique_id"

BASELINE_METRICS = {
    "roc_auc": 0.5355,
    "precision": 0.30,
    "recall": 0.16,
    "f1": 0.21,
    "gini": 0.0710,
}


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=160,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=5,
        max_depth=12,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    categorical = df[features].select_dtypes(include=["object"]).columns.tolist()
    numeric = [col for col in features if col not in categorical]
    return numeric, categorical


def evaluate_predictions(y_true: pd.Series, y_scores: pd.Series, threshold: float = 0.5) -> dict[str, float | list[list[int]] | list[float]]:
    y_pred = (y_scores >= threshold).astype(int)
    roc_auc = float(roc_auc_score(y_true, y_scores))
    gini = (2 * roc_auc) - 1
    cm = confusion_matrix(y_true, y_pred).tolist()
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "gini": float(gini),
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thresholds": [float(x) for x in roc_thresholds],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluacion Sprint 2 vs baseline.")
    parser.add_argument("--features-path", type=Path, default=None)
    parser.add_argument("--source-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--validation-cutoff", type=str, default="2018-07-01")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()
    features_path = args.features_path or (project_root / "data" / "processed" / "06_features_selected.parquet")
    source_path = args.source_path or (project_root / "data" / "processed" / "05_features_rfm.parquet")
    output_json = args.output_json or (project_root / "data" / "processed" / "08_evaluation_metrics.json")
    output_md = args.output_md or (project_root / "reports" / "sprint_02" / "evaluation_vs_baseline.md")

    selected_df = pd.read_parquet(features_path)
    source_df = pd.read_parquet(source_path)[[ID_COLUMN, "last_purchase"]].copy()
    df = selected_df.merge(source_df, on=ID_COLUMN, how="left", validate="one_to_one")

    cutoff = pd.Timestamp(args.validation_cutoff)
    train_df = df[df["last_purchase"] < cutoff].copy()
    val_df = df[df["last_purchase"] >= cutoff].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("El split temporal dejo train o validation vacio.")

    features = [col for col in selected_df.columns if col not in {ID_COLUMN, TARGET_COLUMN}]
    numeric_features, categorical_features = split_feature_types(train_df, features)
    model = build_model(numeric_features, categorical_features)
    model.fit(train_df[features], train_df[TARGET_COLUMN])

    train_scores = model.predict_proba(train_df[features])[:, 1]
    val_scores = model.predict_proba(val_df[features])[:, 1]

    train_metrics = evaluate_predictions(train_df[TARGET_COLUMN], train_scores)
    val_metrics = evaluate_predictions(val_df[TARGET_COLUMN], val_scores)

    payload = {
        "validation_cutoff": args.validation_cutoff,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "feature_count": len(features),
        "features": features,
        "baseline_metrics": BASELINE_METRICS,
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Evaluacion Sprint 2 vs Baseline Sprint 1",
        "",
        f"- Cutoff temporal: `{args.validation_cutoff}`",
        f"- Filas train: `{len(train_df)}`",
        f"- Filas validation: `{len(val_df)}`",
        f"- Features usadas: `{len(features)}`",
        "",
        "| Metrica | Baseline S1 | Sprint 2 Validation |",
        "| --- | ---: | ---: |",
        f"| Precision | {BASELINE_METRICS['precision']:.4f} | {val_metrics['precision']:.4f} |",
        f"| Recall | {BASELINE_METRICS['recall']:.4f} | {val_metrics['recall']:.4f} |",
        f"| F1 | {BASELINE_METRICS['f1']:.4f} | {val_metrics['f1']:.4f} |",
        f"| ROC-AUC | {BASELINE_METRICS['roc_auc']:.4f} | {val_metrics['roc_auc']:.4f} |",
        f"| Gini | {BASELINE_METRICS['gini']:.4f} | {val_metrics['gini']:.4f} |",
        "",
        "## Matriz de confusion validation",
        "",
        f"`{val_metrics['confusion_matrix']}`",
    ]
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Features path: {features_path}")
    print(f"Source path: {source_path}")
    print(f"Output json: {output_json}")
    print(f"Output md: {output_md}")
    print("Validation metrics:", val_metrics)


if __name__ == "__main__":
    main()
