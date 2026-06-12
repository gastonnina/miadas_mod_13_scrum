from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - depende del entorno.
    XGBClassifier = None


TARGET_COLUMN = "is_premium"
ID_COLUMN = "customer_unique_id"


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    categorical = df[features].select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [col for col in features if col not in categorical]
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric_features),
            ("cat", Pipeline(categorical_steps), categorical_features),
        ]
    )


def build_model_catalog(scale_pos_weight: float) -> dict[str, tuple[object, bool]]:
    catalog: dict[str, tuple[object, bool]] = {
        "logistic_regression": (
            LogisticRegression(
                class_weight="balanced",
                max_iter=1500,
                random_state=42,
            ),
            True,
        ),
        "decision_tree": (
            DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=10,
                min_samples_leaf=10,
                random_state=42,
            ),
            False,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=160,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
                min_samples_leaf=5,
                max_depth=12,
            ),
            False,
        ),
        "svm_linear": (
            CalibratedClassifierCV(
                estimator=LinearSVC(
                    class_weight="balanced",
                    C=1.0,
                    random_state=42,
                    max_iter=3000,
                ),
                cv=3,
            ),
            True,
        ),
        "extra_trees": (
            ExtraTreesClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
                min_samples_leaf=3,
                max_depth=14,
            ),
            False,
        ),
    }

    if XGBClassifier is not None:
        catalog["xgboost"] = (
            XGBClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                min_child_weight=2,
                reg_alpha=0.0,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
            ),
            False,
        )

    return catalog


def build_pipeline(
    model: object,
    scale_numeric: bool,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    preprocessor = build_preprocessor(numeric_features, categorical_features, scale_numeric=scale_numeric)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def predict_scores(model: Pipeline, features: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        return pd.Series(model.predict_proba(features)[:, 1], index=features.index)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(features)
        return pd.Series(scores, index=features.index)
    raise ValueError("El modelo no soporta predict_proba ni decision_function.")


def evaluate_predictions(y_true: pd.Series, y_scores: pd.Series, threshold: float = 0.5) -> dict[str, float | list[list[int]]]:
    y_pred = (y_scores >= threshold).astype(int)
    roc_auc = float(roc_auc_score(y_true, y_scores))
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "gini": float((2 * roc_auc) - 1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparacion de modelos de clasificacion.")
    parser.add_argument("--features-path", type=Path, default=None)
    parser.add_argument("--source-path", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--validation-cutoff", type=str, default="2018-07-01")
    return parser.parse_args()


def save_markdown_summary(
    results: list[dict[str, object]],
    validation_cutoff: str,
    train_rows: int,
    validation_rows: int,
    feature_count: int,
    output_path: Path,
) -> None:
    lines = [
        "# Comparacion de Modelos",
        "",
        f"- Cutoff temporal: `{validation_cutoff}`",
        f"- Filas train: `{train_rows}`",
        f"- Filas validation: `{validation_rows}`",
        f"- Features usadas: `{feature_count}`",
        "",
        "| Modelo | ROC-AUC Val | Gini Val | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in results:
        lines.append(
            f"| `{row['model_name']}` | {row['validation_metrics']['roc_auc']:.4f} | {row['validation_metrics']['gini']:.4f} | {row['validation_metrics']['precision']:.4f} | {row['validation_metrics']['recall']:.4f} | {row['validation_metrics']['f1']:.4f} |"
        )

    best_model = results[0]
    lines.extend(
        [
            "",
            f"## Mejor modelo por ROC-AUC",
            "",
            f"- Modelo: `{best_model['model_name']}`",
            f"- ROC-AUC validation: `{best_model['validation_metrics']['roc_auc']:.4f}`",
            f"- Matriz de confusion: `{best_model['validation_metrics']['confusion_matrix']}`",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()
    features_path = args.features_path or (project_root / "data" / "processed" / "06_features_selected.parquet")
    source_path = args.source_path or (project_root / "data" / "processed" / "05_features_rfm.parquet")
    output_json = args.output_json or (project_root / "data" / "processed" / "09_model_comparison.json")
    output_md = args.output_md or (project_root / "reports" / "sprint_03" / "comparacion_modelos.md")

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

    positives = max(1, int(train_df[TARGET_COLUMN].sum()))
    negatives = max(1, int(len(train_df) - positives))
    scale_pos_weight = negatives / positives

    results: list[dict[str, object]] = []
    for model_name, (estimator, scale_numeric) in build_model_catalog(scale_pos_weight).items():
        pipeline = build_pipeline(estimator, scale_numeric, numeric_features, categorical_features)
        pipeline.fit(train_df[features], train_df[TARGET_COLUMN])

        train_scores = predict_scores(pipeline, train_df[features])
        val_scores = predict_scores(pipeline, val_df[features])

        results.append(
            {
                "model_name": model_name,
                "scale_numeric": scale_numeric,
                "train_metrics": evaluate_predictions(train_df[TARGET_COLUMN], train_scores),
                "validation_metrics": evaluate_predictions(val_df[TARGET_COLUMN], val_scores),
            }
        )

    results.sort(key=lambda row: row["validation_metrics"]["roc_auc"], reverse=True)

    payload = {
        "validation_cutoff": args.validation_cutoff,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "feature_count": len(features),
        "features": features,
        "results": results,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    save_markdown_summary(
        results=results,
        validation_cutoff=args.validation_cutoff,
        train_rows=len(train_df),
        validation_rows=len(val_df),
        feature_count=len(features),
        output_path=output_md,
    )

    print(f"Output json: {output_json}")
    print(f"Output md: {output_md}")
    for row in results:
        metrics = row["validation_metrics"]
        print(
            f"{row['model_name']}: roc_auc={metrics['roc_auc']:.4f}, "
            f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
