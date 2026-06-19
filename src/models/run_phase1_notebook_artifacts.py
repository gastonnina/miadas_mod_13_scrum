from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


TARGET = "is_premium"
ID_COL = "customer_unique_id"
CUTOFF = pd.Timestamp("2018-07-01")
SEED = 42

BASELINE_LGBM_PARAMS = {
    "n_estimators": 250,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": SEED,
    "verbosity": -1,
}

BASELINE_XGB_PARAMS = {
    "n_estimators": 250,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": SEED,
}

# Sin Optuna disponible en el entorno, se reutilizan los mejores parámetros ya
# validados y persistidos desde el notebook de tuning.
TUNED_LGBM_PARAMS = {
    "num_leaves": 59,
    "max_depth": 8,
    "learning_rate": 0.054038945566405934,
    "n_estimators": 255,
    "min_child_samples": 68,
    "subsample": 0.9059228577785707,
    "colsample_bytree": 0.855428530082444,
    "reg_alpha": 6.735979730583029e-08,
    "reg_lambda": 0.0005641632739363156,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": SEED,
    "verbosity": -1,
}

TUNED_XGB_PARAMS = {
    "n_estimators": 331,
    "max_depth": 10,
    "learning_rate": 0.017563420206771783,
    "min_child_weight": 3,
    "subsample": 0.6001143839799814,
    "colsample_bytree": 0.7337219366496492,
    "gamma": 1.93801421494858,
    "reg_alpha": 9.752866483313244e-07,
    "reg_lambda": 0.40475076755318734,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": -1,
    "random_state": SEED,
}

DOMAIN_MAP = {
    "customer_unique_id": "customers",
    "is_premium": "target",
    "total_orders": "orders",
    "total_items": "order_items_products",
    "total_products": "order_items_products",
    "avg_review_score": "order_reviews",
    "avg_delivery_days": "orders",
    "avg_estimated_delivery_days": "orders",
    "delivered_orders": "orders",
    "late_deliveries": "orders",
    "payment_methods_count": "order_payments",
    "max_payment_installments": "order_payments",
    "recency_days": "orders",
    "customer_lifetime_days": "orders",
    "cancellation_rate": "orders",
    "products_per_order": "features_derivadas",
    "max_to_avg_price_ratio": "features_derivadas",
    "installments_gt_1_flag": "features_derivadas",
    "installments_gt_6_flag": "features_derivadas",
    "credit_card_flag": "features_derivadas",
    "voucher_flag": "features_derivadas",
    "delivery_gap": "features_derivadas",
    "reviews_per_order": "order_reviews",
    "far_region_flag": "features_derivadas",
    "top_category_is_high_value": "features_derivadas",
    "customer_state": "customers",
    "main_payment_type": "order_payments",
    "top_category": "order_items_products",
    "region_group": "features_derivadas",
    "top_category_group": "features_derivadas",
}


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_preprocessor(num_features: list[str], cat_features: list[str]) -> ColumnTransformer:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, num_features),
            ("cat", cat_pipe, cat_features),
        ]
    )


def evaluate(model, x_train, y_train, x_val, y_val) -> dict[str, float]:
    train_scores = model.predict_proba(x_train)[:, 1]
    val_scores = model.predict_proba(x_val)[:, 1]

    train_auc = float(roc_auc_score(y_train, train_scores))
    val_auc = float(roc_auc_score(y_val, val_scores))
    val_preds = (val_scores >= 0.5).astype(int)

    return {
        "roc_auc_train": train_auc,
        "roc_auc_val": val_auc,
        "gini_val": float((2 * val_auc) - 1),
        "f1_val": float(f1_score(y_val, val_preds)),
        "precision_val": float(precision_score(y_val, val_preds, zero_division=0)),
        "recall_val": float(recall_score(y_val, val_preds, zero_division=0)),
        "overfit_gap": round(train_auc - val_auc, 4),
    }


def aggregate_importance(
    model,
    num_features: list[str],
    cat_features: list[str],
    encoder_feature_names: list[str],
) -> pd.DataFrame:
    processed_names = num_features + encoder_feature_names
    fi = pd.DataFrame(
        {
            "processed_feature": processed_names,
            "importance_raw": model.feature_importances_,
        }
    )

    def to_source_variable(name: str) -> str:
        if name in num_features:
            return name
        for feature in cat_features:
            prefix = f"{feature}_"
            if name.startswith(prefix):
                return feature
        raise ValueError(f"No se pudo mapear la feature procesada: {name}")

    fi["Variable"] = fi["processed_feature"].map(to_source_variable)
    grouped = fi.groupby("Variable", as_index=False)["importance_raw"].sum()
    total = float(grouped["importance_raw"].sum())
    if total > 0:
        grouped["importance_pct"] = (grouped["importance_raw"] / total) * 100.0
    else:
        grouped["importance_pct"] = 0.0
    return grouped.sort_values("importance_pct", ascending=False).reset_index(drop=True)


def build_audit_table(
    selected_model_columns: list[str],
    baseline_importance: pd.DataFrame,
    tuned_importance: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    base_variables = [ID_COL, TARGET, *selected_model_columns]
    rows = []
    for idx, variable in enumerate(base_variables, start=1):
        rows.append(
            {
                "Nro": idx,
                "Dominio": DOMAIN_MAP[variable],
                "Variable": variable,
                "flagSelected": 0 if variable in {ID_COL, TARGET} else 1,
            }
        )

    audit_df = pd.DataFrame(rows)
    audit_df = audit_df.merge(
        baseline_importance[["Variable", "importance_pct"]].rename(
            columns={"importance_pct": "importancia_seleccion"}
        ),
        on="Variable",
        how="left",
    )
    audit_df = audit_df.merge(
        tuned_importance[["Variable", "importance_pct"]].rename(
            columns={"importance_pct": "importancia_modelo_final"}
        ),
        on="Variable",
        how="left",
    )
    audit_df["importancia_seleccion"] = audit_df["importancia_seleccion"].fillna(0.0)
    audit_df["importancia_modelo_final"] = audit_df["importancia_modelo_final"].fillna(0.0)
    audit_df["modelo"] = model_name
    return audit_df


def save_pipeline(path: Path, preprocessor: ColumnTransformer, estimator) -> None:
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(pipeline, f)


def main() -> None:
    project_root = resolve_project_root()
    data_dir = project_root / "data" / "processed"
    reports_dir = project_root / "reports" / "sprint_03"
    models_dir = project_root / "models"

    selected_df = pd.read_parquet(data_dir / "06_features_selected.parquet")
    temporal_df = pd.read_parquet(data_dir / "05_features_rfm.parquet", columns=[ID_COL, "last_purchase"])
    metadata = json.loads((data_dir / "06_features_selected_metadata.json").read_text())

    df = selected_df.merge(temporal_df, on=ID_COL, how="left", validate="one_to_one")
    train_df = df[df["last_purchase"] < CUTOFF].copy()
    val_df = df[df["last_purchase"] >= CUTOFF].copy()

    features = [c for c in selected_df.columns if c not in {ID_COL, TARGET}]
    cat_features = (
        train_df[features]
        .select_dtypes(include=["object", "string", "category"])
        .columns.tolist()
    )
    num_features = [c for c in features if c not in cat_features]

    positives = int(train_df[TARGET].sum())
    negatives = int(len(train_df) - positives)
    scale_pos_weight = negatives / positives

    preprocessor = build_preprocessor(num_features, cat_features)
    x_train = preprocessor.fit_transform(train_df[features])
    x_val = preprocessor.transform(val_df[features])
    y_train = train_df[TARGET].values
    y_val = val_df[TARGET].values

    encoder_feature_names = (
        preprocessor.named_transformers_["cat"]["encoder"]
        .get_feature_names_out(cat_features)
        .tolist()
    )

    baseline_lgbm = LGBMClassifier(**BASELINE_LGBM_PARAMS)
    baseline_lgbm.fit(x_train, y_train)

    baseline_xgb = XGBClassifier(**BASELINE_XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    baseline_xgb.fit(x_train, y_train)

    tuned_lgbm = LGBMClassifier(**TUNED_LGBM_PARAMS)
    tuned_lgbm.fit(x_train, y_train)

    tuned_xgb = XGBClassifier(**TUNED_XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    tuned_xgb.fit(x_train, y_train)

    metrics = {
        "lightgbm_baseline": evaluate(baseline_lgbm, x_train, y_train, x_val, y_val),
        "xgboost_baseline": evaluate(baseline_xgb, x_train, y_train, x_val, y_val),
        "lightgbm_tuned": evaluate(tuned_lgbm, x_train, y_train, x_val, y_val),
        "xgboost_tuned": evaluate(tuned_xgb, x_train, y_train, x_val, y_val),
    }

    baseline_lgbm_importance = aggregate_importance(
        baseline_lgbm, num_features, cat_features, encoder_feature_names
    )
    baseline_xgb_importance = aggregate_importance(
        baseline_xgb, num_features, cat_features, encoder_feature_names
    )
    tuned_lgbm_importance = aggregate_importance(
        tuned_lgbm, num_features, cat_features, encoder_feature_names
    )
    tuned_xgb_importance = aggregate_importance(
        tuned_xgb, num_features, cat_features, encoder_feature_names
    )

    lightgbm_audit = build_audit_table(
        metadata["selected_model_columns"],
        baseline_lgbm_importance,
        tuned_lgbm_importance,
        "lightgbm",
    )
    xgboost_audit = build_audit_table(
        metadata["selected_model_columns"],
        baseline_xgb_importance,
        tuned_xgb_importance,
        "xgboost",
    )

    save_pipeline(models_dir / "baseline" / "lightgbm_baseline.pkl", preprocessor, baseline_lgbm)
    save_pipeline(models_dir / "baseline" / "xgboost_baseline.pkl", preprocessor, baseline_xgb)
    save_pipeline(models_dir / "final" / "lightgbm_tuned.pkl", preprocessor, tuned_lgbm)
    save_pipeline(models_dir / "final" / "xgboost_tuned.pkl", preprocessor, tuned_xgb)

    baseline_lgbm_importance.to_parquet(data_dir / "11_lightgbm_baseline_importance.parquet", index=False)
    baseline_xgb_importance.to_parquet(data_dir / "11_xgboost_baseline_importance.parquet", index=False)
    tuned_lgbm_importance.to_parquet(data_dir / "12_lightgbm_tuned_importance.parquet", index=False)
    tuned_xgb_importance.to_parquet(data_dir / "12_xgboost_tuned_importance.parquet", index=False)

    lightgbm_audit.to_parquet(data_dir / "13_feature_audit_lightgbm.parquet", index=False)
    xgboost_audit.to_parquet(data_dir / "13_feature_audit_xgboost.parquet", index=False)
    lightgbm_audit.to_csv(data_dir / "13_feature_audit_lightgbm.csv", index=False)
    xgboost_audit.to_csv(data_dir / "13_feature_audit_xgboost.csv", index=False)

    phase1_summary = {
        "validation_cutoff": str(CUTOFF.date()),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "selected_feature_count": len(metadata["selected_model_columns"]),
        "metrics": metrics,
        "artifacts": {
            "models": {
                "lightgbm_baseline": "models/baseline/lightgbm_baseline.pkl",
                "xgboost_baseline": "models/baseline/xgboost_baseline.pkl",
                "lightgbm_tuned": "models/final/lightgbm_tuned.pkl",
                "xgboost_tuned": "models/final/xgboost_tuned.pkl",
            },
            "feature_audit": {
                "lightgbm": "data/processed/13_feature_audit_lightgbm.parquet",
                "xgboost": "data/processed/13_feature_audit_xgboost.parquet",
            },
        },
    }
    (data_dir / "13_phase1_summary.json").write_text(
        json.dumps(phase1_summary, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Fase 1 - Modelos e Importancias",
        "",
        f"- Cutoff validacion: `{CUTOFF.date()}`",
        f"- Filas train: `{len(train_df)}`",
        f"- Filas validation: `{len(val_df)}`",
        f"- Features seleccionadas: `{len(metadata['selected_model_columns'])}`",
        "",
        "| Modelo | ROC-AUC val | Gini val | F1 val | Recall val | Gap overfit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in [
        "lightgbm_baseline",
        "lightgbm_tuned",
        "xgboost_baseline",
        "xgboost_tuned",
    ]:
        row = metrics[key]
        report_lines.append(
            f"| {key} | {row['roc_auc_val']:.6f} | {row['gini_val']:.6f} | {row['f1_val']:.6f} | {row['recall_val']:.6f} | {row['overfit_gap']:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "## Artefactos",
            "",
            "- `models/baseline/lightgbm_baseline.pkl`",
            "- `models/baseline/xgboost_baseline.pkl`",
            "- `models/final/lightgbm_tuned.pkl`",
            "- `models/final/xgboost_tuned.pkl`",
            "- `data/processed/13_feature_audit_lightgbm.parquet`",
            "- `data/processed/13_feature_audit_xgboost.parquet`",
        ]
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "phase1_model_artifacts.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("Fase 1 completada.")
    print(json.dumps(phase1_summary, indent=2))


if __name__ == "__main__":
    main()
