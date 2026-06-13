from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


IDENTIFIER_COLUMNS = ["customer_unique_id"]
TARGET_COLUMN = "is_premium"
MANDATORY_EXCLUDE = [
    "customer_unique_id",
    "is_premium",
    "total_spent",
    "avg_ticket",
    "avg_order_price",
    "avg_item_price",
    "max_item_price",
    "avg_freight_value",
    "avg_freight_ratio",
    "freight_to_item_ratio",
    "first_purchase",
    "last_purchase",
    "customer_city",
    "customer_zip_code_prefix",
]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    method: str
    threshold: float | None = None


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def mandatory_base_features(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in MANDATORY_EXCLUDE]


def split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    subset = df[features]
    # Pandas 3 separa de forma explícita los dtypes string de object.
    categorical = subset.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [col for col in features if col not in categorical]
    return numeric, categorical


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
        n_estimators=60,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=5,
        max_depth=12,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_feature_set(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
) -> dict[str, float]:
    numeric_features, categorical_features = split_feature_types(train_df, features)
    model = build_model(numeric_features, categorical_features)
    model.fit(train_df[features], train_df[TARGET_COLUMN])
    train_scores = model.predict_proba(train_df[features])[:, 1]
    val_scores = model.predict_proba(val_df[features])[:, 1]
    auc_train = float(roc_auc_score(train_df[TARGET_COLUMN], train_scores))
    auc_val = float(roc_auc_score(val_df[TARGET_COLUMN], val_scores))
    return {
        "auc_train": auc_train,
        "auc_val": auc_val,
        "gini_train": (2 * auc_train) - 1,
        "gini_val": (2 * auc_val) - 1,
    }


def filter_by_missing(train_df: pd.DataFrame, features: list[str], threshold: float) -> list[str]:
    missing_pct = train_df[features].isna().mean()
    return [col for col in features if missing_pct[col] <= threshold]


def absolute_target_signal(train_df: pd.DataFrame, features: list[str]) -> pd.Series:
    signal = {}
    y = train_df[TARGET_COLUMN]
    for column in features:
        series = train_df[column]
        if pd.api.types.is_numeric_dtype(series):
            signal[column] = abs(series.corr(y))
        else:
            grouped = train_df[[column, TARGET_COLUMN]].copy()
            rate = grouped.groupby(column, dropna=False)[TARGET_COLUMN].mean()
            signal[column] = float(rate.std()) if not rate.empty else 0.0
    return pd.Series(signal).fillna(0.0)


def filter_by_correlation(train_df: pd.DataFrame, features: list[str], threshold: float) -> list[str]:
    numeric_features = [col for col in features if pd.api.types.is_numeric_dtype(train_df[col])]
    categorical_features = [col for col in features if col not in numeric_features]
    if not numeric_features:
        return features

    corr_matrix = train_df[numeric_features].corr().abs().fillna(0.0)
    signal = absolute_target_signal(train_df, numeric_features)
    selected = numeric_features.copy()

    # Si dos variables numericas son demasiado parecidas, conservamos la que
    # muestra mayor señal absoluta frente al target y descartamos la otra.
    for i, left in enumerate(numeric_features):
        if left not in selected:
            continue
        for right in numeric_features[i + 1 :]:
            if right not in selected:
                continue
            if corr_matrix.loc[left, right] > threshold:
                drop_col = left if signal[left] < signal[right] else right
                if drop_col in selected:
                    selected.remove(drop_col)

    return selected + categorical_features


def encode_for_univariate(train_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    encoded = pd.DataFrame(index=train_df.index)
    for column in features:
        series = train_df[column]
        if pd.api.types.is_numeric_dtype(series):
            encoded[column] = series.fillna(series.median())
        else:
            values, _ = pd.factorize(series.fillna("missing"), sort=True)
            encoded[column] = values
    return encoded


def filter_by_univariate_fraction(train_df: pd.DataFrame, features: list[str], fraction: float) -> list[str]:
    encoded = encode_for_univariate(train_df, features)
    discrete = [not pd.api.types.is_numeric_dtype(train_df[col]) for col in features]
    scores = mutual_info_classif(encoded, train_df[TARGET_COLUMN], discrete_features=discrete, random_state=42)
    score_series = pd.Series(scores, index=features).sort_values(ascending=False)
    keep_n = max(1, int(np.ceil(len(features) * fraction)))
    return score_series.head(keep_n).index.tolist()


def build_experiment_table(train_df: pd.DataFrame, val_df: pd.DataFrame, base_features: list[str]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    configs = [
        ExperimentConfig(name="initial", method="initial"),
        ExperimentConfig(name="missing_le_0.10", method="missing", threshold=0.10),
        ExperimentConfig(name="corr_le_0.95", method="correlation", threshold=0.95),
        ExperimentConfig(name="corr_le_0.90", method="correlation", threshold=0.90),
        ExperimentConfig(name="corr_le_0.85", method="correlation", threshold=0.85),
        ExperimentConfig(name="univariate_top_10pct", method="univariate", threshold=0.10),
        ExperimentConfig(name="univariate_top_20pct", method="univariate", threshold=0.20),
        ExperimentConfig(name="univariate_top_30pct", method="univariate", threshold=0.30),
    ]

    rows: list[dict[str, object]] = []
    feature_sets: dict[str, list[str]] = {}

    for config in configs:
        if config.method == "initial":
            features = base_features.copy()
        elif config.method == "missing":
            features = filter_by_missing(train_df, base_features, config.threshold or 0.0)
        elif config.method == "correlation":
            features = filter_by_correlation(train_df, base_features, config.threshold or 1.0)
        elif config.method == "univariate":
            features = filter_by_univariate_fraction(train_df, base_features, config.threshold or 1.0)
        else:
            raise ValueError(f"Metodo no soportado: {config.method}")

        metrics = evaluate_feature_set(train_df, val_df, features)
        feature_sets[config.name] = features
        rows.append(
            {
                "experiment": config.name,
                "method": config.method,
                "threshold": config.threshold,
                "n_features": len(features),
                "auc_train": round(metrics["auc_train"], 6),
                "auc_val": round(metrics["auc_val"], 6),
                "gini_train": round(metrics["gini_train"], 6),
                "gini_val": round(metrics["gini_val"], 6),
                "features": features,
            }
        )

    results = pd.DataFrame(rows).sort_values(["auc_val", "auc_train"], ascending=[False, False]).reset_index(drop=True)
    return results, feature_sets


def choose_best_experiment(results: pd.DataFrame) -> str:
    ranked = results.copy()
    ranked["overfit_gap"] = ranked["auc_train"] - ranked["auc_val"]
    best_auc = float(ranked["auc_val"].max())
    tolerance = 0.002
    # No elegimos solo por AUC maximo: primero armamos una "zona aceptable"
    # cercana al mejor resultado y luego privilegiamos parsimonia y estabilidad.
    candidates = ranked[ranked["auc_val"] >= (best_auc - tolerance)].copy()
    candidates = candidates.sort_values(
        ["n_features", "overfit_gap", "auc_val"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return str(candidates.loc[0, "experiment"])


def save_markdown_summary(results: pd.DataFrame, best_experiment: str, output_path: Path) -> None:
    lines = [
        "# Experimentos de Seleccion de Features",
        "",
        f"- Mejor experimento seleccionado: `{best_experiment}`",
        "",
        "| Experimento | Metodo | Threshold | # Features | AUC Train | AUC Val | Gini Train | Gini Val |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        threshold = "" if pd.isna(row.threshold) else row.threshold
        lines.append(
            f"| `{row.experiment}` | `{row.method}` | {threshold} | {row.n_features} | {row.auc_train:.4f} | {row.auc_val:.4f} | {row.gini_train:.4f} | {row.gini_val:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Detalle de features por experimento",
            "",
        ]
    )
    for row in results.itertuples(index=False):
        lines.append(f"### `{row.experiment}`")
        lines.append("")
        lines.append(f"- Metodo: `{row.method}`")
        lines.append(f"- Threshold: `{'' if pd.isna(row.threshold) else row.threshold}`")
        lines.append(f"- # Features: `{row.n_features}`")
        lines.append(f"- AUC Train: `{row.auc_train:.4f}`")
        lines.append(f"- AUC Val: `{row.auc_val:.4f}`")
        lines.append(f"- Gini Train: `{row.gini_train:.4f}`")
        lines.append(f"- Gini Val: `{row.gini_val:.4f}`")
        lines.append("- Features:")
        for feature in row.features:
            lines.append(f"  - `{feature}`")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experimentos de seleccion de variables para Sprint 2.")
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--results-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--selected-output-path", type=Path, default=None)
    parser.add_argument("--selected-metadata-path", type=Path, default=None)
    parser.add_argument("--validation-cutoff", type=str, default="2018-07-01")
    parser.add_argument("--max-train-rows", type=int, default=30000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()
    input_path = args.input_path or (project_root / "data" / "processed" / "05_features_rfm.parquet")
    results_path = args.results_path or (project_root / "data" / "processed" / "06_feature_selection_experiments.parquet")
    metadata_path = args.metadata_path or (project_root / "data" / "processed" / "06_feature_selection_experiments.json")
    summary_path = args.summary_path or (project_root / "reports" / "sprint_02" / "feature_selection_experiments.md")
    selected_output_path = args.selected_output_path or (project_root / "data" / "processed" / "06_features_selected.parquet")
    selected_metadata_path = args.selected_metadata_path or (project_root / "data" / "processed" / "06_features_selected_metadata.json")

    df = pd.read_parquet(input_path).copy()
    cutoff = pd.Timestamp(args.validation_cutoff)
    train_df = df[df["last_purchase"] < cutoff].copy()
    val_df = df[df["last_purchase"] >= cutoff].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("El corte temporal dejo train o validation vacio. Ajusta validation-cutoff.")
    if args.max_train_rows and len(train_df) > args.max_train_rows:
        # La bateria experimental prueba 8 configuraciones; limitamos train para
        # acelerar comparaciones sin alterar la distribucion de la clase.
        _, train_df = train_test_split(
            train_df,
            train_size=args.max_train_rows,
            stratify=train_df[TARGET_COLUMN],
            random_state=42,
        )

    base_features = mandatory_base_features(df)
    results, feature_sets = build_experiment_table(train_df, val_df, base_features)
    best_experiment = choose_best_experiment(results)
    best_features = feature_sets[best_experiment]

    selected_columns = IDENTIFIER_COLUMNS + [TARGET_COLUMN] + best_features
    selected_df = df[selected_columns].copy()

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(results_path, engine="pyarrow", compression="snappy", index=False)
    selected_df.to_parquet(selected_output_path, engine="pyarrow", compression="snappy", index=False)
    save_markdown_summary(results, best_experiment, summary_path)

    payload = {
        "input_path": str(input_path),
        "validation_cutoff": args.validation_cutoff,
        "max_train_rows": args.max_train_rows,
        "base_feature_count": len(base_features),
        "best_experiment": best_experiment,
        "best_feature_count": len(best_features),
        "best_features": best_features,
        "mandatory_exclude": MANDATORY_EXCLUDE,
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    selected_payload = {
        "source": "src/features/feature_selection_experiments.py",
        "selection_type": "experimental",
        "input_path": str(input_path),
        "selected_output_path": str(selected_output_path),
        "validation_cutoff": args.validation_cutoff,
        "max_train_rows": args.max_train_rows,
        "best_experiment": best_experiment,
        "selected_feature_count": len(best_features),
        "selected_model_columns": best_features,
        "mandatory_exclude": MANDATORY_EXCLUDE,
        "selection_rule": "Best validation AUC within 0.002 tolerance, then fewer features and lower overfit gap.",
    }
    selected_metadata_path.write_text(json.dumps(selected_payload, indent=2), encoding="utf-8")

    print(f"Input path: {input_path}")
    print(f"Results path: {results_path}")
    print(f"Selected output path: {selected_output_path}")
    print(f"Selected metadata path: {selected_metadata_path}")
    print(f"Summary path: {summary_path}")
    print(f"Best experiment: {best_experiment}")
    print(f"Best feature count: {len(best_features)}")
    print(results[["experiment", "method", "threshold", "n_features", "auc_train", "auc_val", "gini_train", "gini_val"]].to_string(index=False))


if __name__ == "__main__":
    main()
