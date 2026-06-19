from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "final" / "modelo_final.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "backtest_features_selected.parquet"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "06_features_selected_metadata.json"
OUT_TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "backtest_lift_table.csv"
OUT_SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "backtest_lift_summary.json"
OUT_FIG_PATH = PROJECT_ROOT / "reports" / "figures" / "sprint_04_backtest_gain_lift.png"


def build_lift_table(y_true: pd.Series, probs: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    eval_df = pd.DataFrame(
        {
            "is_premium": y_true.astype(int).to_numpy(),
            "prob": probs,
        }
    ).sort_values("prob", ascending=False, kind="mergesort").reset_index(drop=True)

    n = len(eval_df)
    eval_df["decile"] = np.ceil((np.arange(1, n + 1) / n) * n_bins).astype(int)
    eval_df["decile"] = eval_df["decile"].clip(upper=n_bins)

    total_events = int(eval_df["is_premium"].sum())
    base_rate = total_events / n

    grouped = (
        eval_df.groupby("decile", sort=True)
        .agg(
            cases=("is_premium", "size"),
            responses=("is_premium", "sum"),
            min_score=("prob", "min"),
            max_score=("prob", "max"),
            avg_score=("prob", "mean"),
        )
        .reset_index()
    )

    grouped["response_rate"] = grouped["responses"] / grouped["cases"]
    grouped["premium_rate_pct"] = grouped["response_rate"] * 100
    grouped["cum_cases"] = grouped["cases"].cumsum()
    grouped["cum_responses"] = grouped["responses"].cumsum()
    grouped["cum_cases_pct"] = grouped["cum_cases"] / n
    grouped["cum_events_pct"] = grouped["cum_responses"] / total_events
    grouped["cum_events_pct_display"] = grouped["cum_events_pct"] * 100
    grouped["gain_pct"] = grouped["cum_events_pct"] * 100
    grouped["lift"] = grouped["response_rate"] / base_rate
    grouped["cum_lift"] = grouped["cum_events_pct"] / grouped["cum_cases_pct"]

    return grouped[
        [
            "decile",
            "cases",
            "responses",
            "response_rate",
            "premium_rate_pct",
            "cum_cases",
            "cum_responses",
            "cum_cases_pct",
            "cum_events_pct",
            "cum_events_pct_display",
            "gain_pct",
            "lift",
            "cum_lift",
            "min_score",
            "max_score",
            "avg_score",
        ]
    ]


def plot_gain_lift(table: pd.DataFrame, out_path: Path) -> None:
    x = table["cum_cases_pct"] * 100
    gains = table["cum_events_pct"] * 100
    lifts = table["cum_lift"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=False)

    axes[0].bar(table["decile"].astype(str), table["responses"], color="#5b7c99", edgecolor="white")
    axes[0].set_ylabel("Premium reales")
    axes[0].set_title("Responses por Decil - Backtest Sprint 4")
    axes[0].grid(axis="y", alpha=0.25, linestyle=":")

    axes[1].plot(x, gains, marker="o", linewidth=2.2, color="#1565C0", label="Modelo")
    axes[1].plot([0, 100], [0, 100], linestyle="--", color="#C0392B", alpha=0.7, label="Aleatorio")
    axes[1].set_ylabel("% de premium acumulado")
    axes[1].set_title("Cumulative Gain Chart - Backtest Sprint 4")
    axes[1].legend()
    axes[1].grid(alpha=0.25, linestyle=":")

    axes[2].plot(x, lifts, marker="o", linewidth=2.2, color="#2E7D32", label="Lift acumulado")
    axes[2].plot([0, 100], [1, 1], linestyle="--", color="#C0392B", alpha=0.7, label="Aleatorio")
    axes[2].set_xlabel("% de la base priorizada")
    axes[2].set_ylabel("Lift")
    axes[2].set_title("Cumulative Lift Chart - Backtest Sprint 4")
    axes[2].legend()
    axes[2].grid(alpha=0.25, linestyle=":")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    with open(METADATA_PATH) as f:
        selected_cols = json.load(f)["selected_model_columns"]
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    X = df[selected_cols]
    y = df["is_premium"]
    probs = pipeline.predict_proba(X)[:, 1]

    table = build_lift_table(y, probs, n_bins=10)
    table.to_csv(OUT_TABLE_PATH, index=False)

    summary = {
        "n_total": int(len(df)),
        "n_events": int(y.sum()),
        "base_rate": float(y.mean()),
        "top_10_capture_pct": float(table.loc[table["decile"] == 1, "cum_events_pct"].iloc[0]),
        "top_20_capture_pct": float(table.loc[table["decile"] == 2, "cum_events_pct"].iloc[0]),
        "top_10_premium_rate_pct": float(table.loc[table["decile"] == 1, "premium_rate_pct"].iloc[0]),
        "decile_1_lift": float(table.loc[table["decile"] == 1, "lift"].iloc[0]),
        "cum_lift_top_10": float(table.loc[table["decile"] == 1, "cum_lift"].iloc[0]),
        "cum_lift_top_20": float(table.loc[table["decile"] == 2, "cum_lift"].iloc[0]),
    }
    with open(OUT_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    plot_gain_lift(table, OUT_FIG_PATH)

    print(f"Lift table saved to: {OUT_TABLE_PATH}")
    print(f"Lift summary saved to: {OUT_SUMMARY_PATH}")
    print(f"Gain/Lift figure saved to: {OUT_FIG_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
