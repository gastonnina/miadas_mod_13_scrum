from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Project root ────────────────────────────────────────────────────────────
def _find_project_root() -> Path:
    candidate = Path(__file__).resolve()
    for parent in [candidate, *candidate.parents]:
        if (parent / "models").exists() and (parent / "data").exists():
            return parent
    raise FileNotFoundError(
        f"No se encontro la raiz del proyecto desde {candidate}. "
        "Asegurate de correr streamlit desde el directorio del proyecto."
    )


PROJECT_ROOT = _find_project_root()
MODEL_PATH = PROJECT_ROOT / "models" / "final" / "modelo_final.pkl"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "06_features_selected_metadata.json"
DEMO_SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "demo_sample_scoring.parquet"

THRESHOLD = 0.55
COLOR_PREMIUM = "#39a96b"
COLOR_RISK = "#d95d39"
COLOR_NEUTRAL = "#5b7c99"
COLOR_TEXT = "#dce6f0"
COLOR_MUTED = "#7a8fa6"
COLOR_CARD = "#141b24"
COLOR_ELEVATED = "#1c2636"

HOLDOUT_METRICS = {
    "roc_auc": 0.9872,
    "gini": 0.9745,
    "precision": 0.4310,
    "recall": 0.5706,
    "f1": 0.4911,
    "n_total": 96_096,
    "n_premium": 1_253,
    "premium_rate": 0.013,
    "total_spend": 985_414.28,
    "premium_spend": 521_001.48,
    "premium_spend_pct": 0.5287,
    "tp_spend": 333_999.80,
    "roi_masiva": -89.57,
    "roi_modelo": 244.79,
    "cost_savings": 1_416_555.00,
    "n_pred_premium": 1_659,
    "tp_count": 715,
}

BACKTEST_METRICS = {
    "roc_auc": 0.9876,
    "gini": 0.9751,
    "precision": 0.4380,
    "recall": 0.5611,
    "f1": 0.4920,
    "n_total": 96_096,
    "n_premium": 1_253,
    "premium_rate": 0.013,
    "total_spend": 985_414.28,
    "premium_spend": 521_001.48,
    "premium_spend_pct": 0.5287,
    "tp_spend": 328_003.99,
    "roi_masiva": -89.57,
    "roi_modelo": 250.40,
    "cost_savings": 1_417_365.00,
    "n_pred_premium": 1_605,
    "tp_count": 703,
}

SPRINT3_VALIDATION_METRICS = {
    "roc_auc": 0.8081,
    "gini": 0.6162,
    "pr_auc": 0.6010,
    "precision": 0.5225,
    "recall": 0.6041,
    "f1": 0.5603,
    "threshold": 0.55,
}

DATASET_CONFIGS: dict[str, dict] = {
    "Holdout (validacion final)": {
        "features_file": "holdout_features_selected.parquet",
        "master_file": "03_master_table_clean_holdout.parquet",
        "metrics": HOLDOUT_METRICS,
        "label": "holdout_3m · 96 096 clientes",
        "tab_title": "Holdout y ROI",
        "has_lift": False,
    },
    "Backtest (ago 2018)": {
        "features_file": "backtest_features_selected.parquet",
        "master_file": "03_master_table_clean_backtest.parquet",
        "metrics": BACKTEST_METRICS,
        "label": "backtest agosto 2018",
        "tab_title": "Backtest y ROI",
        "has_lift": True,
    },
}

# ── Plotly dark theme ───────────────────────────────────────────────────────
_PBG = dict(
    plot_bgcolor=COLOR_CARD,
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLOR_TEXT, family="Space Mono, monospace", size=11),
    margin=dict(l=10, r=10, t=44, b=10),
    hoverlabel=dict(bgcolor="#1c2636", bordercolor="#2e3d52", font=dict(color=COLOR_TEXT, size=11)),
)
_AX = dict(
    gridcolor="rgba(176,212,255,0.05)",
    linecolor="rgba(176,212,255,0.09)",
    tickfont=dict(size=10, color=COLOR_MUTED),
    title_font=dict(size=10, color=COLOR_MUTED),
    zerolinecolor="rgba(176,212,255,0.12)",
)

# ── CSS: Intelligence Terminal aesthetic ────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

:root {
    --bg:       #0c1117;
    --card:     #141b24;
    --elevated: #1c2636;
    --border:   rgba(148,180,220,0.09);
    --green:    #39a96b;
    --red:      #d95d39;
    --blue:     #5b7c99;
    --text:     #dce6f0;
    --muted:    #7a8fa6;
    --syne:     'Syne', sans-serif;
    --mono:     'Space Mono', monospace;
}

/* ── Global ── */
.stApp, .stApp > div, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
}
.main .block-container {
    background: transparent !important;
    padding-top: 1.2rem !important;
    max-width: 1360px !important;
}
h1, h2, h3, h4 { font-family: var(--syne) !important; color: var(--text) !important; }
p, li, span, label, div { color: var(--text); }
.stCaption p { color: var(--muted) !important; font-family: var(--mono) !important; font-size: 0.68rem !important; }
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label { color: var(--text) !important; }
section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: var(--elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.65rem 0.9rem !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.85rem 1rem !important;
}
[data-testid="stMetricLabel"] p {
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 1.25rem !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] {
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    font-family: var(--syne) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.15s ease !important;
}
.stTabs [aria-selected="true"] {
    background: var(--elevated) !important;
    color: var(--text) !important;
}
[data-testid="stTabPanel"] { padding-top: 1.5rem !important; }

/* ── Buttons ── */
.stButton button {
    background: var(--elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
}

/* ── Inputs & widgets ── */
.stSelectbox [data-baseweb="select"] > div { background: var(--card) !important; border-color: var(--border) !important; }
.stTextInput input { background: var(--card) !important; border-color: var(--border) !important; color: var(--text) !important; }
.stSlider [data-baseweb="slider"] { background: var(--elevated) !important; }
.stRadio label { color: var(--text) !important; font-family: var(--mono) !important; font-size: 0.78rem !important; }

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 10px !important; }
.stInfo  { background: rgba(91,124,153,0.1)  !important; border: 1px solid rgba(91,124,153,0.3)  !important; }
.stWarning { background: rgba(217,93,57,0.08) !important; border: 1px solid rgba(217,93,57,0.3)  !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Expanders ── */
.streamlit-expanderHeader {
    background: var(--card) !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
}
.streamlit-expanderContent {
    background: var(--card) !important;
    border-color: var(--border) !important;
}

/* ── Custom HTML components ── */

/* Score hero */
.score-hero {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    display: flex;
    align-items: center;
    gap: 2.5rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}
.score-hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 280px; height: 280px;
    border-radius: 50%;
    pointer-events: none;
}
.score-hero.premium::before { background: radial-gradient(circle, rgba(57,169,107,0.10) 0%, transparent 70%); }
.score-hero.regular::before { background: radial-gradient(circle, rgba(217,93,57,0.08) 0%, transparent 70%); }

.sh-prob { flex: 0 0 auto; display: flex; flex-direction: column; align-items: flex-start; }
.sh-prob-num { font-family: var(--mono); font-size: 4.2rem; font-weight: 700; line-height: 1; letter-spacing: -0.02em; }
.sh-prob-num.premium { color: var(--green); }
.sh-prob-num.regular { color: var(--red); }
.sh-prob-lbl { font-family: var(--mono); font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-top: 0.25rem; }

.sh-center { flex: 1; min-width: 0; }
.sh-classification {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.3rem 0.85rem;
    border-radius: 999px;
    font-family: var(--mono); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em;
    margin-bottom: 0.8rem;
}
.sh-classification.premium { background: rgba(57,169,107,0.14); color: var(--green); border: 1px solid rgba(57,169,107,0.32); }
.sh-classification.regular { background: rgba(217,93,57,0.12); color: var(--red); border: 1px solid rgba(217,93,57,0.32); }

.sh-bar-wrap { height: 5px; background: var(--elevated); border-radius: 3px; overflow: hidden; margin-bottom: 0.3rem; }
.sh-bar-fill { height: 100%; border-radius: 3px; }
.sh-threshold { font-family: var(--mono); font-size: 0.62rem; color: var(--muted); }

.sh-right { flex: 0 0 auto; display: flex; flex-direction: column; gap: 0.55rem; border-left: 1px solid var(--border); padding-left: 2rem; }
.sh-meta-row { display: flex; flex-direction: column; gap: 0.05rem; }
.sh-meta-lbl { font-family: var(--mono); font-size: 0.58rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.sh-meta-val { font-family: var(--mono); font-size: 0.78rem; color: var(--text); font-weight: 700; }
.sh-meta-val.green { color: var(--green); }
.sh-meta-val.red { color: var(--red); }
.sh-meta-val.muted { color: var(--muted); }

/* Section eyebrow */
.eyebrow {
    display: flex; align-items: center; gap: 0.6rem;
    margin: 1.75rem 0 0.6rem 0;
}
.eyebrow-num { font-family: var(--mono); font-size: 0.58rem; color: var(--muted); border: 1px solid var(--border); border-radius: 4px; padding: 0.15rem 0.45rem; letter-spacing: 0.06em; }
.eyebrow-line { flex: 1; height: 1px; background: var(--border); }
.eyebrow-label { font-family: var(--mono); font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); }

/* Profile card */
.profile-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    height: 100%;
}
.profile-title { font-family: var(--syne); font-size: 0.78rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 1rem; }
.profile-stat { display: flex; flex-direction: column; gap: 0.1rem; padding: 0.6rem 0; border-bottom: 1px solid var(--border); }
.profile-stat:last-child { border-bottom: none; }
.profile-stat-lbl { font-family: var(--mono); font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.profile-stat-val { font-family: var(--mono); font-size: 0.95rem; font-weight: 700; color: var(--text); }

/* Truth badge */
.truth-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-family: var(--mono); font-size: 0.68rem; font-weight: 700;
    margin-top: 0.5rem;
}
.truth-badge.ok  { background: rgba(57,169,107,0.1); color: var(--green); border: 1px solid rgba(57,169,107,0.3); }
.truth-badge.err { background: rgba(217,93,57,0.1);  color: var(--red);   border: 1px solid rgba(217,93,57,0.3);  }

/* Narrative block */
.narrative {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--blue);
    border-radius: 0 10px 10px 0;
    padding: 0.85rem 1.2rem;
    margin: 0.6rem 0 1.2rem 0;
}
.narrative.green { border-left-color: var(--green); }
.narrative.red   { border-left-color: var(--red);   }
.narrative-t { font-family: var(--syne); font-size: 0.82rem; font-weight: 600; color: var(--text); margin-bottom: 0.3rem; }
.narrative-b { font-family: var(--mono); font-size: 0.7rem; color: var(--muted); line-height: 1.7; }

/* ROI comparison inline */
.roi-compare { display: flex; gap: 1rem; margin: 0.75rem 0; }
.roi-block {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    display: flex; flex-direction: column; gap: 0.2rem;
}
.roi-block.good  { border-top: 3px solid var(--green); }
.roi-block.bad   { border-top: 3px solid var(--red);   }
.roi-block-lbl   { font-family: var(--mono); font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
.roi-block-val   { font-family: var(--mono); font-size: 1.6rem; font-weight: 700; }
.roi-block-val.good { color: var(--green); }
.roi-block-val.bad  { color: var(--red);   }
.roi-block-sub   { font-family: var(--mono); font-size: 0.65rem; color: var(--muted); }
</style>
"""


# ── Cache de recursos ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_scoring_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def load_metadata() -> dict:
    with open(METADATA_PATH) as f:
        return json.load(f)


@st.cache_data
def load_master_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data
def load_demo_sample() -> pd.DataFrame:
    return pd.read_parquet(DEMO_SAMPLE_PATH)


# ── Helpers ─────────────────────────────────────────────────────────────────
def compute_contributions(pipeline, X_row: pd.DataFrame) -> pd.Series:
    preprocessor = pipeline.named_steps["preprocessor"]
    lgb_model = pipeline.named_steps["model"]
    X_proc = preprocessor.transform(X_row)
    feature_names = preprocessor.get_feature_names_out()
    X_proc_df = pd.DataFrame(X_proc, columns=feature_names, index=X_row.index)
    contribs = lgb_model.predict(X_proc_df, pred_contrib=True)
    clean_names = [n.replace("num__", "").replace("cat__", "") for n in feature_names]
    return pd.Series(contribs[0, :-1], index=clean_names).sort_values(ascending=False)


def safe_int(value, default: str = "N/A"):
    if value is None or pd.isna(value):
        return default
    return int(value)


def safe_float_metric(value, default: str = "N/A") -> str:
    if value is None or pd.isna(value):
        return default
    return f"{float(value):,.2f}"


@st.cache_data(show_spinner="Calculando scores del dataset...")
def build_selector_frame(_pipeline, df: pd.DataFrame, selected_cols: list[str]) -> pd.DataFrame:
    enriched = df[["customer_unique_id", "is_premium"]].copy()
    enriched["row_idx"] = df.index.astype(int)
    enriched["premium_probability"] = _pipeline.predict_proba(df[selected_cols])[:, 1]
    enriched["is_pred_premium"] = enriched["premium_probability"] >= THRESHOLD
    enriched["selection_bucket"] = "Todos"
    enriched.loc[enriched["is_premium"] == 1, "selection_bucket"] = "Premium reales"
    enriched.loc[enriched["is_pred_premium"], "selection_bucket"] = "Premium predichos"
    return enriched


def format_selector_option(option_row: pd.Series) -> str:
    real_label = "PREM" if int(option_row["is_premium"]) == 1 else "REG"
    pred_label = "PREM" if bool(option_row["is_pred_premium"]) else "REG"
    return (
        f"{int(option_row['row_idx']):>5} | "
        f"{str(option_row['customer_unique_id'])[:12]}... | "
        f"prob={option_row['premium_probability']:.1%} | "
        f"real={real_label} | pred={pred_label}"
    )


def get_confusion_counts(metrics: dict) -> dict[str, int]:
    tp = int(metrics["tp_count"])
    fp = int(metrics["n_pred_premium"] - tp)
    fn = int(metrics["n_premium"] - tp)
    tn = int(metrics["n_total"] - tp - fp - fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


@st.cache_data(show_spinner="Calculando tabla lift/gain...")
def build_lift_table(_pipeline, df: pd.DataFrame, selected_cols: list[str]) -> pd.DataFrame:
    scored = pd.DataFrame(
        {
            "is_premium": df["is_premium"].astype(int),
            "total_orders": df["total_orders"].fillna(0),
            "prob": _pipeline.predict_proba(df[selected_cols])[:, 1],
        }
    )
    scored = scored[scored["total_orders"] > 0].sort_values("prob", ascending=False, kind="mergesort").reset_index(drop=True)

    n_rows = len(scored)
    scored["decile"] = (
        pd.Series(range(1, n_rows + 1))
        .div(n_rows)
        .mul(10)
        .apply(lambda x: int(-(-x // 1)))
    )
    total_events = int(scored["is_premium"].sum())
    base_rate = scored["is_premium"].mean()

    lift_table = (
        scored.groupby("decile", sort=True)
        .agg(
            cases=("is_premium", "size"),
            responses=("is_premium", "sum"),
            min_score=("prob", "min"),
            max_score=("prob", "max"),
            avg_score=("prob", "mean"),
        )
        .reset_index()
    )
    lift_table["response_rate"] = lift_table["responses"] / lift_table["cases"]
    lift_table["premium_rate_pct"] = lift_table["response_rate"] * 100
    lift_table["cum_cases"] = lift_table["cases"].cumsum()
    lift_table["cum_responses"] = lift_table["responses"].cumsum()
    lift_table["cum_cases_pct"] = lift_table["cum_cases"] / n_rows
    lift_table["cum_events_pct"] = lift_table["cum_responses"] / total_events
    lift_table["gain_pct"] = lift_table["cum_events_pct"] * 100
    lift_table["lift"] = lift_table["response_rate"] / base_rate
    lift_table["cum_lift"] = lift_table["cum_events_pct"] / lift_table["cum_cases_pct"]
    lift_table.attrs["base_rate"] = float(base_rate)
    lift_table.attrs["n_rows"] = int(n_rows)
    lift_table.attrs["total_events"] = int(total_events)
    return lift_table


# ── Presentation helpers ─────────────────────────────────────────────────────

def eyebrow(num: str, label: str) -> None:
    st.markdown(
        f'<div class="eyebrow">'
        f'<span class="eyebrow-num">{num}</span>'
        f'<span class="eyebrow-line"></span>'
        f'<span class="eyebrow-label">{label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def narrative(title: str, body: str, color: str = "") -> None:
    cls = f"narrative {color}".strip()
    st.markdown(
        f'<div class="{cls}"><div class="narrative-t">{title}</div>'
        f'<div class="narrative-b">{body}</div></div>',
        unsafe_allow_html=True,
    )


def render_score_hero(
    prob: float,
    is_pred_premium: bool,
    cid: str,
    is_true_premium: int | None,
) -> None:
    cls = "premium" if is_pred_premium else "regular"
    label = "★ CLIENTE PREMIUM" if is_pred_premium else "◇ NO PREMIUM"
    bar_color = COLOR_PREMIUM if is_pred_premium else COLOR_RISK
    bar_width = f"{prob * 100:.1f}%"
    prob_str = f"{prob:.1%}"

    truth_html = ""
    if is_true_premium is not None:
        real = "PREMIUM" if is_true_premium else "NO PREMIUM"
        correct = is_pred_premium == bool(is_true_premium)
        truth_cls = "ok" if correct else "err"
        truth_icon = "✓" if correct else "✗"
        truth_html = (
            f'<div class="truth-badge {truth_cls}">'
            f'{truth_icon}&nbsp;Etiqueta real: {real}'
            f'</div>'
        )

    st.markdown(
        f"""
        <div class="score-hero {cls}">
          <div class="sh-prob">
            <span class="sh-prob-num {cls}">{prob_str}</span>
            <span class="sh-prob-lbl">probabilidad premium</span>
          </div>

          <div class="sh-center">
            <div class="sh-classification {cls}">{label}</div>
            {truth_html}
            <div class="sh-bar-wrap" style="margin-top:0.9rem;">
              <div class="sh-bar-fill" style="width:{bar_width}; background:{bar_color};"></div>
            </div>
            <p class="sh-threshold">umbral de decisión: {THRESHOLD:.0%}
              &nbsp;·&nbsp; modelo: LightGBM + ColumnTransformer
            </p>
          </div>

          <div class="sh-right">
            <div class="sh-meta-row">
              <span class="sh-meta-lbl">cliente ID</span>
              <span class="sh-meta-val muted">{cid[:20]}…</span>
            </div>
            <div class="sh-meta-row">
              <span class="sh-meta-lbl">score crudo</span>
              <span class="sh-meta-val">{prob:.6f}</span>
            </div>
            <div class="sh-meta-row">
              <span class="sh-meta-lbl">umbral</span>
              <span class="sh-meta-val">{THRESHOLD:.2f}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_profile_card(info_row: pd.DataFrame, feat_row: pd.DataFrame) -> None:
    if info_row.empty:
        st.markdown(
            '<div class="profile-card"><p class="profile-title">Perfil no disponible</p></div>',
            unsafe_allow_html=True,
        )
        return
    info = info_row.iloc[0]
    recency = feat_row["recency_days"].values[0] if "recency_days" in feat_row.columns else None
    stats = [
        ("Gasto total (BRL)", safe_float_metric(info.get("total_spent", 0))),
        ("Órdenes",           str(safe_int(info.get("total_orders", 0), default="0"))),
        ("Ticket promedio",   safe_float_metric(info.get("avg_ticket", 0))),
        ("Estado",            str(info.get("customer_state", "N/A"))),
        ("Recencia (días)",   str(safe_int(recency))),
    ]
    rows_html = "".join(
        f'<div class="profile-stat">'
        f'<span class="profile-stat-lbl">{lbl}</span>'
        f'<span class="profile-stat-val">{val}</span>'
        f'</div>'
        for lbl, val in stats
    )
    st.markdown(
        f'<div class="profile-card"><p class="profile-title">Perfil del cliente</p>{rows_html}</div>',
        unsafe_allow_html=True,
    )


def render_contribution_chart(contrib: pd.Series, title: str) -> None:
    top5 = contrib.head(5)
    bot5 = contrib.tail(5)
    combined = pd.concat([top5, bot5.iloc[::-1]])
    colors = [COLOR_PREMIUM if v >= 0 else COLOR_RISK for v in combined]

    fig = go.Figure(go.Bar(
        x=combined.values,
        y=combined.index,
        orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>Contribución: %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=COLOR_NEUTRAL, line_width=1)
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color=COLOR_MUTED)),
        xaxis_title="Contribución al score (log-odds)",
        yaxis=dict(autorange="reversed", **_AX),
        xaxis=dict(**_AX),
        height=360,
        **_PBG,
    )
    st.plotly_chart(fig, width="stretch")


def render_confusion_matrix(cm: dict[str, int], label: str) -> None:
    cm_matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    x_labels = ["Pred: NO PREMIUM", "Pred: PREMIUM"]
    y_labels = ["Real: NO PREMIUM", "Real: PREMIUM"]

    fig = go.Figure(go.Heatmap(
        z=cm_matrix,
        x=x_labels,
        y=y_labels,
        colorscale="Blues",
        showscale=False,
        hovertemplate="%{y} → %{x}<br>Conteo: %{z:,}<extra></extra>",
    ))
    max_cell = max(cm_matrix[0] + cm_matrix[1])
    for i in range(2):
        for j in range(2):
            val = cm_matrix[i][j]
            txt_color = "white" if val >= max_cell * 0.55 else "#1a2332"
            fig.add_annotation(
                x=x_labels[j], y=y_labels[i],
                text=f"<b>{val:,}</b>",
                showarrow=False,
                font=dict(size=15, color=txt_color),
            )
    fig.update_xaxes(showgrid=False, linecolor="rgba(0,0,0,0)", tickfont=dict(size=10, color=COLOR_MUTED))
    fig.update_yaxes(showgrid=False, linecolor="rgba(0,0,0,0)", tickfont=dict(size=10, color=COLOR_MUTED))
    fig.update_layout(
        title=dict(text=label, font=dict(size=12, color=COLOR_MUTED)),
        height=300,
        **_PBG,
    )
    st.plotly_chart(fig, width="stretch")


def render_roi_bars(m: dict) -> None:
    labels = ["Campaña Masiva<br>(target all)", "Campaña Optimizada<br>(modelo · umbral 0.55)"]
    rois = [m["roi_masiva"], m["roi_modelo"]]
    colors = [COLOR_RISK, COLOR_PREMIUM]
    subtexts = [f"{m['n_total']:,} clientes", f"{m['n_pred_premium']:,} clientes"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=rois,
        marker_color=colors,
        text=[f"<b>{v:.2f}%</b>" for v in rois],
        textposition="outside",
        textfont=dict(color=COLOR_TEXT, size=12),
        customdata=subtexts,
        hovertemplate="<b>%{x}</b><br>ROI: %{y:.2f}%<br>%{customdata}<extra></extra>",
        width=0.5,
    ))
    fig.add_hline(y=0, line_color=COLOR_NEUTRAL, line_width=1)
    fig.update_xaxes(tickfont=dict(size=10, color=COLOR_MUTED), linecolor="rgba(0,0,0,0)", showgrid=False)
    fig.update_yaxes(title_text="ROI (%)", **_AX)
    fig.update_layout(
        title=dict(text="ROI por estrategia de campaña", font=dict(size=12, color=COLOR_MUTED)),
        height=330,
        **_PBG,
    )
    st.plotly_chart(fig, width="stretch")


def render_lift_charts(lift_table: pd.DataFrame) -> None:
    x_pct  = lift_table["cum_cases_pct"] * 100
    gains  = lift_table["cum_events_pct"] * 100
    c_lift = lift_table["cum_lift"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        subplot_titles=[
            "Respuestas por decil - clientes activos",
            "Curva de ganancia acumulada - clientes activos",
            "Lift acumulado - clientes activos",
        ],
        vertical_spacing=0.09,
    )

    # Row 1 — responses per decile
    fig.add_trace(
        go.Bar(
            x=lift_table["decile"].astype(str),
            y=lift_table["responses"],
            marker_color=COLOR_NEUTRAL,
            name="Premium reales",
            hovertemplate="Decil %{x} → <b>%{y}</b> premium<extra></extra>",
        ),
        row=1, col=1,
    )

    # Row 2 — cumulative gains
    fig.add_trace(
        go.Scatter(
            x=x_pct, y=gains,
            mode="lines+markers",
            line=dict(color="#4fa3e0", width=2.2),
            marker=dict(size=6, color="#4fa3e0"),
            name="Modelo",
            hovertemplate="Top %{x:.1f}% → captura %{y:.1f}% premium<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 100], y=[0, 100],
            mode="lines",
            line=dict(color=COLOR_RISK, dash="dash", width=1.2),
            name="Aleatorio",
            opacity=0.6,
            hoverinfo="skip",
        ),
        row=2, col=1,
    )

    # Row 3 — cumulative lift
    fig.add_trace(
        go.Scatter(
            x=x_pct, y=c_lift,
            mode="lines+markers",
            line=dict(color=COLOR_PREMIUM, width=2.2),
            marker=dict(size=6, color=COLOR_PREMIUM),
            name="Lift acumulado",
            hovertemplate="Top %{x:.1f}% → lift %{y:.2f}x<extra></extra>",
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 100], y=[1, 1],
            mode="lines",
            line=dict(color=COLOR_RISK, dash="dash", width=1.2),
            name="Aleatorio (lift=1)",
            opacity=0.6,
            hoverinfo="skip",
        ),
        row=3, col=1,
    )

    for r in range(1, 4):
        fig.update_xaxes(row=r, col=1, **_AX)
        fig.update_yaxes(row=r, col=1, **_AX)
    fig.update_xaxes(title_text="% de la base activa priorizada", row=3, col=1, title_font=dict(size=10, color=COLOR_MUTED))
    fig.update_yaxes(title_text="Premium reales", row=1, col=1, title_font=dict(size=10, color=COLOR_MUTED))
    fig.update_yaxes(title_text="% premium acumulado", row=2, col=1, title_font=dict(size=10, color=COLOR_MUTED))
    fig.update_yaxes(title_text="Lift", row=3, col=1, title_font=dict(size=10, color=COLOR_MUTED))

    # Style subplot titles
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=11, color=COLOR_MUTED))

    fig.update_layout(
        height=880,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1,
            font=dict(size=10, color=COLOR_MUTED),
            bgcolor="rgba(0,0,0,0)",
        ),
        **_PBG,
    )
    st.plotly_chart(fig, width="stretch")


# ── App principal ────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Premium Intelligence · MVP",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Cargar recursos base ─────────────────────────────────────────────────
    error_msg = None
    try:
        pipeline     = load_model()
        metadata     = load_metadata()
        demo_df      = load_demo_sample()
        holdout_df   = load_scoring_data(str(PROJECT_ROOT / "data" / "processed" / "holdout_features_selected.parquet"))
        holdout_master = load_master_table(str(PROJECT_ROOT / "data" / "processed" / "03_master_table_clean_holdout.parquet"))
    except FileNotFoundError as e:
        error_msg = f"Archivo no encontrado: {e}"
    except Exception as e:
        error_msg = f"Error al cargar recursos: {e}"

    if error_msg:
        st.error(error_msg)
        st.stop()

    selected_cols = metadata["selected_model_columns"]

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Selector de cliente")

        modo = st.radio(
            "Modo",
            ["Casos demo (4 ejemplos)", "Explorar dataset"],
            index=0,
        )

        if modo == "Explorar dataset":
            dataset_mode = st.radio(
                "Dataset",
                list(DATASET_CONFIGS.keys()),
                index=0,
                help=(
                    "Holdout: últimos 3 meses (validación final del MVP). "
                    "Backtest: agosto 2018 (calidad del ranking con lift/gain)."
                ),
            )
        else:
            dataset_mode = "Holdout (validacion final)"

        active_config  = DATASET_CONFIGS[dataset_mode]
        active_metrics = active_config["metrics"]

        if modo == "Casos demo (4 ejemplos)":
            df        = holdout_df
            master_df = holdout_master
            opciones  = {
                f"{r['customer_unique_id'][:12]}… "
                f"({'PREMIUM' if r['is_premium'] else 'REGULAR'})": r["customer_unique_id"]
                for _, r in demo_df.iterrows()
            }
            seleccion = st.selectbox("Cliente", list(opciones.keys()))
            cid = opciones[seleccion]
            row = df[df["customer_unique_id"] == cid]

        else:
            try:
                df        = load_scoring_data(str(PROJECT_ROOT / "data" / "processed" / active_config["features_file"]))
                master_df = load_master_table(str(PROJECT_ROOT / "data" / "processed" / active_config["master_file"]))
            except FileNotFoundError as e:
                st.error(f"Dataset no disponible: {e}")
                st.stop()

            missing_cols = [c for c in selected_cols if c not in df.columns]
            if missing_cols:
                st.error(f"Columnas faltantes: {missing_cols}")
                st.stop()

            selector_df = build_selector_frame(pipeline, df, selected_cols)

            filtro = st.selectbox("Segmento", ["Premium predichos", "Premium reales", "Todos"])
            orden  = st.selectbox("Orden", ["Mayor score", "Menor score", "Índice original"])
            search_text = st.text_input("Buscar prefijo ID", placeholder="Ej: 861eff47").strip().lower()

            filtered = selector_df.copy()
            if filtro == "Premium predichos":
                filtered = filtered[filtered["is_pred_premium"]]
            elif filtro == "Premium reales":
                filtered = filtered[filtered["is_premium"] == 1]
            if search_text:
                filtered = filtered[filtered["customer_unique_id"].str.lower().str.contains(search_text, na=False)]
            if orden == "Mayor score":
                filtered = filtered.sort_values("premium_probability", ascending=False)
            elif orden == "Menor score":
                filtered = filtered.sort_values("premium_probability", ascending=True)
            else:
                filtered = filtered.sort_values("row_idx", ascending=True)

            st.caption(f"Coincidencias: {len(filtered):,}")
            if filtered.empty:
                st.warning("Sin coincidencias.")
                st.stop()

            max_options = min(len(filtered), 200)
            visibles    = st.slider("Candidatos visibles", 10, max_options, min(50, max_options), 10)
            filtered    = filtered.head(visibles)

            option_ids = filtered["customer_unique_id"].tolist()
            option_map = {
                cid_opt: format_selector_option(filtered.loc[filtered["customer_unique_id"] == cid_opt].iloc[0])
                for cid_opt in option_ids
            }
            selected_cid = st.selectbox("Cliente", option_ids, format_func=lambda c: option_map[c])

            with st.expander("Modo técnico"):
                adv_filter = st.selectbox(
                    "Explorar errores/aciertos",
                    ["Sin cambiar selección", "Aciertos premium (TP)", "Falsos positivos (FP)", "Falsos negativos (FN)"],
                )
                adv_df = selector_df.copy()
                if adv_filter == "Aciertos premium (TP)":
                    adv_df = adv_df[(adv_df["is_premium"] == 1) & adv_df["is_pred_premium"]]
                elif adv_filter == "Falsos positivos (FP)":
                    adv_df = adv_df[(adv_df["is_premium"] == 0) & adv_df["is_pred_premium"]]
                elif adv_filter == "Falsos negativos (FN)":
                    adv_df = adv_df[(adv_df["is_premium"] == 1) & ~adv_df["is_pred_premium"]]

                if adv_filter != "Sin cambiar selección":
                    adv_df   = adv_df.sort_values("premium_probability", ascending=False).head(50)
                    adv_ids  = adv_df["customer_unique_id"].tolist()
                    adv_map  = {c: format_selector_option(adv_df.loc[adv_df["customer_unique_id"] == c].iloc[0]) for c in adv_ids}
                    selected_cid = st.selectbox("Cliente técnico", adv_ids, format_func=lambda c: adv_map[c])

                idx = st.number_input("Ir a índice", 0, len(df) - 1, int(filtered.iloc[0]["row_idx"]), 1)
                if st.button("Ir"):
                    selected_cid = df.iloc[int(idx)]["customer_unique_id"]

            cid = selected_cid
            row = df[df["customer_unique_id"] == cid]

        st.divider()
        st.caption(dataset_mode.upper())
        col_a, col_b = st.columns(2)
        col_a.metric("Total", f"{active_metrics['n_total']:,}")
        col_b.metric("Premium", f"{active_metrics['n_premium']:,}")
        col_c, col_d = st.columns(2)
        col_c.metric("ROC-AUC", f"{active_metrics['roc_auc']:.4f}")
        col_d.metric("Gini", f"{active_metrics['gini']:.4f}")

    # ── Guards ───────────────────────────────────────────────────────────────
    if row.empty:
        st.warning("No se encontró el cliente seleccionado.")
        st.stop()

    X = row[selected_cols]
    try:
        prob = float(pipeline.predict_proba(X)[:, 1][0])
    except Exception as e:
        st.error(f"Error al predecir: {e}")
        st.stop()

    is_pred_premium = prob >= THRESHOLD
    is_true_premium = int(row["is_premium"].values[0]) if "is_premium" in row.columns else None

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(
        f'<h1 style="font-family:var(--syne,Syne);font-size:1.5rem;font-weight:700;color:#dce6f0;margin-bottom:0.1rem;">'
        f'Premium Intelligence</h1>'
        f'<p style="font-family:\'Space Mono\',monospace;font-size:0.65rem;color:#7a8fa6;margin-bottom:1.2rem;">'
        f'LightGBM · 28 features · umbral {THRESHOLD:.0%} · {active_config["label"]}</p>',
        unsafe_allow_html=True,
    )

    tab_scoring, tab_validacion, tab_eval = st.tabs(
        ["01 · Scoring Individual", "02 · Validación Sprint 3", f"03 · {active_config['tab_title']}"]
    )

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — Scoring individual
    # ─────────────────────────────────────────────────────────────────────────
    with tab_scoring:

        # Hero score card
        render_score_hero(prob, is_pred_premium, cid, is_true_premium)

        # SHAP chart  |  Profile card
        col_shap, col_profile = st.columns([3, 2], gap="medium")

        with col_shap:
            eyebrow("01", "EXPLICABILIDAD — ¿Por qué este score?")
            st.caption(
                "Top 5 variables que empujan hacia premium (verde) y hacia regular (rojo) "
                "para este cliente específico."
            )
            try:
                contrib = compute_contributions(pipeline, X)
                render_contribution_chart(contrib, f"Contribuciones SHAP locales")
            except Exception as e:
                st.warning(f"No se pudieron calcular contribuciones: {e}")

        with col_profile:
            eyebrow("02", "PERFIL DEL CLIENTE")
            info_row = master_df[master_df["customer_unique_id"] == cid]
            render_profile_card(info_row, row)

        # Full feature table
        eyebrow("03", "VARIABLES DEL MODELO")
        with st.expander("Ver los 28 valores de entrada"):
            feature_display = X.T.rename(columns={X.index[0]: "Valor"}).astype(str)
            st.dataframe(feature_display, width="stretch")

        with st.expander("Ver tabla de contribuciones completa"):
            try:
                contrib_df = contrib.reset_index()
                contrib_df.columns = ["Variable", "Contribución"]
                st.dataframe(contrib_df, width="stretch")
            except NameError:
                st.caption("Contribuciones no disponibles.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — Validación metodológica Sprint 3
    # ─────────────────────────────────────────────────────────────────────────
    with tab_validacion:
        s3 = SPRINT3_VALIDATION_METRICS

        eyebrow("01", "REFERENCIA METODOLÓGICA")
        st.markdown(
            "**Validación temporal oficial del Sprint 3.** Train < julio 2018 · Validación ≥ julio 2018.",
            unsafe_allow_html=False,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("ROC-AUC", f"{s3['roc_auc']:.4f}")
        m2.metric("Gini",    f"{s3['gini']:.4f}")
        m3.metric("PR-AUC",  f"{s3['pr_auc']:.4f}")
        m4, m5, m6 = st.columns(3)
        m4.metric("Precisión", f"{s3['precision']:.2%}")
        m5.metric("Recall",    f"{s3['recall']:.2%}")
        m6.metric("F1-Score",  f"{s3['f1']:.4f}")

        eyebrow("02", "INTERPRETACIÓN")

        narrative(
            "¿Por qué estas métricas importan?",
            "El Sprint 3 valida metodológicamente el clasificador sobre un split temporal riguroso. "
            "Un ROC-AUC de 0.81 sobre un holdout temporal real es la evidencia principal de que "
            "el modelo generaliza — no memoriza.",
            color="green",
        )
        narrative(
            "Relación Sprint 3 → Sprint 4",
            "Sprint 4 no reemplaza ni contradice Sprint 3: lo complementa con integración operativa "
            "(scoring sobre 96K clientes), explicabilidad SHAP y cuantificación de ROI. "
            "El AUC más alto en holdout final se debe a que ese dataset tiene menos varianza temporal.",
        )

        eyebrow("03", "CONTEXTO DEL MODELO")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Escenario de entrenamiento**")
            st.write("- Split temporal · no aleatorio")
            st.write("- LightGBM seleccionado entre 5 candidatos")
            st.write("- Umbral operativo calibrado a 0.55 sobre validación")
        with c2:
            st.markdown("**Por qué importa la separación**")
            st.write("- Aporta contexto al resultado observado en holdout")
            st.write("- Separa validez metodológica de demostración operativa")
            st.write("- Permite juzgar generalización temporal real")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — Evaluación operativa (Holdout / Backtest)
    # ─────────────────────────────────────────────────────────────────────────
    with tab_eval:
        m            = active_metrics
        cm           = get_confusion_counts(m)
        is_backtest  = active_config["has_lift"]
        section_label = "backtest (agosto 2018)" if is_backtest else "holdout_3m"

        # ── 01 — Desempeño del modelo ────────────────────────────────────────
        eyebrow("01", "DESEMPEÑO DEL MODELO")
        st.caption(
            f"Métricas precalculadas sobre {section_label} · umbral {THRESHOLD:.0%}"
        )

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("ROC-AUC",   f"{m['roc_auc']:.4f}")
        mc2.metric("Gini",      f"{m['gini']:.4f}")
        mc3.metric("Precisión", f"{m['precision']:.2%}")
        mc4.metric("Recall",    f"{m['recall']:.2%}")
        mc5.metric("F1-Score",  f"{m['f1']:.2%}")

        narrative(
            f"Interpretación sobre {section_label}",
            f"Con umbral 0.55, el modelo clasifica {m['n_pred_premium']:,} clientes como premium. "
            f"De esos, {m['tp_count']:,} son premium reales (precisión {m['precision']:.0%}). "
            f"Se pierden {m['n_premium'] - m['tp_count']:,} premium reales no detectados (FN).",
        )

        # ── 02 — Análisis de clasificación ──────────────────────────────────
        eyebrow("02", "ANÁLISIS DE CLASIFICACIÓN")

        count_cols = st.columns(4)
        count_cols[0].metric("TP — Detectados correctos", f"{cm['tp']:,}",   delta="Premium reales avisados")
        count_cols[1].metric("FP — Falsos positivos",     f"{cm['fp']:,}",   delta="Costo adicional", delta_color="inverse")
        count_cols[2].metric("FN — Perdidos",             f"{cm['fn']:,}",   delta="Premium no detectados", delta_color="inverse")
        count_cols[3].metric("TN — Descartados OK",       f"{cm['tn']:,}")

        st.divider()
        col_cm, col_cm_note = st.columns([3, 2], gap="medium")
        with col_cm:
            render_confusion_matrix(cm, section_label)
        with col_cm_note:
            st.markdown("")
            st.markdown("")
            narrative(
                "Cómo leer la matriz",
                "Filas: etiqueta real · Columnas: predicción del modelo. "
                "La celda superior-izquierda (TN) domina porque el 98.7% de los "
                "clientes son no-premium. Los FP tienen costo de campaña; los FN "
                "son la oportunidad perdida.",
                color="green",
            )
            st.markdown("")
            fc1, fc2 = st.columns(2)
            fc1.metric(
                "% premium capturado",
                f"{cm['tp'] / m['n_premium']:.1%}",
                delta=f"{cm['tp']:,} de {m['n_premium']:,}",
            )
            fc2.metric(
                "Precisión efectiva",
                f"{m['precision']:.1%}",
                delta=f"{m['n_pred_premium']:,} clasificados premium",
            )

        # ── 03 — Impacto de negocio ──────────────────────────────────────────
        eyebrow("03", "IMPACTO DE NEGOCIO")

        st.markdown(
            '<div class="roi-compare">'
            f'<div class="roi-block bad">'
            f'  <span class="roi-block-lbl">Campaña masiva · {m["n_total"]:,} clientes</span>'
            f'  <span class="roi-block-val bad">{m["roi_masiva"]:.2f}%</span>'
            f'  <span class="roi-block-sub">ROI &nbsp;·&nbsp; contactar a todos</span>'
            f'</div>'
            f'<div class="roi-block good">'
            f'  <span class="roi-block-lbl">Campaña optimizada · {m["n_pred_premium"]:,} clientes</span>'
            f'  <span class="roi-block-val good">+{m["roi_modelo"]:.2f}%</span>'
            f'  <span class="roi-block-sub">ROI &nbsp;·&nbsp; sólo predichos premium</span>'
            f'</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        col_roi, col_roi_stats = st.columns([3, 2], gap="medium")
        with col_roi:
            render_roi_bars(m)
            st.caption("Supuesto: costo BRL 15/cliente · retorno BRL 120/premium convertido.")
        with col_roi_stats:
            st.markdown("")
            st.markdown("")
            r1, r2 = st.columns(2)
            r1.metric("Gasto total",       f"BRL {m['total_spend']:,.0f}")
            r2.metric("Gasto en premium",  f"BRL {m['premium_spend']:,.0f}", delta=f"{m['premium_spend_pct']:.1%} del total")
            r3, r4 = st.columns(2)
            r3.metric("TP spend capturado", f"BRL {m['tp_spend']:,.0f}", delta=f"{m['tp_spend']/m['premium_spend']:.1%} del gasto premium")
            r4.metric("Ahorro estimado",    f"BRL {m['cost_savings']:,.0f}", delta=f"reducción {1 - m['n_pred_premium']/m['n_total']:.1%}")

        # ── 04 — Lift y ranking (backtest only) ─────────────────────────────
        if is_backtest:
            eyebrow("04", "CALIDAD DEL RANKING")

            lift_table = build_lift_table(pipeline, df, selected_cols)
            top10 = lift_table.iloc[0]
            base_rate = lift_table.attrs.get("base_rate", 0.0)
            n_active = lift_table.attrs.get("n_rows", 0)
            total_events = lift_table.attrs.get("total_events", 0)

            lg1, lg2, lg3 = st.columns(3)
            lg1.metric("Top 10% activos captura", f"{top10['cum_events_pct']:.1%}", delta="de los premium reales activos")
            lg2.metric("Tasa premium decil 1",    f"{top10['premium_rate_pct']:.1f}%", delta=f"vs base {base_rate:.1%}")
            lg3.metric("Lift acumulado decil 1",  f"{top10['lift']:.2f}x", delta="vs selección aleatoria")

            narrative(
                "Lectura correcta sobre clientes activos",
                f"En agosto analizamos {n_active:,} clientes activos, con {total_events:,} premium reales. "
                f"El primer decil alcanza una tasa premium de ~{top10['premium_rate_pct']:.0f}% "
                f"y captura {top10['cum_events_pct']:.0%} de los premium reales seleccionando solo el 10% de la base activa.",
                color="green",
            )

            gl_table, gl_charts = st.columns([2, 3], gap="medium")
            with gl_table:
                eyebrow("", "TABLA DE DECILES")
                display_table = lift_table[
                    ["decile", "cases", "responses", "premium_rate_pct",
                     "cum_responses", "gain_pct", "lift", "cum_lift"]
                ].copy()
                display_table.columns = ["Decil", "Casos", "Premium", "Tasa %", "Cum.Premium", "Gain %", "Lift", "Lift Cum."]
                for col in ["Tasa %", "Gain %"]:
                    display_table[col] = display_table[col].map(lambda x: f"{x:.2f}%")
                for col in ["Lift", "Lift Cum."]:
                    display_table[col] = display_table[col].map(lambda x: f"{x:.2f}")
                st.dataframe(display_table, width="stretch", hide_index=True)

            with gl_charts:
                eyebrow("", "CURVAS INTERACTIVAS")
                render_lift_charts(lift_table)


if __name__ == "__main__":
    main()
