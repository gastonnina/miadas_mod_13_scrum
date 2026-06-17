from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Rutas ──────────────────────────────────────────────────────────────────
def _find_project_root() -> Path:
    # Sube directorios hasta encontrar la raiz del proyecto (tiene data/ y models/)
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
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "holdout_features_selected.parquet"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "06_features_selected_metadata.json"
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "03_master_table_clean_holdout.parquet"
DEMO_SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "demo_sample_scoring.parquet"

THRESHOLD = 0.55

# Metricas pre-computadas en notebook 04_demo_validation.ipynb (holdout_3m, umbral 0.55)
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


# ── Cache de recursos ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_scoring_data() -> pd.DataFrame:
    return pd.read_parquet(DATA_PATH)


@st.cache_data
def load_metadata() -> dict:
    with open(METADATA_PATH) as f:
        return json.load(f)


@st.cache_data
def load_master_table() -> pd.DataFrame:
    return pd.read_parquet(MASTER_PATH)


@st.cache_data
def load_demo_sample() -> pd.DataFrame:
    return pd.read_parquet(DEMO_SAMPLE_PATH)


# ── Helpers ─────────────────────────────────────────────────────────────────
def compute_contributions(pipeline, X_row: pd.DataFrame) -> pd.Series:
    """Contribuciones locales via pred_contrib nativo de LightGBM."""
    preprocessor = pipeline.named_steps["preprocessor"]
    lgb_model = pipeline.named_steps["model"]
    X_proc = preprocessor.transform(X_row)
    feature_names = preprocessor.get_feature_names_out()
    X_proc_df = pd.DataFrame(X_proc, columns=feature_names, index=X_row.index)
    contribs = lgb_model.predict(X_proc_df, pred_contrib=True)
    clean_names = [
        n.replace("num__", "").replace("cat__", "") for n in feature_names
    ]
    return pd.Series(contribs[0, :-1], index=clean_names).sort_values(ascending=False)


def render_contribution_chart(contrib: pd.Series, title: str) -> None:
    top5 = contrib.head(5)
    bot5 = contrib.tail(5)
    combined = pd.concat([top5, bot5.iloc[::-1]])

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in combined]
    ax.barh(combined.index, combined.values, color=colors, edgecolor="white")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Contribucion al score (log-odds)")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def safe_int(value, default: str = "N/A"):
    """Devuelve entero si el valor es valido; si no, retorna un texto por defecto."""
    if value is None or pd.isna(value):
        return default
    return int(value)


def safe_float_metric(value, default: str = "N/A") -> str:
    """Formatea metricas numericas evitando NaN en el dashboard."""
    if value is None or pd.isna(value):
        return default
    return f"{float(value):,.2f}"


# ── App principal ────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="MVP Clientes Premium",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Identificacion de Clientes Premium")
    st.caption(
        f"Modelo: `modelo_final.pkl` · Pipeline: LightGBM + ColumnTransformer · "
        f"Umbral: {THRESHOLD:.0%} · Dataset: holdout\\_3m (96 096 clientes)"
    )

    # Cargar recursos
    error_msg = None
    try:
        pipeline = load_model()
        df = load_scoring_data()
        metadata = load_metadata()
        master_df = load_master_table()
        demo_df = load_demo_sample()
    except FileNotFoundError as e:
        error_msg = f"Archivo no encontrado: {e}"
    except Exception as e:
        error_msg = f"Error al cargar recursos: {e}"

    if error_msg:
        st.error(error_msg)
        st.stop()

    selected_cols = metadata["selected_model_columns"]
    missing_cols = [c for c in selected_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Columnas faltantes en el dataset: {missing_cols}")
        st.stop()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Selector de cliente")

        modo = st.radio(
            "Modo de seleccion",
            ["Casos demo (4 ejemplos)", "Holdout completo (96k)"],
            index=0,
        )

        if modo == "Casos demo (4 ejemplos)":
            opciones = {
                f"{row['customer_unique_id'][:12]}... "
                f"({'PREMIUM' if row['is_premium'] else 'REGULAR'})": row["customer_unique_id"]
                for _, row in demo_df.iterrows()
            }
            seleccion = st.selectbox("Cliente", list(opciones.keys()))
            cid = opciones[seleccion]
            row = df[df["customer_unique_id"] == cid]
        else:
            idx = st.number_input(
                "Indice de registro (0 a 96 095)",
                min_value=0,
                max_value=len(df) - 1,
                value=0,
                step=1,
            )
            row = df.iloc[[int(idx)]]
            cid = row["customer_unique_id"].values[0]

        st.divider()
        st.markdown("**Estadisticas del holdout**")
        m = HOLDOUT_METRICS
        st.metric("Total clientes", f"{m['n_total']:,}")
        st.metric("Clientes premium reales", f"{m['n_premium']:,} ({m['premium_rate']:.1%})")
        st.metric("ROC-AUC holdout", f"{m['roc_auc']:.4f}")
        st.metric("Gini holdout", f"{m['gini']:.4f}")

    if row.empty:
        st.warning("No se encontro el cliente seleccionado.")
        st.stop()

    X = row[selected_cols]

    # ── Prediccion ────────────────────────────────────────────────────────────
    try:
        prob = float(pipeline.predict_proba(X)[:, 1][0])
    except Exception as e:
        st.error(f"Error al predecir: {e}")
        st.stop()

    is_pred_premium = prob >= THRESHOLD
    is_true_premium = int(row["is_premium"].values[0]) if "is_premium" in row.columns else None

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_scoring, tab_metricas = st.tabs(
        ["Scoring Individual", "Metricas del Modelo"]
    )

    # ─── Tab 1: Scoring individual ─────────────────────────────────────────
    with tab_scoring:
        st.subheader("Resultado de la clasificacion")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Clasificacion",
            "PREMIUM" if is_pred_premium else "NO PREMIUM",
        )
        col2.metric("Probabilidad de premium", f"{prob:.1%}")
        if is_true_premium is not None:
            real_label = "PREMIUM" if is_true_premium else "NO PREMIUM"
            correcto = is_pred_premium == bool(is_true_premium)
            col3.metric(
                "Etiqueta real",
                real_label,
                delta="Correcto" if correcto else "Incorrecto",
            )
        col4.metric("Umbral de decision", f"{THRESHOLD:.0%}")

        st.progress(prob, text=f"Score de premium: {prob:.1%}")
        st.divider()

        # Perfil del cliente desde master table
        info_row = master_df[master_df["customer_unique_id"] == cid]
        if not info_row.empty:
            st.subheader("Perfil del cliente")
            info = info_row.iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Gasto total (BRL)", safe_float_metric(info.get("total_spent", 0)))
            c2.metric("Ordenes totales", safe_int(info.get("total_orders", 0), default="0"))
            c3.metric("Ticket promedio (BRL)", safe_float_metric(info.get("avg_ticket", 0)))
            c4.metric("Estado", str(info.get("customer_state", "N/A")))
            recency = row["recency_days"].values[0] if "recency_days" in row.columns else None
            c5.metric(
                "Recencia (dias)",
                safe_int(recency),
            )
            st.divider()

        # Contribuciones locales (SHAP nativo LightGBM)
        st.subheader("Variables mas relevantes para este cliente")
        st.caption("SHAP local del cliente seleccionado. Verde empuja hacia premium; rojo empuja hacia regular.")
        try:
            contrib = compute_contributions(pipeline, X)
            render_contribution_chart(
                contrib,
                f"Contribuciones al score — {cid[:20]}...",
            )
            with st.expander("Ver tabla de contribuciones completa"):
                contrib_df = contrib.reset_index()
                contrib_df.columns = ["Variable", "Contribucion"]
                st.dataframe(contrib_df, width="stretch")
        except Exception as e:
            st.warning(f"No se pudieron calcular contribuciones locales: {e}")

        # Valores de las 28 features — cast a str para evitar error Arrow con tipos mixtos
        with st.expander("Valores de las 28 variables del cliente"):
            feature_display = X.T.rename(columns={X.index[0]: "Valor"}).astype(str)
            st.dataframe(feature_display, width="stretch")

    # ─── Tab 2: Metricas del modelo ────────────────────────────────────────
    with tab_metricas:
        m = HOLDOUT_METRICS

        st.subheader("Metricas tecnicas (holdout_3m, umbral 0.55)")
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
        mc2.metric("Gini", f"{m['gini']:.4f}")
        mc3.metric("Precision", f"{m['precision']:.2%}")
        mc4.metric("Recall", f"{m['recall']:.2%}")
        mc5.metric("F1-Score", f"{m['f1']:.2%}")

        st.divider()
        st.subheader("Distribucion de facturacion capturada")
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Facturacion total holdout", f"BRL {m['total_spend']:,.2f}")
        fc2.metric(
            "Facturacion segmento premium real",
            f"BRL {m['premium_spend']:,.2f}",
            delta=f"{m['premium_spend_pct']:.1%} del total",
        )
        fc3.metric(
            "Gasto premium detectado (TP)",
            f"BRL {m['tp_spend']:,.2f}",
            delta=f"{m['tp_spend'] / m['premium_spend']:.1%} del gasto premium",
        )

        st.divider()
        st.subheader("Simulacion de campana de marketing")
        st.caption(
            "Supuesto: costo BRL 15 por cliente contactado | retorno BRL 120 por cliente premium convertido"
        )

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric(
            "ROI Campana Masiva (all)",
            f"{m['roi_masiva']:.2f}%",
            delta=f"{m['n_total']:,} clientes contactados",
            delta_color="inverse",
        )
        cc2.metric(
            "ROI Campana Optimizada (modelo)",
            f"{m['roi_modelo']:.2f}%",
            delta=f"{m['n_pred_premium']:,} clientes contactados",
        )
        cc3.metric(
            "Ahorro en costo de marketing",
            f"BRL {m['cost_savings']:,.2f}",
            delta=f"Reduccion del {1 - m['n_pred_premium'] / m['n_total']:.1%}",
        )

        # Grafico comparativo de ROI
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        labels = ["Campana Masiva\n(Target All)", "Campana Optimizada\n(Modelo, umbral 0.55)"]
        rois = [m["roi_masiva"], m["roi_modelo"]]
        roi_colors = ["#e74c3c", "#27ae60"]
        bars = ax2.bar(labels, rois, color=roi_colors, edgecolor="white", width=0.4)
        ax2.axhline(0, color="#333333", linewidth=0.8)
        ax2.set_ylabel("ROI (%)")
        ax2.set_title("Comparativa de Retorno de Inversion por Estrategia de Campana")
        for bar, val in zip(bars, rois):
            ypos = val + 5 if val >= 0 else val - 12
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                f"{val:.2f}%",
                ha="center",
                fontweight="bold",
            )
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)


if __name__ == "__main__":
    main()
