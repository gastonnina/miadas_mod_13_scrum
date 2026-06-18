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
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "backtest_features_selected.parquet"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "06_features_selected_metadata.json"
MASTER_PATH = PROJECT_ROOT / "data" / "processed" / "03_master_table_clean_backtest.parquet"
DEMO_SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "demo_sample_scoring.parquet"

THRESHOLD = 0.55
COLOR_PREMIUM = "#39a96b"
COLOR_RISK = "#d95d39"
COLOR_NEUTRAL = "#5b7c99"
COLOR_TEXT = "#213547"
CMAP_CONFUSION = "GnBu"

# Metricas oficiales del backtest (agosto 2018, umbral 0.55)
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
    colors = [COLOR_PREMIUM if v >= 0 else COLOR_RISK for v in combined]
    ax.barh(combined.index, combined.values, color=colors, edgecolor="white")
    ax.axvline(0, color=COLOR_NEUTRAL, linewidth=0.8)
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


@st.cache_data(show_spinner="Calculando scores del backtest completo...")
def build_backtest_selector_frame(_pipeline, df: pd.DataFrame, selected_cols: list[str]) -> pd.DataFrame:
    """Precalcula score y etiquetas para un selector mas util del backtest."""
    enriched = df[["customer_unique_id", "is_premium"]].copy()
    enriched["row_idx"] = df.index.astype(int)
    enriched["premium_probability"] = _pipeline.predict_proba(df[selected_cols])[:, 1]
    enriched["is_pred_premium"] = enriched["premium_probability"] >= THRESHOLD
    enriched["selection_bucket"] = "Todos"
    enriched.loc[enriched["is_premium"] == 1, "selection_bucket"] = "Premium reales"
    enriched.loc[enriched["is_pred_premium"], "selection_bucket"] = "Premium predichos"
    return enriched


def format_backtest_option(option_row: pd.Series) -> str:
    real_label = "PREM" if int(option_row["is_premium"]) == 1 else "REG"
    pred_label = "PREM" if bool(option_row["is_pred_premium"]) else "REG"
    return (
        f"{int(option_row['row_idx']):>5} | "
        f"{str(option_row['customer_unique_id'])[:12]}... | "
        f"prob={option_row['premium_probability']:.1%} | "
        f"real={real_label} | pred={pred_label}"
    )


def get_confusion_counts() -> dict[str, int]:
    tp = int(BACKTEST_METRICS["tp_count"])
    fp = int(BACKTEST_METRICS["n_pred_premium"] - tp)
    fn = int(BACKTEST_METRICS["n_premium"] - tp)
    tn = int(BACKTEST_METRICS["n_total"] - tp - fp - fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


@st.cache_data(show_spinner="Calculando tabla lift/gain del backtest...")
def build_backtest_lift_table(_pipeline, df: pd.DataFrame, selected_cols: list[str]) -> pd.DataFrame:
    scored = pd.DataFrame(
        {
            "is_premium": df["is_premium"].astype(int),
            "prob": _pipeline.predict_proba(df[selected_cols])[:, 1],
        }
    ).sort_values("prob", ascending=False, kind="mergesort").reset_index(drop=True)

    n_rows = len(scored)
    scored["decile"] = pd.Series(range(1, n_rows + 1)).div(n_rows).mul(10).apply(lambda x: int(-(-x // 1)))
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
    lift_table["cum_events_pct_display"] = lift_table["cum_events_pct"] * 100
    lift_table["gain_pct"] = lift_table["cum_events_pct"] * 100
    lift_table["lift"] = lift_table["response_rate"] / base_rate
    lift_table["cum_lift"] = lift_table["cum_events_pct"] / lift_table["cum_cases_pct"]
    return lift_table


# ── App principal ────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="MVP Clientes Premium",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        f"""
        <style>
        .stMetric {{
            border: 1px solid rgba(91, 124, 153, 0.18);
            border-radius: 14px;
            padding: 0.35rem 0.6rem 0.5rem 0.6rem;
            background: rgba(255,255,255,0.02);
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-bottom: 2px solid transparent;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Identificacion de Clientes Premium")
    st.caption(
        f"Modelo: `modelo_final.pkl` · Pipeline: LightGBM + ColumnTransformer · "
        f"Umbral: {THRESHOLD:.0%} · Dataset: backtest (agosto 2018, 96 096 clientes)"
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

    backtest_selector_df = build_backtest_selector_frame(pipeline, df, selected_cols)
    lift_table = build_backtest_lift_table(pipeline, df, selected_cols)

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
            filtro = st.selectbox(
                "Segmento a explorar",
                [
                    "Premium predichos",
                    "Premium reales",
                    "Todos",
                ],
            )
            orden = st.selectbox(
                "Orden",
                ["Mayor score", "Menor score", "Indice original"],
            )
            search_text = st.text_input(
                "Buscar por prefijo de customer_unique_id",
                value="",
                placeholder="Ej: 861eff47",
            ).strip().lower()

            filtered = backtest_selector_df.copy()
            if filtro == "Premium predichos":
                filtered = filtered[filtered["is_pred_premium"]]
            elif filtro == "Premium reales":
                filtered = filtered[filtered["is_premium"] == 1]
            if search_text:
                filtered = filtered[
                    filtered["customer_unique_id"].str.lower().str.contains(search_text, na=False)
                ]

            if orden == "Mayor score":
                filtered = filtered.sort_values("premium_probability", ascending=False)
            elif orden == "Menor score":
                filtered = filtered.sort_values("premium_probability", ascending=True)
            else:
                filtered = filtered.sort_values("row_idx", ascending=True)

            st.caption(f"Coincidencias: {len(filtered):,}")

            if filtered.empty:
                st.warning("No hay clientes que coincidan con ese filtro.")
                st.stop()

            max_options = min(len(filtered), 200)
            visibles = st.slider(
                "Cantidad de candidatos visibles",
                min_value=10,
                max_value=max_options,
                value=min(50, max_options),
                step=10,
            )
            filtered = filtered.head(visibles)

            option_ids = filtered["customer_unique_id"].tolist()
            option_map = {
                cid_option: format_backtest_option(filtered.loc[filtered["customer_unique_id"] == cid_option].iloc[0])
                for cid_option in option_ids
            }
            selected_cid = st.selectbox(
                "Cliente del backtest",
                option_ids,
                format_func=lambda cid_option: option_map[cid_option],
            )

            with st.expander("Modo tecnico y analisis avanzado"):
                advanced_filter = st.selectbox(
                    "Explorar errores/aciertos del modelo",
                    [
                        "Sin cambiar seleccion actual",
                        "Aciertos premium (TP)",
                        "Falsos positivos (FP)",
                        "Falsos negativos (FN)",
                    ],
                )
                advanced_df = backtest_selector_df.copy()
                if advanced_filter == "Aciertos premium (TP)":
                    advanced_df = advanced_df[
                        (advanced_df["is_premium"] == 1) & (advanced_df["is_pred_premium"])
                    ]
                elif advanced_filter == "Falsos positivos (FP)":
                    advanced_df = advanced_df[
                        (advanced_df["is_premium"] == 0) & (advanced_df["is_pred_premium"])
                    ]
                elif advanced_filter == "Falsos negativos (FN)":
                    advanced_df = advanced_df[
                        (advanced_df["is_premium"] == 1) & (~advanced_df["is_pred_premium"])
                    ]

                if advanced_filter != "Sin cambiar seleccion actual":
                    st.caption(f"Coincidencias tecnicas: {len(advanced_df):,}")
                    advanced_df = advanced_df.sort_values("premium_probability", ascending=False).head(50)
                    advanced_ids = advanced_df["customer_unique_id"].tolist()
                    advanced_map = {
                        cid_option: format_backtest_option(
                            advanced_df.loc[advanced_df["customer_unique_id"] == cid_option].iloc[0]
                        )
                        for cid_option in advanced_ids
                    }
                    selected_cid = st.selectbox(
                        "Cliente tecnico",
                        advanced_ids,
                        format_func=lambda cid_option: advanced_map[cid_option],
                    )

                idx = st.number_input(
                    "Indice de registro (0 a 96 095)",
                    min_value=0,
                    max_value=len(df) - 1,
                    value=int(filtered.iloc[0]["row_idx"]),
                    step=1,
                )
                if st.button("Ir a indice"):
                    selected_cid = df.iloc[int(idx)]["customer_unique_id"]

            cid = selected_cid
            row = df[df["customer_unique_id"] == cid]

        st.divider()
        st.markdown("**Estadisticas del backtest**")
        m = BACKTEST_METRICS
        st.metric("Total clientes", f"{m['n_total']:,}")
        st.metric("Clientes premium reales", f"{m['n_premium']:,} ({m['premium_rate']:.1%})")
        st.metric("ROC-AUC backtest", f"{m['roc_auc']:.4f}")
        st.metric("Gini backtest", f"{m['gini']:.4f}")

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
    tab_scoring, tab_validacion, tab_backtest = st.tabs(
        ["Scoring Individual", "Validacion Sprint 3", "Backtest y ROI"]
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

    # ─── Tab 2: Validacion metodologica Sprint 3 ──────────────────────────
    with tab_validacion:
        s3 = SPRINT3_VALIDATION_METRICS

        st.subheader("Referencia metodologica del modelo final")
        st.caption(
            "Estas metricas corresponden a la validacion temporal oficial del Sprint 3. "
            "Son la referencia principal para juzgar generalizacion del modelo."
        )
        v1, v2, v3, v4, v5, v6 = st.columns(6)
        v1.metric("ROC-AUC val", f"{s3['roc_auc']:.4f}")
        v2.metric("Gini val", f"{s3['gini']:.4f}")
        v3.metric("PR-AUC val", f"{s3['pr_auc']:.4f}")
        v4.metric("Precision val", f"{s3['precision']:.2%}")
        v5.metric("Recall val", f"{s3['recall']:.2%}")
        v6.metric("F1 val", f"{s3['f1']:.4f}")

        st.info(
            "Interpretacion recomendada: Sprint 3 valida metodologicamente el clasificador "
            "sobre un split temporal train/validacion. Sprint 4 no reemplaza esa referencia; "
            "la complementa con integracion del MVP, scoring sobre backtest, explicabilidad y ROI."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Escenario Sprint 3**")
            st.write("- Split temporal de desarrollo y validacion.")
            st.write("- Base para comparar modelos y elegir LightGBM final.")
            st.write("- Umbral operativo adoptado: 0.55.")
        with c2:
            st.markdown("**Por que importa**")
            st.write("- Evita leer el backtest como mejora milagrosa.")
            st.write("- Separa validez metodologica de demostracion operativa.")
            st.write("- Da contexto a la metrica alta del backtest.")

    # ─── Tab 3: Backtest operativo Sprint 4 ───────────────────────────────
    with tab_backtest:
        m = BACKTEST_METRICS
        cm = get_confusion_counts()

        st.subheader("Desempeno operativo sobre backtest (agosto 2018)")
        st.caption(
            "Estas metricas estan precalculadas sobre el backtest oficial de Sprint 4. "
            "Sirven para evaluar la operacion del MVP, no como reemplazo directo de la validacion de Sprint 3."
        )
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("ROC-AUC", f"{m['roc_auc']:.4f}")
        mc2.metric("Gini", f"{m['gini']:.4f}")
        mc3.metric("Precision", f"{m['precision']:.2%}")
        mc4.metric("Recall", f"{m['recall']:.2%}")
        mc5.metric("F1-Score", f"{m['f1']:.2%}")

        st.caption("Con umbral 0.55, el modelo prioriza un equilibrio entre precision y recall para campanas focalizadas.")

        st.divider()
        st.subheader("Conteos de clasificacion")
        cc_tp, cc_fp, cc_fn, cc_tn = st.columns(4)
        cc_tp.metric("TP", f"{cm['tp']:,}", delta="Premium reales detectados")
        cc_fp.metric("FP", f"{cm['fp']:,}", delta="Contactos sin premium real", delta_color="inverse")
        cc_fn.metric("FN", f"{cm['fn']:,}", delta="Premium reales omitidos", delta_color="inverse")
        cc_tn.metric("TN", f"{cm['tn']:,}", delta="No premium correctamente descartados")

        st.divider()
        st.subheader("Distribucion de facturacion capturada")
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Facturacion total backtest", f"BRL {m['total_spend']:,.2f}")
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
        st.subheader("Lift y gains del ranking")
        top10 = lift_table.iloc[0]
        top20 = lift_table.iloc[1]
        lg1, lg2, lg3 = st.columns(3)
        lg1.metric("Top 10% captura", f"{top10['cum_events_pct']:.1%}", delta="de premium reales")
        lg2.metric("Tasa premium en decil 1", f"{top10['premium_rate_pct']:.2f}%", delta="vs base 1.30%")
        lg3.metric("Lift primer decil", f"{top10['lift']:.2f}x", delta="vs seleccion aleatoria")

        st.caption(
            "Ordenamos el backtest por score descendente, lo partimos en 10 grupos iguales y medimos "
            "cuantos premium reales captura cada tramo del ranking."
        )
        st.info(
            "Lectura correcta: el primer decil no es 100% premium. Su tasa premium es 13.04%, "
            "pero contiene el 100% de los premium reales del backtest. Por eso el gain acumulado "
            "salta a 100% desde el primer 10% de la base."
        )

        gl_col1, gl_col2 = st.columns(2)
        with gl_col1:
            st.markdown("**Tabla de deciles**")
            display_table = lift_table[
                ["decile", "cases", "responses", "premium_rate_pct", "cum_responses", "gain_pct", "lift", "cum_lift"]
            ].copy()
            display_table["premium_rate_pct"] = display_table["premium_rate_pct"].map(lambda x: f"{x:.2f}%")
            display_table["gain_pct"] = display_table["gain_pct"].map(lambda x: f"{x:.2f}%")
            display_table["lift"] = display_table["lift"].map(lambda x: f"{x:.2f}")
            display_table["cum_lift"] = display_table["cum_lift"].map(lambda x: f"{x:.2f}")
            st.dataframe(display_table, width="stretch", hide_index=True)

        with gl_col2:
            st.markdown("**Responses por decil y curvas gain / lift**")
            fig_gl, axes_gl = plt.subplots(3, 1, figsize=(6.2, 9.8), sharex=False)
            axes_gl[0].bar(
                lift_table["decile"].astype(str),
                lift_table["responses"],
                color=COLOR_NEUTRAL,
                edgecolor="white",
            )
            axes_gl[0].set_ylabel("Premium reales")
            axes_gl[0].set_title("Responses por decil")
            axes_gl[0].grid(axis="y", alpha=0.25, linestyle=":")

            x = lift_table["cum_cases_pct"] * 100
            gains = lift_table["cum_events_pct"] * 100
            cum_lift = lift_table["cum_lift"]

            axes_gl[1].plot(x, gains, marker="o", linewidth=2.2, color="#1565C0", label="Modelo")
            axes_gl[1].plot([0, 100], [0, 100], linestyle="--", color=COLOR_RISK, alpha=0.7, label="Aleatorio")
            axes_gl[1].set_ylabel("% premium acumulado")
            axes_gl[1].set_title("Cumulative gains")
            axes_gl[1].legend()
            axes_gl[1].grid(alpha=0.25, linestyle=":")

            axes_gl[2].plot(x, cum_lift, marker="o", linewidth=2.2, color=COLOR_PREMIUM, label="Lift acumulado")
            axes_gl[2].plot([0, 100], [1, 1], linestyle="--", color=COLOR_RISK, alpha=0.7, label="Aleatorio")
            axes_gl[2].set_xlabel("% de la base priorizada")
            axes_gl[2].set_ylabel("Lift")
            axes_gl[2].set_title("Cumulative lift")
            axes_gl[2].legend()
            axes_gl[2].grid(alpha=0.25, linestyle=":")
            plt.tight_layout()
            st.pyplot(fig_gl)
            plt.close(fig_gl)

        st.divider()
        st.subheader("Matriz de confusion y ROI")
        st.caption(
            "La matriz resume aciertos/errores del clasificador; el ROI muestra el valor de usar el score para focalizar la campana."
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

        col_cm, col_roi = st.columns(2)

        with col_cm:
            st.markdown("**Matriz de confusion**")
            st.caption("Filas: etiqueta real. Columnas: prediccion del modelo.")
            fig_cm, ax_cm = plt.subplots(figsize=(5.0, 4.0))
            cm_matrix = [
                [cm["tn"], cm["fp"]],
                [cm["fn"], cm["tp"]],
            ]
            im = ax_cm.imshow(cm_matrix, cmap=CMAP_CONFUSION)
            ax_cm.set_xticks([0, 1], labels=["Pred: NO PREMIUM", "Pred: PREMIUM"])
            ax_cm.set_yticks([0, 1], labels=["Real: NO PREMIUM", "Real: PREMIUM"])
            ax_cm.set_title("Backtest")
            max_cell = max(cm_matrix[0] + cm_matrix[1])
            for i in range(2):
                for j in range(2):
                    ax_cm.text(
                        j,
                        i,
                        f"{cm_matrix[i][j]:,}",
                        ha="center",
                        va="center",
                        color=COLOR_TEXT if cm_matrix[i][j] < max_cell * 0.65 else "white",
                        fontweight="bold",
                    )
            fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

        with col_roi:
            st.markdown("**Comparativa de ROI**")
            st.caption("Supuesto: costo BRL 15 por cliente | retorno BRL 120 por premium convertido.")
            fig2, ax2 = plt.subplots(figsize=(6.0, 4.0))
            labels = ["Campana Masiva\n(Target All)", "Campana Optimizada\n(Modelo, umbral 0.55)"]
            rois = [m["roi_masiva"], m["roi_modelo"]]
            roi_colors = [COLOR_RISK, COLOR_PREMIUM]
            bars = ax2.bar(labels, rois, color=roi_colors, edgecolor="white", width=0.45)
            ax2.axhline(0, color=COLOR_NEUTRAL, linewidth=0.8)
            ax2.set_ylabel("ROI (%)")
            ax2.set_title("Retorno por estrategia")
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
