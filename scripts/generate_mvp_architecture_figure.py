from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "sprint_04_mvp_architecture.png"


def add_box(ax, x, y, w, h, face, edge, title, lines, tag=None, tag_color=None):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.8,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)

    if tag and tag_color:
        tag_box = FancyBboxPatch(
            (x + 0.015, y + h - 0.06),
            0.11,
            0.04,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=0,
            facecolor=tag_color,
        )
        ax.add_patch(tag_box)
        ax.text(
            x + 0.07,
            y + h - 0.04,
            tag,
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )

    ax.text(x + 0.02, y + h - 0.11, title, fontsize=15, fontweight="bold", color="#213b57")

    start_y = y + h - 0.17
    for idx, line in enumerate(lines):
        ax.text(x + 0.02, start_y - idx * 0.045, line, fontsize=11.5, color="#324e68")


def add_arrow(ax, start, end, label=None):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.2,
        color="#5e7488",
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2 + 0.03
        ax.text(mx, my, label, fontsize=10.5, color="#4f667d", ha="center", fontweight="bold")


fig = plt.figure(figsize=(16, 9), dpi=160)
ax = plt.axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("#f5f7fb")
ax.set_facecolor("#f5f7fb")

ax.text(0.04, 0.93, "Sprint 4 - Arquitectura Final del MVP", fontsize=23, fontweight="bold", color="#1f3b5b")
ax.text(
    0.04,
    0.895,
    "Pipeline batch + modelo compartido + dashboard Streamlit + API FastAPI + Docker Compose",
    fontsize=12.5,
    color="#57718c",
)

add_box(
    ax,
    0.04,
    0.18,
    0.24,
    0.62,
    "#edf6f0",
    "#8db19c",
    "Pipeline holdout",
    [
        "1. Split temporal holdout_3m",
        "2. build_features.sh",
        "3. build_holdout_scoring_dataset.py",
        "",
        "Salidas:",
        "holdout_features_selected.parquet",
        "demo_sample_scoring.parquet",
    ],
    tag="DATOS",
    tag_color="#4f8c64",
)

add_box(
    ax,
    0.34,
    0.18,
    0.18,
    0.62,
    "#fcf3ea",
    "#d6a06e",
    "Modelo final compartido",
    [
        "models/final/modelo_final.pkl",
        "",
        "Incluye:",
        "preprocessor",
        "LightGBM classifier",
        "threshold y esquema comun",
        "",
        "Runtime validado:",
        "Python 3.12",
    ],
    tag="ARTEFACTO ML",
    tag_color="#d18335",
)

add_box(
    ax,
    0.58,
    0.18,
    0.18,
    0.62,
    "#ffffff",
    "#bfd0e4",
    "Dashboard",
    [
        "Container separado",
        "Streamlit - puerto 8501",
        "",
        "Funciones:",
        "score por cliente",
        "SHAP local",
        "KPIs y ROI",
        "demo para jurado",
    ],
)

add_box(
    ax,
    0.78,
    0.18,
    0.18,
    0.62,
    "#ffffff",
    "#bfd0e4",
    "API",
    [
        "Container separado",
        "FastAPI - puerto 8000",
        "",
        "Endpoints:",
        "GET /health",
        "POST /predict",
        "",
        "Mismo modelo y features",
    ],
)

compose = FancyBboxPatch(
    (0.56, 0.08),
    0.36,
    0.07,
    boxstyle="round,pad=0.012,rounding_size=0.02",
    linewidth=1.5,
    edgecolor="#b4c8e0",
    facecolor="#e4edf9",
)
ax.add_patch(compose)
ax.text(0.74, 0.115, "Docker Compose orquesta dashboard + api", ha="center", va="center", fontsize=12, color="#304e68")

footer = FancyBboxPatch(
    (0.04, 0.03),
    0.92,
    0.045,
    boxstyle="round,pad=0.012,rounding_size=0.016",
    linewidth=1.0,
    edgecolor="#d7e0eb",
    facecolor="#ffffff",
)
ax.add_patch(footer)
ax.text(
    0.05,
    0.052,
    "Mensaje clave: Sprint 3 valida el modelo; Sprint 4 demuestra que ese modelo puede operarse y explicarse como MVP.",
    fontsize=11.5,
    color="#314d68",
    va="center",
)

add_arrow(ax, (0.28, 0.49), (0.34, 0.49), "features listas")
add_arrow(ax, (0.52, 0.57), (0.58, 0.57), "mismo .pkl")
add_arrow(ax, (0.52, 0.40), (0.58, 0.40), "scoring")
add_arrow(ax, (0.76, 0.49), (0.78, 0.49))

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
