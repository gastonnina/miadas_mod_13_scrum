from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Rutas ───────────────────────────────────────────────────────────────────
def _find_project_root() -> Path:
    candidate = Path(__file__).resolve()
    for parent in [candidate, *candidate.parents]:
        if (parent / "models").exists() and (parent / "data").exists():
            return parent
    raise FileNotFoundError(f"No se encontro la raiz del proyecto desde {candidate}.")


PROJECT_ROOT = _find_project_root()
MODEL_PATH = PROJECT_ROOT / "models" / "final" / "modelo_final.pkl"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "06_features_selected_metadata.json"

THRESHOLD = 0.55
MODEL_VERSION = "modelo_final.pkl"
PREDICT_REQUEST_EXAMPLE = {
    "total_orders": 2.0,
    "total_items": 7.0,
    "total_products": 3.0,
    "avg_review_score": 5.0,
    "avg_delivery_days": 6.0,
    "avg_estimated_delivery_days": 32.5,
    "delivered_orders": 2.0,
    "late_deliveries": 0.0,
    "payment_methods_count": 1.0,
    "max_payment_installments": 10.0,
    "recency_days": 65.0,
    "customer_lifetime_days": 0.0,
    "cancellation_rate": 0.0,
    "products_per_order": 1.5,
    "max_to_avg_price_ratio": 3.4008097166,
    "installments_gt_1_flag": 1.0,
    "installments_gt_6_flag": 1.0,
    "credit_card_flag": 1.0,
    "voucher_flag": 0.0,
    "delivery_gap": 26.5,
    "reviews_per_order": 1.0,
    "far_region_flag": 0.0,
    "top_category_is_high_value": 1.0,
    "customer_state": "SC",
    "main_payment_type": "credit_card",
    "top_category": "construcao_ferramentas_construcao",
    "region_group": "south",
    "top_category_group": "other",
}
PREDICT_RESPONSE_PREMIUM_EXAMPLE = {
    "customer_classification": "PREMIUM",
    "is_premium": True,
    "premium_probability": 0.992563,
    "threshold_used": 0.55,
    "model_version": "modelo_final.pkl",
}
PREDICT_RESPONSE_REGULAR_EXAMPLE = {
    "customer_classification": "REGULAR",
    "is_premium": False,
    "premium_probability": 0.083142,
    "threshold_used": 0.55,
    "model_version": "modelo_final.pkl",
}
HEALTH_RESPONSE_EXAMPLE = {
    "status": "ok",
    "model_loaded": True,
    "model_version": "modelo_final.pkl",
    "features_expected": 28,
    "threshold": 0.55,
}

# ── Carga del modelo al arrancar ─────────────────────────────────────────────
_pipeline = None
_selected_cols: list[str] = []


def _load_resources() -> None:
    global _pipeline, _selected_cols
    with open(MODEL_PATH, "rb") as f:
        _pipeline = pickle.load(f)
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    _selected_cols = meta["selected_model_columns"]


# ── Schemas ──────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """
    Las 28 features seleccionadas que espera modelo_final.pkl.
    Las categoricas aceptan None para que el preprocessor aplique su imputacion.
    """

    # Numericas
    total_orders: float = Field(..., example=2.0)
    total_items: float = Field(..., example=3.0)
    total_products: float = Field(..., example=2.0)
    avg_review_score: float = Field(..., example=4.5)
    avg_delivery_days: float = Field(..., example=12.0)
    avg_estimated_delivery_days: float = Field(..., example=18.0)
    delivered_orders: float = Field(..., example=2.0)
    late_deliveries: float = Field(..., example=0.0)
    payment_methods_count: float = Field(..., example=1.0)
    max_payment_installments: float = Field(..., example=6.0)
    recency_days: float = Field(..., example=65.0)
    customer_lifetime_days: float = Field(..., example=0.0)
    cancellation_rate: float = Field(..., example=0.0)
    products_per_order: float = Field(..., example=1.5)
    max_to_avg_price_ratio: float = Field(..., example=1.2)
    installments_gt_1_flag: float = Field(..., example=1.0)
    installments_gt_6_flag: float = Field(..., example=0.0)
    credit_card_flag: float = Field(..., example=1.0)
    voucher_flag: float = Field(..., example=0.0)
    delivery_gap: float = Field(..., example=5.0)
    reviews_per_order: float = Field(..., example=1.0)
    far_region_flag: float = Field(..., example=0.0)
    top_category_is_high_value: float = Field(..., example=1.0)

    # Categoricas
    customer_state: Optional[str] = Field(None, example="SP")
    main_payment_type: Optional[str] = Field(None, example="credit_card")
    top_category: Optional[str] = Field(None, example="informatica_acessorios")
    region_group: Optional[str] = Field(None, example="southeast")
    top_category_group: Optional[str] = Field(None, example="tech")

    class Config:
        json_schema_extra = {
            "example": PREDICT_REQUEST_EXAMPLE,
        }


class PredictResponse(BaseModel):
    customer_classification: str
    is_premium: bool
    premium_probability: float
    threshold_used: float
    model_version: str

    class Config:
        json_schema_extra = {
            "examples": [
                PREDICT_RESPONSE_PREMIUM_EXAMPLE,
                PREDICT_RESPONSE_REGULAR_EXAMPLE,
            ]
        }


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    features_expected: int
    threshold: float

    class Config:
        json_schema_extra = {
            "example": HEALTH_RESPONSE_EXAMPLE,
        }


# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Premium Customers API",
    description=(
        "API minima para scoring de clientes premium sobre el dataset Olist. "
        "Reutiliza modelo_final.pkl del Sprint 3."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def startup_event() -> None:
    _load_resources()


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["ops"],
    summary="Estado del servicio",
)
def health() -> HealthResponse:
    """Verificacion de disponibilidad del servicio y del modelo."""
    return HealthResponse(
        status="ok" if _pipeline is not None else "degraded",
        model_loaded=_pipeline is not None,
        model_version=MODEL_VERSION,
        features_expected=len(_selected_cols),
        threshold=THRESHOLD,
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["scoring"],
    summary="Scoring de cliente premium",
    responses={
        200: {
            "description": "Prediccion exitosa",
            "content": {
                "application/json": {
                    "examples": {
                        "premium_detectado": {
                            "summary": "Cliente clasificado como PREMIUM",
                            "value": PREDICT_RESPONSE_PREMIUM_EXAMPLE,
                        },
                        "cliente_regular": {
                            "summary": "Cliente clasificado como REGULAR",
                            "value": PREDICT_RESPONSE_REGULAR_EXAMPLE,
                        },
                    }
                }
            },
        }
    },
)
def predict(body: PredictRequest) -> PredictResponse:
    """
    Clasifica un cliente como PREMIUM o REGULAR.

    Request: las 28 features seleccionadas por el modelo (ver schema).
    Response: clasificacion, probabilidad y umbral utilizado.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")

    row = pd.DataFrame([body.model_dump()])[_selected_cols]

    missing = [c for c in _selected_cols if c not in row.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Columnas faltantes en el request: {missing}",
        )

    try:
        prob = float(_pipeline.predict_proba(row)[:, 1][0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error de inferencia: {exc}") from exc

    is_premium = prob >= THRESHOLD
    return PredictResponse(
        customer_classification="PREMIUM" if is_premium else "REGULAR",
        is_premium=is_premium,
        premium_probability=round(prob, 6),
        threshold_used=THRESHOLD,
        model_version=MODEL_VERSION,
    )
