"""API de inferência de churn.

Endpoints:
    GET  /health   -> verifica se a API está no ar.
    POST /predict  -> recebe os dados de um cliente e retorna a propensão de churn.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from churn_prediction.api.schemas import ChurnPrediction, CustomerFeatures, HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="API de inferência para o modelo campeão de previsão de churn.",
    version="0.1.0",
)

# Aponta para o modelo campeao, escolhido em train.py (Etapa 2) por comparacao
# entre Regressao Logistica, Random Forest e MLPClassifier via ROC-AUC.
MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "champion_model.joblib"

_model = None


def get_model():
    """Carrega o modelo sob demanda (lazy load), uma única vez por processo."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Modelo não encontrado em {MODEL_PATH}. "
                    "Rode `python -m churn_prediction.train` antes de subir a API."
                ),
            )
        logger.info("Carregando modelo de %s", MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerFeatures) -> ChurnPrediction:
    model = get_model()

    input_df = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = "Yes" if probability >= 0.5 else "No"

    return ChurnPrediction(churn_probability=probability, churn_prediction=prediction)
