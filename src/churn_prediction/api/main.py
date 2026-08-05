"""API de inferência de churn.

Endpoints:
    GET  /health   -> verifica se a API está no ar.
    POST /predict  -> recebe os dados de um cliente e retorna a propensão de churn.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from churn_prediction.api.schemas import ChurnPrediction, CustomerFeatures, HealthResponse
from churn_prediction.inference import predict_churn
from churn_prediction.model_loader import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(
    title="Churn Prediction API",
    description="API de inferência para o modelo campeão de previsão de churn.",
    version="0.1.0",
)


def get_model():
    """Traduz a ausência do modelo para uma resposta HTTP da API."""
    try:
        return load_model()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerFeatures) -> ChurnPrediction:
    model = get_model()

    probability, prediction = predict_churn(model, customer.model_dump())

    return ChurnPrediction(churn_probability=probability, churn_prediction=prediction)
