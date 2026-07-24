"""Teste da API — verifica que /health responde corretamente.

Nota: o teste de /predict fica marcado para rodar apenas se o modelo já foi
treinado (models/*.joblib existe), já que a API depende de um artefato gerado
por `python -m churn_prediction.train`.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from churn_prediction.api.main import MODEL_PATH, app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason="Modelo ainda não treinado — rode `python -m churn_prediction.train` primeiro.",
)
def test_predict_returns_valid_probability():
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }

    response = client.post("/predict", json=sample_customer)

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in {"Yes", "No"}
