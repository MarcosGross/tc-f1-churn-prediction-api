"""Testes de caracterização dos endpoints da API de churn."""

import numpy as np
from fastapi.testclient import TestClient

from churn_prediction.api import main as api_main

client = TestClient(api_main.app)


class FakeChurnModel:
    """Modelo controlado para testar a API sem depender do arquivo .joblib."""

    def predict_proba(self, input_df):
        return np.array([[0.20, 0.80]])


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_characterized_prediction(monkeypatch):
    monkeypatch.setattr(api_main, "_model", FakeChurnModel())

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
    assert response.json() == {
        "churn_probability": 0.80,
        "churn_prediction": "Yes",
    }
