"""Testes de caracterização dos endpoints da API de churn."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from churn_prediction.api import main as api_main

client = TestClient(api_main.app)


class FakeChurnModel:
    """Modelo controlado para testar a API sem depender do arquivo .joblib."""

    def predict_proba(self, input_df):
        return np.array([[0.20, 0.80]])


@pytest.fixture
def sample_customer():
    """Fornece um cliente válido reutilizável nos testes de /predict."""
    return {
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


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_characterized_prediction(monkeypatch, sample_customer):
    monkeypatch.setattr(api_main, "load_model", lambda: FakeChurnModel())

    response = client.post("/predict", json=sample_customer)

    assert response.status_code == 200
    assert response.json() == {
        "churn_probability": 0.80,
        "churn_prediction": "Yes",
    }


def test_predict_returns_service_unavailable_when_model_is_missing(monkeypatch, sample_customer):
    error_message = "Modelo não encontrado para o teste."

    def raise_model_not_found():
        raise FileNotFoundError(error_message)

    monkeypatch.setattr(api_main, "load_model", raise_model_not_found)

    response = client.post("/predict", json=sample_customer)

    assert response.status_code == 503
    assert response.json() == {"detail": error_message}


def test_predict_rejects_invalid_customer_data(sample_customer):
    invalid_customer = {**sample_customer, "SeniorCitizen": 2}

    response = client.post("/predict", json=invalid_customer)

    assert response.status_code == 422
    assert any(error["loc"] == ["body", "SeniorCitizen"] for error in response.json()["detail"])
