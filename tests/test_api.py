"""Testes da API Churn (FastAPI) — /health e /predict."""
from fastapi.testclient import TestClient

from churn_prediction.api.main import app

client = TestClient(app)

# Payload completo e válido (espelha o exemplo do schemas.py — dataset Telco)
VALID_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85,
}


def test_health_ok():
    """/health deve responder 200 (usado pelo healthcheck do container)."""
    resp = client.get("/health")
    assert resp.status_code == 200


def test_predict_retorna_previsao():
    """/predict com payload válido deve retornar probabilidade e classe."""
    resp = client.post("/predict", json=VALID_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert "churn_probability" in body
    assert "churn_prediction" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in ("Yes", "No")


def test_predict_payload_invalido():
    """Payload malformado deve ser rejeitado pelo Pydantic (422)."""
    resp = client.post("/predict", json={"campo_invalido": 1})
    assert resp.status_code == 422