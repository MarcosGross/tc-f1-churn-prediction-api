"""Testes unitários da lógica de inferência de churn."""

import numpy as np

from churn_prediction.inference import predict_churn


class FixedProbabilityModel:
    """Modelo falso que sempre devolve a probabilidade informada no teste."""

    def __init__(self, churn_probability: float):
        self.churn_probability = churn_probability

    def predict_proba(self, input_df):
        return np.array([[1 - self.churn_probability, self.churn_probability]])


def test_predict_churn_returns_yes_above_default_threshold():
    model = FixedProbabilityModel(churn_probability=0.80)

    probability, prediction = predict_churn(model, {"tenure": 1})

    assert probability == 0.80
    assert prediction == "Yes"


def test_predict_churn_returns_no_below_default_threshold():
    model = FixedProbabilityModel(churn_probability=0.20)

    probability, prediction = predict_churn(model, {"tenure": 24})

    assert probability == 0.20
    assert prediction == "No"


def test_predict_churn_returns_yes_at_default_threshold():
    model = FixedProbabilityModel(churn_probability=0.50)

    probability, prediction = predict_churn(model, {"tenure": 12})

    assert probability == 0.50
    assert prediction == "Yes"
