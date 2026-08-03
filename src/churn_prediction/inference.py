"""Lógica de inferência para o modelo campeão de churn."""

from __future__ import annotations

import pandas as pd

DEFAULT_CHURN_THRESHOLD = 0.5


def predict_churn(
    model,
    customer_data: dict[str, object],
    threshold: float = DEFAULT_CHURN_THRESHOLD,
) -> tuple[float, str]:
    """Calcula a probabilidade de churn e converte o resultado em Yes ou No."""
    input_df = pd.DataFrame([customer_data])
    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = "Yes" if probability >= threshold else "No"

    return probability, prediction
