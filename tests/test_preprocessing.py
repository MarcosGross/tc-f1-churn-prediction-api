"""Teste unitário da função de limpeza de dados (pré-processamento)."""

import pandas as pd

from churn_prediction.preprocessing import clean_raw_dataframe, split_features_target


def test_clean_raw_dataframe_imputes_empty_total_charges():
    """TotalCharges vazio (string) deve virar 0.0, não NaN nem erro de tipo."""
    df = pd.DataFrame(
        {
            "customerID": ["0001-TEST"],
            "tenure": [0],
            "TotalCharges": [" "],  # caso real observado no dataset (tenure=0)
            "MonthlyCharges": [50.0],
            "Churn": ["No"],
        }
    )

    cleaned = clean_raw_dataframe(df)

    assert cleaned["TotalCharges"].dtype.kind == "f"
    assert cleaned["TotalCharges"].iloc[0] == 0.0


def test_split_features_target_encodes_churn_as_binary():
    """A coluna Churn (Yes/No) deve virar 1/0, e customerID deve ser removido de X."""
    df = pd.DataFrame(
        {
            "customerID": ["0001-TEST", "0002-TEST"],
            "tenure": [1, 24],
            "Churn": ["Yes", "No"],
        }
    )

    X, y = split_features_target(df)

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert list(y) == [1, 0]
