"""Teste unitário da função de limpeza de dados (pré-processamento)."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_prediction.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_full_pipeline,
    build_preprocessing_pipeline,
    clean_raw_dataframe,
    split_features_target,
)


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
    assert df["TotalCharges"].iloc[0] == " "


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


def test_split_features_target_without_target_returns_none():
    df = pd.DataFrame(
        {
            "customerID": ["0001-TEST"],
            "tenure": [12],
        }
    )

    X, y = split_features_target(df)

    assert list(X.columns) == ["tenure"]
    assert y is None
    assert list(df.columns) == ["customerID", "tenure"]


def test_build_preprocessing_pipeline_uses_expected_transformers():
    preprocessing = build_preprocessing_pipeline()
    transformers = {
        name: (transformer, columns) for name, transformer, columns in preprocessing.transformers
    }

    numeric_transformer, numeric_columns = transformers["num"]
    categorical_transformer, categorical_columns = transformers["cat"]

    assert isinstance(preprocessing, ColumnTransformer)
    assert isinstance(numeric_transformer, StandardScaler)
    assert numeric_columns == NUMERIC_FEATURES
    assert isinstance(categorical_transformer, OneHotEncoder)
    assert categorical_transformer.handle_unknown == "ignore"
    assert categorical_transformer.drop == "if_binary"
    assert categorical_columns == CATEGORICAL_FEATURES


def test_build_full_pipeline_combines_preprocessing_and_estimator():
    estimator = DummyClassifier(strategy="most_frequent")

    pipeline = build_full_pipeline(estimator)

    assert list(pipeline.named_steps) == ["preprocessing", "model"]
    assert isinstance(pipeline.named_steps["preprocessing"], ColumnTransformer)
    assert pipeline.named_steps["model"] is estimator
