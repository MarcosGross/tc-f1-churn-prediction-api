"""Funções de limpeza e pipeline de pré-processamento para o dataset de churn.

Este módulo existe para que a MESMA lógica usada na EDA (notebooks/01_eda.ipynb)
seja reaproveitada no treino (train.py) e na API (api/main.py) — sem duplicar
código entre experimentação e produção.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica a limpeza básica identificada na EDA.

    Decisão documentada na EDA: `TotalCharges` chega como string e tem 11
    registros vazios, todos com tenure=0 (clientes recém-chegados que ainda não
    fecharam um ciclo de cobrança). Tratamos isso como 0, não como dado ausente
    real — não descartamos as linhas.
    """
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        n_missing = df["TotalCharges"].isna().sum()
        if n_missing:
            logger.info(
                "Imputando %d valores ausentes em TotalCharges com 0 (clientes tenure=0)",
                n_missing,
            )
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) do alvo (y), descartando o identificador do cliente."""
    df = df.copy()

    y = None
    if TARGET_COLUMN in df.columns:
        y = (df[TARGET_COLUMN] == "Yes").astype(int)
        df = df.drop(columns=[TARGET_COLUMN])

    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    return df, y


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Cria o ColumnTransformer usado por todos os modelos (baseline, RF, MLP).

    - Numéricas: padronizadas (StandardScaler) — importante especialmente para
      Regressão Logística e MLPClassifier, que são sensíveis à escala.
    - Categóricas: one-hot encoded, ignorando categorias não vistas em produção
      (`handle_unknown="ignore"`) para a API não quebrar com um valor novo.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_full_pipeline(estimator) -> Pipeline:
    """Combina pré-processamento + estimador em um único Pipeline sklearn.

    Isso garante que o mesmo objeto salvo (.joblib) faz a limpeza, o encoding
    E a predição — a API não precisa reimplementar nada disso manualmente.
    """
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("model", estimator),
        ]
    )
