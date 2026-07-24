"""Treina o baseline de churn (Regressão Logística) e salva o modelo.

Uso:
    python -m churn_prediction.train

Este script cobre o entregável da Etapa 1: "notebook de EDA + baseline". O
notebook (notebooks/01_eda.ipynb) explora e documenta as decisões; este módulo
é a versão produtiva, reprodutível e testável dessas mesmas decisões.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from churn_prediction.preprocessing import (
    build_full_pipeline,
    clean_raw_dataframe,
    split_features_target,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "Telco-Customer-Churn.csv"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "baseline_logistic_regression.joblib"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    logger.info("Carregando dataset de %s", path)
    return pd.read_csv(path)


def train_baseline(df: pd.DataFrame) -> dict:
    """Treina a Regressão Logística baseline e retorna as métricas de avaliação."""
    df = clean_raw_dataframe(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,  # mantém a proporção ~73/27 de Churn em treino e teste
    )

    pipeline = build_full_pipeline(
        LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced")
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "recall_churn": recall_score(y_test, y_pred),
    }

    logger.info("Métricas do baseline (Regressão Logística): %s", metrics)
    logger.info("Relatório de classificação:\n%s", classification_report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info("Modelo salvo em %s", MODEL_PATH)

    return metrics


def main() -> None:
    df = load_data()
    train_baseline(df)


if __name__ == "__main__":
    main()
