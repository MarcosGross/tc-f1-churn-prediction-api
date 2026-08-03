"""Carregamento do artefato do modelo campeão de churn."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "champion_model.joblib"


@lru_cache(maxsize=1)
def load_model(model_path: Path = MODEL_PATH):
    """Carrega e mantém em memória o modelo usado nas previsões."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {model_path}. "
            "Rode `python -m churn_prediction.train` antes de subir a API."
        )

    logger.info("Carregando modelo de %s", model_path)
    return joblib.load(model_path)
