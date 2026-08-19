"""GATE DE QUALIDADE DO MODELO (Aula 07 FIAP) — alinhado ao ML Canvas (ROC-AUC).

Só promove a imagem se o campeão cumprir o contrato:
  1) ROC-AUC >= piso absoluto (ROC_AUC_MINIMO)
  2) não regredir mais que DELTA_ROC_AUC_MAX vs. o baseline salvo (Etapa 2)
"""
import json
import os

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

ROC_AUC_MINIMO = 0.80          # piso absoluto (Telco LogReg fica ~0.84)
DELTA_ROC_AUC_MAX = 0.02       # regressão máxima tolerada vs. baseline

MODEL_PATH = os.getenv("MODEL_PATH", "models/champion_model.joblib")
TEST_DATA = os.getenv("TEST_DATA", "data/test.parquet")
BASELINE_METRICS = os.getenv("BASELINE_METRICS", "models/baseline_metrics.json")
TARGET_COL = os.getenv("TARGET_COL", "Churn")


def _baseline_roc_auc() -> float:
    """Lê o ROC-AUC do baseline (Etapa 2). Se não existir, usa o piso mínimo."""
    try:
        with open(BASELINE_METRICS) as f:
            return float(json.load(f)["roc_auc"])
    except (FileNotFoundError, KeyError):
        return ROC_AUC_MINIMO


def test_modelo_cumpre_contrato_de_qualidade():
    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(TEST_DATA)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    proba = model.predict_proba(X)[:, 1]
    roc_atual = roc_auc_score(y, proba)
    roc_baseline = _baseline_roc_auc()

    # 1) Piso absoluto de qualidade
    assert roc_atual >= ROC_AUC_MINIMO, (
        f"ROC-AUC={roc_atual:.4f} abaixo do mínimo ({ROC_AUC_MINIMO}). Deploy bloqueado."
    )

    # 2) Não regredir mais que o tolerado vs. baseline
    delta = roc_baseline - roc_atual
    assert delta <= DELTA_ROC_AUC_MAX, (
        f"Regressão de ROC-AUC = {delta:.4f} (> {DELTA_ROC_AUC_MAX}). "
        f"Atual={roc_atual:.4f} vs. baseline={roc_baseline:.4f}. Deploy bloqueado."
    )

    print(f"✅ Contrato OK | ROC-AUC={roc_atual:.4f} | baseline={roc_baseline:.4f} | Δ={delta:.4f}")
