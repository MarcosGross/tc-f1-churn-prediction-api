"""Gera os artefatos da Etapa 2 para o gate de qualidade do CI:
  - models/champion_model.joblib  (modelo campeão)
  - models/baseline_metrics.json  (métricas do campeão — inclui roc_auc)
  - data/test.parquet             (hold-out X_test + coluna Churn)

Reusa o MESMO split/seed do train.py (determinístico). Não altera o train.py.
Uso:  python scripts/gerar_baseline.py
"""
from __future__ import annotations

import json
import logging

import joblib
from sklearn.model_selection import train_test_split

from churn_prediction.preprocessing import clean_raw_dataframe, split_features_target
from churn_prediction.train import (
    CHAMPION_MODEL_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    RANDOM_STATE,
    load_data,
    select_champion,
    train_and_compare,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEST_DATA_PATH = PROJECT_ROOT / "data" / "test.parquet"
BASELINE_METRICS_PATH = MODELS_DIR / "baseline_metrics.json"
TARGET_COLUMN = "Churn"


def main() -> None:
    df = load_data()

    # Treina/compara os 3 candidatos (mesmo split/seed) e escolhe o campeão
    comparison_df, fitted_pipelines = train_and_compare(df)
    champion_name = select_champion(comparison_df)
    champion = fitted_pipelines[champion_name]

    # Persiste o modelo campeão (idempotente com o train.py)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(champion, CHAMPION_MODEL_PATH)

    # Recria o MESMO hold-out (determinístico) para exportar
    df_clean = clean_raw_dataframe(df)
    X, y = split_features_target(df_clean)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # data/test.parquet = features de teste + alvo (0/1)
    TEST_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.to_numpy()
    test_df.to_parquet(TEST_DATA_PATH, index=False)

    # models/baseline_metrics.json = linha completa do campeão (roc_auc, f1, etc.)
    baseline_metrics = {
        "champion": champion_name,
        **{k: float(v) for k, v in comparison_df.loc[champion_name].to_dict().items()},
    }
    BASELINE_METRICS_PATH.write_text(json.dumps(baseline_metrics, indent=2))

    logger.info("Campeão: %s", champion_name)
    logger.info("Métricas baseline: %s", baseline_metrics)
    logger.info(
        "Artefatos salvos:\n  %s\n  %s\n  %s",
        CHAMPION_MODEL_PATH,
        TEST_DATA_PATH,
        BASELINE_METRICS_PATH,
    )


if __name__ == "__main__":
    main()
