"""Treina e compara os modelos de churn. Escolhe o campeao e salva em models/.

Uso:
    python -m churn_prediction.train              # treino + selecao do campeao
    python -m churn_prediction.train --scenarios  # + comparacao de cenarios

Candidatos (Etapa 2): Regressao Logistica (baseline), Random Forest,
MLPClassifier e KNN (candidato adicional, alem do exigido pelo enunciado).

Tratamento de desbalanceamento (73% No / 27% Yes) por modelo:
- LogisticRegression: class_weight="balanced" (nativo).
- RandomForestClassifier: class_weight="balanced" (nativo).
- MLPClassifier e KNN: NAO tem parametro class_weight no sklearn -- compensamos
  com oversampling manual da classe minoritaria, aplicado SOMENTE no treino.

Nota: testamos tambem SMOTE e SMOTENC (via imbalanced-learn) para comparar as
estrategias sob o mesmo protocolo. Nenhuma superou a abordagem atual; optamos
por esta (mais simples, sem dependencia externa). Ver Model Card, secao 4.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from churn_prediction.preprocessing import (
    build_preprocessing_pipeline,
    build_raw_pipeline,
    build_scaled_pipeline,
    clean_raw_dataframe,
    split_features_target,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODELS_DIR = PROJECT_ROOT / "models"
CHAMPION_MODEL_PATH = MODELS_DIR / "champion_model.joblib"
COMPARISON_TABLE_PATH = MODELS_DIR / "model_comparison.csv"
SCENARIOS_TABLE_PATH = MODELS_DIR / "preprocessing_scenarios.csv"

CHAMPION_METRIC = "roc_auc"

# Margem minima de melhoria exigida de um challenger sobre o baseline (LR)
# para evitar trocar de campeao por uma diferenca dentro do ruido. A validacao
# cruzada (ver evaluation.py) confirmou que o desvio entre folds e' ~0.014.
MIN_IMPROVEMENT = 0.01

# Modelos sem tratamento nativo de desbalanceamento -- recebem oversampling
# manual no treino.
MODELS_REQUIRING_OVERSAMPLING = {"mlp_classifier", "knn"}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    logger.info("Carregando dataset de %s", path)
    return pd.read_csv(path)


def get_candidate_models() -> dict:
    """Candidatos avaliados: linear, ensemble, MLP e KNN.

    Os tres primeiros sao exigidos pelo Tech Challenge; o KNN foi adicionado
    como candidato extra para ampliar a comparacao.
    """
    return {
        "logistic_regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "mlp_classifier": MLPClassifier(
            random_state=RANDOM_STATE,
            hidden_layer_sizes=(32, 16),
            max_iter=500,
            early_stopping=True,
        ),
        "knn": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }


def build_full_pipeline(estimator) -> Pipeline:
    """Combina pre-processamento + estimador em um Pipeline sklearn puro."""
    return Pipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("model", estimator),
        ]
    )


def oversample_minority(X_train: pd.DataFrame, y_train: pd.Series):
    """Duplica as linhas da classe minoritaria (Churn=1) SOMENTE no treino.

    Nunca aplicar isso no conjunto de teste -- inflaria a metrica de avaliacao
    de forma artificial, escondendo o desempenho real do modelo em producao.
    """
    minority_mask = y_train == 1
    X_minority = X_train[minority_mask]
    y_minority = y_train[minority_mask]

    X_balanced = pd.concat([X_train, X_minority, X_minority], ignore_index=True)
    y_balanced = pd.concat([y_train, y_minority, y_minority], ignore_index=True)

    logger.info(
        "Oversampling aplicado: treino foi de %d para %d linhas (classe 1: %d -> %d)",
        len(X_train), len(X_balanced), minority_mask.sum(), (y_balanced == 1).sum(),
    )
    return X_balanced, y_balanced


def evaluate_pipeline(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "recall_churn": recall_score(y_test, y_pred),
    }


def _split(df: pd.DataFrame):
    """Split estratificado 80/20 com seed fixa -- unico ponto de divisao."""
    df = clean_raw_dataframe(df)
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )


def train_and_compare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Treina os candidatos com o MESMO split/seed e retorna a comparacao.

    O oversampling (quando aplicavel) acontece DEPOIS do split, usando apenas
    os dados de treino -- o conjunto de teste nunca e' tocado.
    """
    X_train, X_test, y_train, y_test = _split(df)

    fitted_pipelines = {}
    rows = []

    for name, estimator in get_candidate_models().items():
        logger.info("Treinando %s...", name)

        if name in MODELS_REQUIRING_OVERSAMPLING:
            X_fit, y_fit = oversample_minority(X_train, y_train)
        else:
            X_fit, y_fit = X_train, y_train

        pipeline = build_full_pipeline(estimator)
        pipeline.fit(X_fit, y_fit)

        metrics = evaluate_pipeline(pipeline, X_test, y_test)
        rows.append({"model": name, **metrics})
        fitted_pipelines[name] = pipeline

        logger.info("%s -> %s", name, metrics)

    comparison_df = pd.DataFrame(rows).set_index("model").sort_values(
        CHAMPION_METRIC, ascending=False
    )

    champion_name = select_champion(comparison_df)
    y_pred_champion = fitted_pipelines[champion_name].predict(X_test)
    logger.info(
        "Classification report do campeao (%s):\n%s",
        champion_name,
        classification_report(y_test, y_pred_champion),
    )

    return comparison_df, fitted_pipelines


def compare_preprocessing_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Analise exploratoria: mede o impacto de escalonamento e balanceamento.

    Cenarios avaliados (mesmo split/seed em todos):
    - Raw:             one-hot, numericas sem escala, sem balanceamento
    - Scaled:          + StandardScaler nas numericas
    - Scaled+Balanced: + oversampling da classe minoritaria no treino

    IMPORTANTE: esta funcao NAO seleciona o campeao nem altera artefatos de
    producao. Ela existe para documentar empiricamente por que o pipeline de
    producao usa escalonamento + balanceamento. Os modelos aqui rodam SEM
    class_weight, para que o balanceamento seja de fato a variavel testada.
    """
    X_train, X_test, y_train, y_test = _split(df)

    models = {
        "Logistic Reg": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE, n_estimators=300, n_jobs=-1
        ),
        "MLP": MLPClassifier(
            random_state=RANDOM_STATE, hidden_layer_sizes=(32, 16),
            max_iter=500, early_stopping=True,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }

    scenarios = {
        "Raw": (build_raw_pipeline, False),
        "Scaled": (build_scaled_pipeline, False),
        "Scaled+Balanced": (build_scaled_pipeline, True),
    }

    rows = []
    for scenario_name, (builder, apply_balancing) in scenarios.items():
        if apply_balancing:
            X_fit, y_fit = oversample_minority(X_train, y_train)
        else:
            X_fit, y_fit = X_train, y_train

        for model_name, model in models.items():
            logger.info("Cenario %s | %s", scenario_name, model_name)

            pipeline = builder(model)
            pipeline.fit(X_fit, y_fit)

            metrics = evaluate_pipeline(pipeline, X_test, y_test)
            rows.append({"scenario": scenario_name, "model": model_name, **metrics})

    return pd.DataFrame(rows).sort_values(
        ["scenario", CHAMPION_METRIC], ascending=[True, False]
    )


def select_champion(comparison_df: pd.DataFrame) -> str:
    """Escolhe o campeao pela metrica tecnica primaria (ROC-AUC, ver ML Canvas).

    Regra do Tech Challenge: Regressao Logistica e' o baseline; os demais
    candidatos PRECISAM supera-la em ROC-AUC por uma margem minima de
    MIN_IMPROVEMENT, para evitar trocar de campeao por diferenca de ruido.
    """
    baseline_score = comparison_df.loc["logistic_regression", CHAMPION_METRIC]
    challengers = comparison_df.drop(index="logistic_regression")
    best_challenger = challengers[CHAMPION_METRIC].idxmax()

    if challengers.loc[best_challenger, CHAMPION_METRIC] > baseline_score + MIN_IMPROVEMENT:
        return best_challenger

    logger.warning(
        "Nenhum challenger superou o baseline em %s por uma margem >= %.4f --"
        " mantendo Regressao Logistica como campeao.",
        CHAMPION_METRIC,
        MIN_IMPROVEMENT,
    )
    return "logistic_regression"


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino e comparacao de modelos de churn.")
    parser.add_argument(
        "--scenarios",
        action="store_true",
        help="Roda tambem a comparacao exploratoria de cenarios de pre-processamento.",
    )
    args = parser.parse_args()

    df = load_data()
    comparison_df, fitted_pipelines = train_and_compare(df)

    logger.info("Tabela comparativa de modelos:\n%s", comparison_df.round(4))

    champion_name = select_champion(comparison_df)
    logger.info("Modelo campeao escolhido: %s", champion_name)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_pipelines[champion_name], CHAMPION_MODEL_PATH)
    comparison_df.round(4).to_csv(COMPARISON_TABLE_PATH)

    logger.info("Modelo campeao salvo em %s", CHAMPION_MODEL_PATH)
    logger.info("Tabela comparativa salva em %s", COMPARISON_TABLE_PATH)

    if args.scenarios:
        logger.info("=== COMPARACAO DE CENARIOS DE PRE-PROCESSAMENTO ===")
        scenarios_df = compare_preprocessing_scenarios(df)
        logger.info("\n%s", scenarios_df.round(4).to_string(index=False))
        scenarios_df.round(4).to_csv(SCENARIOS_TABLE_PATH, index=False)
        logger.info("Tabela de cenarios salva em %s", SCENARIOS_TABLE_PATH)


if __name__ == "__main__":
    main()