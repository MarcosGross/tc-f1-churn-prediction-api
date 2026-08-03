"""Treina e compara os modelos de churn: Regressao Logistica, Random Forest e
MLPClassifier. Escolhe o campeao e salva em models/.

Uso:
    python -m churn_prediction.train

Etapa 1: baseline (Regressao Logistica) isolado -> train_baseline().
Etapa 2: comparacao dos 3 modelos com o MESMO split/seed -> train_and_compare().

Tratamento de desbalanceamento: os 3 candidatos usam a MESMA estrategia --
SMOTE aplicado dentro do pipeline (so' no .fit(), nunca no .predict()) --
para que a comparacao entre modelos seja sob o mesmo protocolo. Nao usamos
class_weight="balanced" em nenhum deles, pois MLPClassifier nao suporta esse
parametro; misturar estrategias (class_weight em uns, SMOTE em outro)
tornaria a comparacao entre candidatos injusta.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
from sklearn.neural_network import MLPClassifier

from churn_prediction.preprocessing import (
    build_preprocessing_pipeline,
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

# Metrica usada para escolher o campeao. E' a metrica tecnica primaria definida
# no ML Canvas (bloco Offline Evaluation) -- AUC-ROC, por ser robusta a
# desbalanceamento de classes.
CHAMPION_METRIC = "roc_auc"

# Margem minima de melhoria exigida de um challenger sobre o baseline (LR)
# para evitar trocar de campeao por uma diferenca dentro do ruido.
MIN_IMPROVEMENT = 0.01


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    logger.info("Carregando dataset de %s", path)
    return pd.read_csv(path)


def get_candidate_models() -> dict:
    """Os 3 candidatos exigidos pelo Tech Challenge: linear, ensemble e MLP.

    Nenhum usa class_weight aqui -- o desbalanceamento e' tratado de forma
    uniforme via SMOTE no pipeline (ver build_pipeline_with_smote), para que
    os 3 candidatos sejam comparados sob o MESMO protocolo de tratamento de
    desbalanceamento, e nao com estrategias diferentes por modelo.
    """
    return {
        "logistic_regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=300,
            n_jobs=-1,
        ),
        "mlp_classifier": MLPClassifier(
            random_state=RANDOM_STATE,
            hidden_layer_sizes=(32, 16),
            max_iter=500,
            early_stopping=True,  # evita overfitting, para de treinar se parar de melhorar
        ),
    }


def build_pipeline_with_smote(estimator) -> ImbPipeline:
    """Preprocessamento -> SMOTE (so' no fit) -> estimador.

    Usado para os 3 candidatos igualmente. O imblearn.Pipeline garante que o
    SMOTE so' roda durante o .fit() (nos dados de treino) e e' automaticamente
    ignorado no .predict()/.predict_proba(), evitando vazamento de dados
    sinteticos para o conjunto de teste.
    """
    return ImbPipeline(
        steps=[
            ("preprocessing", build_preprocessing_pipeline()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", estimator),
        ]
    )


def evaluate_pipeline(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "recall_churn": recall_score(y_test, y_pred),
    }


def train_and_compare(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Treina os 3 candidatos com o MESMO split/seed e retorna a comparacao.

    Usar o mesmo split para todos os modelos e' o que torna a comparacao justa
    -- do contrario, um modelo poderia parecer melhor so' por ter testado em
    uma fatia de dados mais facil. Da mesma forma, usar a MESMA estrategia de
    tratamento de desbalanceamento (SMOTE) para os 3 evita que um candidato
    pareca melhor ou pior so' por ter recebido um tratamento diferente.
    """
    df = clean_raw_dataframe(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    fitted_pipelines = {}
    rows = []

    for name, estimator in get_candidate_models().items():
        logger.info("Treinando %s...", name)
        pipeline = build_pipeline_with_smote(estimator)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_pipeline(pipeline, X_test, y_test)
        rows.append({"model": name, **metrics})
        fitted_pipelines[name] = pipeline

        logger.info("%s -> %s", name, metrics)

    comparison_df = pd.DataFrame(rows).set_index("model").sort_values(
        CHAMPION_METRIC, ascending=False
    )

    # Relatorio detalhado so' do campeao (definido abaixo), para nao poluir o log
    champion_name = select_champion(comparison_df)
    y_pred_champion = fitted_pipelines[champion_name].predict(X_test)
    logger.info(
        "Classification report do campeao (%s):\n%s",
        champion_name,
        classification_report(y_test, y_pred_champion),
    )

    return comparison_df, fitted_pipelines


def select_champion(comparison_df: pd.DataFrame) -> str:
    """Escolhe o campeao pela metrica tecnica primaria (ROC-AUC, ver ML Canvas).

    Regra do Tech Challenge: Regressao Logistica e' o baseline; Random Forest e
    MLPClassifier PRECISAM supera-la em ROC-AUC (por uma margem minima de
    MIN_IMPROVEMENT, para evitar trocar de campeao por diferenca de ruido)
    para serem escolhidos.
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


if __name__ == "__main__":
    main()