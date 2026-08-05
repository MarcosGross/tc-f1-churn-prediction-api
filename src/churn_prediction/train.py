"""Tratamento de desbalanceamento (estrategia MISTA, por modelo):
- LogisticRegression e RandomForestClassifier usam class_weight="balanced"
  (nativo do sklearn, nao gera dados sinteticos, e' a tecnica que teve o
  melhor recall em testes empiricos para esses 2 modelos).
- MLPClassifier usa SMOTENC (oversampling), pois nao suporta class_weight
  nem sample_weight -- essa e' uma limitacao da API do sklearn, nao do
  algoritmo em si. Usamos SMOTENC (nao SMOTE puro) porque a maioria das
  features deste dataset e' categorica, e SMOTENC evita gerar combinacoes
  de categoria invalidas (ex.: Contract_OneYear=0.34) ao copiar o valor de
  um vizinho em vez de interpolar.

NOTA DE DESIGN: esta versao abandona o principio de "mesmo protocolo de
desbalanceamento para todos os candidatos" em favor de "melhor tecnica
disponivel para cada modelo". Testamos empiricamente 3 alternativas
uniformes (class_weight-only quando possivel, SMOTE puro, SMOTENC) e em
todas elas o Random Forest teve recall_churn pior sob oversampling do que
sob class_weight (0.64 com class_weight vs. ~0.58 com SMOTE/SMOTENC), entao
optamos por dar a cada modelo sua melhor estrategia individual em vez de
uniformizar. Ver ML Canvas / README para a justificativa completa dessa
decisao e os resultados comparativos entre as versoes testadas.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTENC
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
    CATEGORICAL_FEATURES,
    build_full_pipeline,
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

    LR e RF usam class_weight="balanced" (suportado nativamente e com melhor
    recall empirico). MLPClassifier nao suporta class_weight/sample_weight,
    entao recebe tratamento via SMOTENC no pipeline (ver
    build_mlp_pipeline_with_smotenc / build_pipeline_for).
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
            early_stopping=True,  # evita overfitting, para de treinar se parar de melhorar
        ),
    }


def get_categorical_feature_mask(X: pd.DataFrame) -> list[bool]:
    """Mascara booleana indicando quais colunas de X sao categoricas.

    Necessario porque SMOTENC precisa saber, por posicao, quais colunas
    tratar como categoricas -- e ele roda ANTES do OneHotEncoder, entao
    ainda estamos com as colunas originais (nao expandidas em 0/1).
    """
    return [col in CATEGORICAL_FEATURES for col in X.columns]


def build_mlp_pipeline_with_smotenc(estimator, categorical_mask: list[bool]) -> ImbPipeline:
    """So' para o MLP: SMOTENC -> preprocessamento -> MLP.

    MLPClassifier nao aceita class_weight nem sample_weight, entao a forma
    de compensar o desbalanceamento e' reamostrar os dados de treino antes
    do fit. SMOTENC (em vez de SMOTE puro) preserva categorias validas ao
    nao interpolar as colunas categoricas. O imblearn.Pipeline garante que
    o SMOTENC so' roda durante o .fit(), nunca no .predict()/.predict_proba(),
    evitando vazamento de dados sinteticos para o conjunto de teste.
    """
    return ImbPipeline(
        steps=[
            (
                "smote",
                SMOTENC(categorical_features=categorical_mask, random_state=RANDOM_STATE),
            ),
            ("preprocessing", build_preprocessing_pipeline()),
            ("model", estimator),
        ]
    )


def build_pipeline_for(name: str, estimator, categorical_mask: list[bool]):
    """Escolhe a estrategia de desbalanceamento certa para cada modelo.

    LR e RF: class_weight="balanced" ja esta' no proprio estimator (definido
    em get_candidate_models), entao so' precisam do pipeline padrao de
    preprocessamento. MLP: precisa do pipeline com SMOTENC, pois nao suporta
    class_weight.
    """
    if name == "mlp_classifier":
        return build_mlp_pipeline_with_smotenc(estimator, categorical_mask)
    return build_full_pipeline(estimator)


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
    uma fatia de dados mais facil. O tratamento de desbalanceamento e' MISTO
    por modelo (ver docstring do modulo) -- decisao documentada e testada
    empiricamente, nao uma escolha arbitraria.
    """
    df = clean_raw_dataframe(df)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    categorical_mask = get_categorical_feature_mask(X_train)

    fitted_pipelines = {}
    rows = []

    for name, estimator in get_candidate_models().items():
        logger.info("Treinando %s...", name)
        pipeline = build_pipeline_for(name, estimator, categorical_mask)
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