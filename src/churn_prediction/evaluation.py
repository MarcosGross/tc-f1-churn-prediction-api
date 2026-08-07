"""Avaliacoes complementares do modelo campeao.

Cobre duas lacunas apontadas no Model Card:
1. Auditoria de vies por subgrupo demografico (secao 6 do Model Card).
2. Validacao cruzada estratificada, para verificar se a diferenca entre os
   candidatos e' real ou esta' dentro da variacao do split (secao 5).

Uso:
    python -m churn_prediction.evaluation
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from churn_prediction.preprocessing import build_full_pipeline, clean_raw_dataframe, split_features_target
from churn_prediction.train import (
    MODELS_REQUIRING_OVERSAMPLING,
    RANDOM_STATE,

    get_candidate_models,
    load_data,
    oversample_minority,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Atributos sensiveis auditados. Limite de disparidade definido no ML Canvas
# (bloco Impact Simulation): sinalizar se a diferenca entre grupos passar de
# 10 pontos percentuais.
SENSITIVE_ATTRIBUTES = ["gender", "SeniorCitizen", "Partner", "Dependents"]
DISPARITY_THRESHOLD = 0.10

N_SPLITS = 5


def audit_subgroup_performance(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Compara Recall e taxa de falso positivo (FPR) entre subgrupos.

    Recall baixo em um grupo = o modelo deixa escapar mais churners daquele grupo.
    FPR alto em um grupo = o modelo incomoda mais clientes fieis daquele grupo
    com ofertas desnecessarias. Ambos sao formas de tratamento desigual.
    """
    y_pred = model.predict(X_test)
    rows = []

    for attribute in SENSITIVE_ATTRIBUTES:
        for group_value in sorted(X_test[attribute].unique()):
            mask = X_test[attribute] == group_value
            y_true_group = y_test[mask]
            y_pred_group = y_pred[mask.to_numpy()]

            # Precisa das duas classes presentes para a matriz de confusao 2x2
            if y_true_group.nunique() < 2:
                continue

            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
            recall = tp / (tp + fn) if (tp + fn) else float("nan")
            fpr = fp / (fp + tn) if (fp + tn) else float("nan")

            rows.append(
                {
                    "attribute": attribute,
                    "group": group_value,
                    "n": int(mask.sum()),
                    "churn_rate": round(float(y_true_group.mean()), 4),
                    "recall": round(float(recall), 4),
                    "fpr": round(float(fpr), 4),
                }
            )

    return pd.DataFrame(rows)


def report_disparities(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a diferenca max-min por atributo e sinaliza o que passa do limite."""
    rows = []
    for attribute, group_df in audit_df.groupby("attribute"):
        recall_gap = group_df["recall"].max() - group_df["recall"].min()
        fpr_gap = group_df["fpr"].max() - group_df["fpr"].min()
        rows.append(
            {
                "attribute": attribute,
                "recall_gap": round(float(recall_gap), 4),
                "fpr_gap": round(float(fpr_gap), 4),
                "flagged": bool(
                    recall_gap > DISPARITY_THRESHOLD or fpr_gap > DISPARITY_THRESHOLD
                ),
            }
        )
    return pd.DataFrame(rows)


def cross_validate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Roda validacao cruzada estratificada (5 folds) nos 3 candidatos.

    O desvio padrao entre folds mostra se a diferenca de ROC-AUC entre os
    modelos e' maior que a variacao natural causada pela escolha do split.
    """
    df = clean_raw_dataframe(df)
    X, y = split_features_target(df)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, estimator in get_candidate_models().items():
        logger.info("Validacao cruzada de %s (%d folds)...", name, N_SPLITS)

        # Nota: o oversampling do MLP nao entra no cross_val_score porque
        # precisaria ser reaplicado dentro de cada fold. Para o MLP, o CV aqui
        # mede a estabilidade SEM oversampling -- comparar com cautela.
        pipeline = build_full_pipeline(estimator)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

        rows.append(
            {
                "model": name,
                "roc_auc_mean": round(float(scores.mean()), 4),
                "roc_auc_std": round(float(scores.std()), 4),
                "roc_auc_min": round(float(scores.min()), 4),
                "roc_auc_max": round(float(scores.max()), 4),
            }
        )

    return pd.DataFrame(rows).set_index("model").sort_values("roc_auc_mean", ascending=False)


def main() -> None:
    import joblib

    from churn_prediction.model_loader import MODEL_PATH

    df = load_data()

    # --- Auditoria de vies (usa o campeao ja treinado e o mesmo split do train.py)
    df_clean = clean_raw_dataframe(df)
    X, y = split_features_target(df_clean)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    audit_df = audit_subgroup_performance(model, X_test, y_test)
    logger.info("Performance por subgrupo:\n%s", audit_df.to_string(index=False))

    disparity_df = report_disparities(audit_df)
    logger.info("Disparidades (limite %.2f):\n%s", DISPARITY_THRESHOLD,
                disparity_df.to_string(index=False))

    # --- Validacao cruzada
    cv_df = cross_validate_candidates(df)
    logger.info("Validacao cruzada (%d folds):\n%s", N_SPLITS, cv_df)


if __name__ == "__main__":
    main()
