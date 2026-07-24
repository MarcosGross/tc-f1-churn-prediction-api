# TC-F1 — Churn Prediction API

Tech Challenge — Fase 1 · Pós Tech Machine Learning Engineering (FIAP)
Repositório: `tc-f1-churn-prediction-api`

## Contexto
Modelo preditivo de churn para uma operadora de telecomunicações, construído da
EDA até uma API de inferencia, comparando modelos do ecossistema **Scikit-Learn**
(Regressao Logistica, Random Forest/Ensemble e `MLPClassifier`) e servido via
**FastAPI**.

> Status: ML Canvas fechado. EDA e treino do baseline ainda **não foram
> executados** pelo grupo — são o próximo passo.

## Estrutura do repositorio
```
├── src/churn_prediction/
│   ├── preprocessing.py   # limpeza de dados + pipeline sklearn (producao)
│   ├── train.py            # treino do baseline, salva modelo em models/
│   └── api/
│       ├── main.py         # FastAPI: /health e /predict
│       └── schemas.py      # schemas Pydantic de entrada/saida
├── data/
│   ├── raw/                 # dados brutos (nao versionados, ver .gitignore)
│   └── processed/
├── models/                  # modelo campeao salvo (.joblib)
├── notebooks/
│   └── 01_eda.ipynb         # EDA + baseline documentado
├── tests/                   # pytest (>=2 testes obrigatorios)
├── docs/
│   ├── ml_canvas.md / ml_canvas_en.md
│   └── model_card.md        # template, preenchido de fato na Etapa 4
└── pyproject.toml
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# alternativa, se preferir requirements.txt em vez de pyproject.toml:
# pip install -r requirements.txt
```

## Dataset
Baixe o CSV (Telco Customer Churn — IBM) e salve em:
```
data/raw/Telco-Customer-Churn.csv
```
Fonte: github.com/IBM/telco-customer-churn-on-icp4d

## Rodando o treino do baseline
```bash
python -m churn_prediction.train
# treina Regressao Logistica, imprime metricas (ROC-AUC, PR-AUC, F1, Recall)
# e salva o modelo em models/baseline_logistic_regression.joblib
```

## Rodando os testes
```bash
pytest
```

## Rodando a API
```bash
# IMPORTANTE: treine o modelo primeiro (passo acima), a API carrega o .joblib
uvicorn churn_prediction.api.main:app --reload
```
- Docs interativas: http://localhost:8000/docs
- `GET /health` -> `{"status": "ok"}`
- `POST /predict` -> recebe os dados de um cliente, retorna `churn_probability`
  e `churn_prediction` (Yes/No)

## Notebook de EDA
Abra `notebooks/01_eda.ipynb` para ver a analise exploratoria completa: qualidade
dos dados, distribuicao do target, churn por categoria, e o treino/avaliacao do
baseline.

## Modelos a comparar (Etapa 2)
| Modelo | Papel | Status |
|---|---|---|
| Regressao Logistica | Baseline | Pendente — codigo pronto em `train.py`, ainda nao executado |
| Random Forest (ou outro ensemble) | Modelo nao-linear | Pendente |
| MLPClassifier (Scikit-Learn) | Rede neural simples | Pendente |

## Time
| Nome | Papel |
|---|---|
| _preencher_ | _preencher_ |

## Documentacao relacionada
- [ML Canvas (PT)](docs/ml_canvas.md) · [ML Canvas (EN)](docs/ml_canvas_en.md)
- [Model Card](docs/model_card.md) — template, preenchido na Etapa 4
