# TC-F1 — Churn Prediction API

Tech Challenge — Fase 1 · Pós Tech Machine Learning Engineering (FIAP)
Repositório: `tc-f1-churn-prediction-api`

## Contexto
Modelo preditivo de churn para uma operadora de telecomunicações, construído da
EDA até uma API de inferencia, comparando modelos do ecossistema **Scikit-Learn**
(Regressao Logistica, Random Forest/Ensemble e `MLPClassifier`) e servido via
**FastAPI**.

> Status: Etapa 1 fechada (ML Canvas + EDA + baseline). Etapa 2 (comparação
> de modelos) com código pronto em `train.py` e notebook
> `02_model_comparison.ipynb` — execução pelo grupo ainda pendente.

## Estrutura do repositorio
```
├── src/churn_prediction/
│   ├── preprocessing.py   # limpeza de dados + pipeline sklearn (producao)
│   ├── train.py            # treina e compara os 3 modelos, salva o campeao
│   └── api/
│       ├── main.py         # FastAPI: /health e /predict (carrega o campeao)
│       └── schemas.py      # schemas Pydantic de entrada/saida
├── data/
│   ├── raw/                 # dados brutos (nao versionados, ver .gitignore)
│   └── processed/
├── models/                  # modelo campeao salvo (.joblib) + tabela comparativa
├── notebooks/
│   ├── 01_eda.ipynb         # Etapa 1: EDA + baseline documentado
│   └── 02_model_comparison.ipynb  # Etapa 2: comparacao + analise de custo
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

## 1. Explorar os dados (Etapa 1)
Abra `notebooks/01_eda.ipynb` para ver a analise exploratoria completa:
qualidade dos dados, distribuicao do target, churn por categoria, e o
treino/avaliacao do baseline (Regressao Logistica).

## 2. Treinar e comparar os modelos (Etapa 2)
```bash
python -m churn_prediction.train
# treina Regressao Logistica, Random Forest e MLPClassifier no MESMO split/seed,
# imprime a tabela comparativa, escolhe o campeao por ROC-AUC e salva:
#   models/champion_model.joblib
#   models/model_comparison.csv
```
Discussao guiada (incluindo analise de custo de negocio):
`notebooks/02_model_comparison.ipynb`.

| Modelo | Papel | Status |
|---|---|---|
| Regressao Logistica | Baseline | Codigo pronto, execucao pendente pelo grupo |
| Random Forest | Ensemble (arvores) | Codigo pronto, execucao pendente pelo grupo |
| MLPClassifier (Scikit-Learn) | Rede neural simples | Codigo pronto, execucao pendente pelo grupo |

## 3. Rodar os testes
```bash
pytest
```

## 4. Subir a API
```bash
# IMPORTANTE: rode o passo 2 primeiro -- a API carrega models/champion_model.joblib
uvicorn churn_prediction.api.main:app --reload
```
- Docs interativas: http://localhost:8000/docs
- `GET /health` -> `{"status": "ok"}`
- `POST /predict` -> recebe os dados de um cliente, retorna `churn_probability`
  e `churn_prediction` (Yes/No)

## Time
| Nome | Papel |
|---|---|
| _preencher_ | _preencher_ |

## Documentacao relacionada
- [ML Canvas (PT)](docs/ml_canvas.md) · [ML Canvas (EN)](docs/ml_canvas_en.md)
- [Model Card](docs/model_card.md) — template, preenchido na Etapa 4
