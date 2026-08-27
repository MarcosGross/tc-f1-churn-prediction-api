# TC-F1 — Churn Prediction API

Tech Challenge — Fase 1 · Pós Tech Machine Learning Engineering (FIAP)
Repositório: `tc-f1-churn-prediction-api`

## Contexto
Modelo preditivo de churn para uma operadora de telecomunicações, construído da
EDA até uma API de inferencia, comparando modelos do ecossistema **Scikit-Learn**
(Regressao Logistica, Random Forest/Ensemble, `MLPClassifier` e KNN) e servido
via **FastAPI**.

> Status: **Todas as 4 etapas concluídas.** ML Canvas, EDA, comparação de
> modelos, API refatorada e testada, Model Card completo (com auditoria de viés
> e validação cruzada) e vídeo STAR gravado.

## 🎥 Vídeo de apresentação (método STAR)
[Tech Challenge FIAP Fase 1 — Churn Prediction API · Os Outliers](https://www.youtube.com/watch?v=BQDWi-BpBI8)

## Quickstart (do zero a API rodando)
```bash
# 1. Clonar e entrar no projeto
git clone https://github.com/MarcosGross/tc-f1-churn-prediction-api.git
cd tc-f1-churn-prediction-api

# 2. Criar ambiente virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 3. Baixar o dataset para data/raw/Telco-Customer-Churn.csv (ver secao Dataset)

# 4. Treinar o modelo campeao (gera models/champion_model.joblib)
python -m churn_prediction.train

# 5. Subir a API
uvicorn churn_prediction.api.main:app --reload
```
Acesse http://localhost:8000/docs para testar os endpoints.

**Pre-requisito:** Python 3.11 ou superior.

## Estrutura do repositorio
```
├── src/churn_prediction/
│   ├── preprocessing.py    # limpeza de dados + pipeline sklearn (producao)
│   ├── train.py             # treina e compara os 4 candidatos, salva o campeao
│   ├── inference.py         # calcula probabilidade/predicao de churn
│   ├── model_loader.py      # carrega e cacheia o modelo campeao (.joblib)
│   ├── evaluation.py        # auditoria de vies por subgrupo + validacao cruzada
│   └── api/
│       ├── main.py          # FastAPI: /health e /predict
│       └── schemas.py       # schemas Pydantic de entrada/saida
├── data/
│   ├── raw/                  # dados brutos (nao versionados, ver .gitignore)
│   └── processed/
├── models/                   # modelo campeao (.joblib) + tabela comparativa
├── notebooks/
│   ├── 01_eda.ipynb          # Etapa 1: EDA + baseline documentado
│   └── 02_model_comparison.ipynb  # Etapa 2: comparacao + analise de custo
├── tests/                    # pytest (15 testes: preprocessing, api,
│                              #  inference, model_loader)
├── docs/
│   ├── ml_canvas.md / ml_canvas_en.md
│   └── model_card.md         # performance, limitacoes, vieses (preenchido)
├── pyproject.toml            # dependencias, ruff, pytest
└── uv.lock / .python-version # lockfile opcional para quem usa `uv` em vez
                               # de pip -- nao obrigatorio para rodar o projeto
```

## Setup

**Opcao 1 — pip (recomendado se o grupo nao usa `uv`):**
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
# alternativa, se preferir requirements.txt em vez de pyproject.toml:
# pip install -r requirements.txt
```

**Opcao 2 — uv (mais rapido, usa o uv.lock ja commitado):**
```bash
pip install uv   # ou o instalador proprio: https://docs.astral.sh/uv/
uv sync
```
As duas opcoes instalam as mesmas dependencias declaradas no `pyproject.toml`
— usem a que o grupo preferir, nao ha necessidade de padronizar.

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
# treina Regressao Logistica, Random Forest, MLPClassifier e KNN no MESMO
# split/seed, imprime a tabela comparativa, escolhe o campeao por ROC-AUC
# (com margem minima de 0.01 sobre o baseline) e salva:
#   models/champion_model.joblib
#   models/model_comparison.csv

python -m churn_prediction.train --scenarios
# alem do acima, roda a comparacao exploratoria de cenarios de pre-processamento
# (Raw / Scaled / Scaled+Balanced) e salva models/preprocessing_scenarios.csv
```
Discussao guiada (incluindo analise de custo de negocio):
`notebooks/02_model_comparison.ipynb`.

Tratamento de desbalanceamento por modelo: `class_weight="balanced"` nativo
em Regressao Logistica e Random Forest; oversampling manual (so' no treino)
para o MLPClassifier, que nao suporta `class_weight` no Scikit-Learn.
Testamos tambem SMOTE uniforme via `imbalanced-learn` em branch separada —
resultado equivalente, optamos pelo oversampling manual por nao exigir
dependencia externa nova.

| Modelo | Papel | ROC-AUC |
|---|---|---|
| Regressao Logistica (campea) | Baseline | 0.8415 |
| MLPClassifier | Rede neural simples | 0.8349 |
| Random Forest | Ensemble (arvores) | 0.8218 |
| KNN | Candidato adicional | 0.7656 |

A comparacao de cenarios (`--scenarios`) mostrou que o escalonamento quase nao
altera o ROC-AUC, mas o balanceamento eleva o Recall da classe Churn de ~0.56
para ~0.80 — justificando empiricamente o pipeline de producao adotado.

## 3. Avaliacao complementar (auditoria de vies + validacao cruzada)
```bash
python -m churn_prediction.evaluation
# 1) Auditoria de vies: compara Recall e taxa de falso positivo (FPR) entre
#    subgrupos de gender, SeniorCitizen, Partner e Dependents. Sinaliza gaps
#    acima de 10 pontos percentuais (limite definido no ML Canvas).
# 2) Validacao cruzada estratificada (5 folds) nos 3 candidatos, para verificar
#    se a diferenca entre eles e' maior que a variacao entre splits.
```

Resultados e interpretacao completos em [`docs/model_card.md`](docs/model_card.md),
secoes 5 (limitacoes), 6 (vieses) e 6.1 (investigacao do gap em `Dependents`).

Resumo dos achados:
- **Validacao cruzada**: Regressao Logistica (0.8450 ±0.0134) e MLPClassifier
  (0.8394 ±0.0155) estao **estatisticamente empatados** — a diferenca e' menor
  que o desvio entre folds. Random Forest (0.8251 ±0.0115) fica consistentemente
  atras.
- **Auditoria de vies**: `gender` sem disparidade relevante; `SeniorCitizen`,
  `Partner` e `Dependents` sinalizados, com o gap de FPR sendo o achado mais
  significativo (clientes idosos fieis tem ~2x mais chance de receber abordagem
  desnecessaria).

## 4. Rodar os testes
```bash
pytest -v
```
15 testes cobrindo pre-processamento, API, inferencia e cache do modelo.

## 5. Rodar o linter
```bash
ruff check src tests
```

## 6. Subir a API
```bash
# IMPORTANTE: rode o passo 2 primeiro -- a API carrega models/champion_model.joblib
uvicorn churn_prediction.api.main:app --reload
```
- Docs interativas (Swagger): http://localhost:8000/docs
- A API sobe por padrao em `http://localhost:8000`. Para trocar a porta:
  `uvicorn churn_prediction.api.main:app --reload --port 8080`

### `GET /health`
Verifica se a API esta no ar.
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok"}
```

### `POST /predict`
Recebe os dados de um cliente e retorna a propensao de churn.
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
  }'
```
Resposta:
```json
{
  "churn_probability": 0.8047499944163377,
  "churn_prediction": "Yes"
}
```

**Codigos de resposta:**

| Codigo | Significado |
|---|---|
| 200 | Predicao realizada com sucesso |
| 422 | Payload invalido (campo ausente ou valor fora do dominio esperado) |
| 503 | Modelo nao encontrado — rode `python -m churn_prediction.train` primeiro |

**Threshold de decisao:** 0.5
(`churn_prediction.inference.DEFAULT_CHURN_THRESHOLD`). Este valor **nao foi
otimizado** — ver Model Card secao 6.1 para a analise de threshold e o motivo
de mante-lo em 0.5 nesta versao.

## Arquitetura da API
- `api/main.py` — define os endpoints, traduz erros de dominio (modelo ausente)
  em respostas HTTP.
- `inference.py` — logica pura de calculo de probabilidade/classificacao,
  sem conhecimento de HTTP.
- `model_loader.py` — carregamento do `.joblib` com cache (`lru_cache`), para
  nao recarregar o modelo a cada requisicao.

## Troubleshooting

| Erro | Causa e solucao |
|---|---|
| `503 Modelo nao encontrado` ao chamar `/predict` | O artefato `models/champion_model.joblib` nao existe. Rode `python -m churn_prediction.train`. |
| `FileNotFoundError: ...Telco-Customer-Churn.csv` | Dataset nao baixado. Ver secao **Dataset** e salvar em `data/raw/`. |
| `ModuleNotFoundError: No module named 'churn_prediction'` | Pacote nao instalado no ambiente. Ative o venv e rode `pip install -e ".[dev]"`. |
| `[Errno 48/98] Address already in use` | Porta 8000 ocupada. Suba em outra: `uvicorn churn_prediction.api.main:app --port 8080`. |
| Notebook nao enxerga alteracoes no codigo | O kernel Jupyter mantem os modulos em cache. Use **Restart** e rode as celulas novamente. |

## Time — Os Outliers
| Nome | Principais contribuições |
|---|---|
| Marcos Grosse Moraes | Estrutura do repositório, ML Canvas, EDA e baseline, comparação de modelos, auditoria de viés e validação cruzada, documentação |
| Felipe Marques | Refatoração modular (`inference.py`, `model_loader.py`), suíte de testes, ambiente com `uv` |
| Cristian | Testes de estratégias de desbalanceamento (SMOTE / SMOTENC) |
| João | Candidato KNN e comparação de cenários de pré-processamento |

## Documentacao relacionada
- [Vídeo STAR (5 min)](https://www.youtube.com/watch?v=BQDWi-BpBI8)
- ML Canvas — template oficial OWNML preenchido:
  [`docs/OWNML_Machine_Learning_Canvas_Churn.pdf`](docs/OWNML_Machine_Learning_Canvas_Churn.pdf)
  · versões em markdown: [PT](docs/ml_canvas.md) · [EN](docs/ml_canvas_en.md)
- [Model Card](docs/model_card.md) — performance, limitacoes, vieses e
  investigacao de mitigacao
