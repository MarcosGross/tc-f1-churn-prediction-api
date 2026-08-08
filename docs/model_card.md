# Model Card — Churn Prediction

Tech Challenge Fase 1 · Pós Tech Machine Learning Engineering (FIAP)
Repositório: `tc-f1-churn-prediction-api`

---

## 1. Visão geral

| Item | Descrição |
|---|---|
| **Tarefa** | Classificação binária — prever se um cliente vai cancelar (churn) |
| **Modelo campeão** | Regressão Logística (`class_weight="balanced"`, `max_iter=1000`, `random_state=42`) |
| **Artefato** | `models/champion_model.joblib` (Pipeline sklearn: pré-processamento + estimador) |
| **Dataset** | Telco Customer Churn (IBM), 7.043 registros, 21 colunas |
| **Split** | 80/20 estratificado, seed fixa (`random_state=42`) |
| **Servido via** | API FastAPI (`/predict`), threshold de decisão 0.5 |

### Features utilizadas
- **Numéricas** (padronizadas com `StandardScaler`): `tenure`, `MonthlyCharges`, `TotalCharges`
- **Categóricas** (one-hot, `handle_unknown="ignore"`): `gender`, `SeniorCitizen`, `Partner`,
  `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`,
  `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`,
  `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Excluída**: `customerID` (identificador, sem poder preditivo)

---

## 2. Performance

Avaliado no conjunto de teste (1.409 registros, nunca vistos no treino).

### Modelo campeão (Regressão Logística)

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.8415 |
| PR-AUC | 0.6329 |
| F1-score (classe Churn) | 0.6136 |
| Recall (classe Churn) | 0.7834 |
| Acurácia | 0.74 |

**Relatório de classificação:**

| Classe | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| 0 (não churn) | 0.90 | 0.72 | 0.80 | 1.035 |
| 1 (churn) | 0.50 | 0.78 | 0.61 | 374 |

### Comparação com os candidatos

| Modelo | ROC-AUC | PR-AUC | F1 | Recall (Churn) |
|---|---|---|---|---|
| **Regressão Logística (campeã)** | **0.8415** | **0.6329** | 0.6136 | 0.7834 |
| MLPClassifier | 0.8349 | 0.6213 | 0.6205 | 0.8021 |
| Random Forest | 0.8218 | 0.6108 | 0.5958 | 0.6444 |

**Critério de escolha:** ROC-AUC (métrica técnica primária definida no ML Canvas),
com margem mínima de 0.01 exigida de um challenger para destronar o baseline —
evitando trocar de modelo por diferença dentro do ruído.

---

## 3. Análise de custo de negócio

Premissas (valores ilustrativos, definidos no bloco *Impact Simulation* do ML Canvas):
- Custo de perder um cliente (LTV perdido) = **R$ 1.200** por falso negativo
- Custo de uma oferta de retenção desperdiçada = **R$ 50** por falso positivo

| Modelo | Falsos Negativos | Falsos Positivos | Custo total |
|---|---|---|---|
| Regressão Logística | 81 | 288 | R$ 111.600 |
| **MLPClassifier** | 74 | 293 | **R$ 103.450** |
| Random Forest | 133 | 194 | R$ 169.300 |

**Trade-off documentado:** o MLPClassifier teria custo de negócio ~7% menor que o
campeão, apesar de perder em ROC-AUC. Mantivemos a Regressão Logística como campeã
por coerência com o critério técnico definido previamente no ML Canvas, mas
registramos aqui que **uma decisão orientada puramente por custo escolheria o MLP**.
Isso é uma decisão em aberto para revisão futura, junto com o ajuste de threshold
(ver seção 6).

---

## 4. Tratamento de desbalanceamento

O dataset é moderadamente desbalanceado: **73,5% não-churn / 26,5% churn**.

| Modelo | Estratégia |
|---|---|
| Regressão Logística | `class_weight="balanced"` (nativo do sklearn) |
| Random Forest | `class_weight="balanced"` (nativo do sklearn) |
| MLPClassifier | Oversampling manual da classe minoritária, aplicado **somente no treino** |

**Por que estratégias diferentes:** o `MLPClassifier` do Scikit-Learn não possui
parâmetro `class_weight`. Sem compensação, seu Recall na classe Churn era 0.5294;
com oversampling, subiu para 0.8021 — confirmando que estava sendo penalizado
injustamente na comparação.

### Alternativas testadas e descartadas

Avaliamos duas abordagens baseadas em geração de exemplos sintéticos, ambas via
`imbalanced-learn`, em branch separada (`refactor--etapa-2-testes-SMOTENC`).

**1. SMOTE aplicado uniformemente aos 3 candidatos**

A motivação era metodológica: usar a *mesma* estratégia de tratamento para todos os
modelos, evitando que um candidato parecesse melhor ou pior por ter recebido
tratamento diferente. O campeão e as métricas se mantiveram equivalentes ao
oversampling manual.

**2. SMOTENC — tecnicamente mais correto, sem ganho empírico**

Testamos SMOTENC como alternativa ao SMOTE puro, especificamente para o
MLPClassifier (único modelo sem suporte a `class_weight`). A motivação técnica é
sólida: a maioria das features do dataset é categórica, e o SMOTE puro pode gerar
**combinações de categoria inválidas** ao interpolar entre vizinhos — produzindo
valores fracionários em colunas que deveriam ser binárias após o one-hot encoding.

Apesar de mais adequado a dados mistos, o SMOTENC **não trouxe ganho empírico**:

| Modelo | Recall com `class_weight` | Recall com SMOTE puro | Recall com SMOTENC |
|---|---|---|---|
| Random Forest | **0.6444** | 0.5775 | 0.5829 |

O Random Forest não melhorou de forma significativa (0.5829 vs 0.5775), e ambos
ficaram abaixo do resultado com `class_weight="balanced"` (0.6444). O recall da
Regressão Logística e do MLPClassifier **piorou** em relação ao SMOTE puro.

**Hipótese não testada** para o resultado: o SMOTENC calcula a distância entre
vizinhos usando as features numéricas em **escala original**, antes do
`StandardScaler` do pipeline. Como `TotalCharges` varia de 0 a ~8.700 enquanto
`tenure` vai de 0 a 72, a métrica de distância fica dominada pela feature de maior
magnitude — o que pode ter degradado a qualidade dos exemplos sintéticos gerados.
Verificar isso exigiria reordenar o pipeline para escalar antes do resample,
mantendo o tratamento correto das colunas categóricas.

**Decisão:** mantivemos `class_weight="balanced"` (Regressão Logística e Random
Forest) e oversampling manual (MLPClassifier). Justificativas:
1. Melhor resultado empírico entre as três abordagens testadas.
2. Não exige dependência externa (`imbalanced-learn`), mantendo o escopo alinhado
   ao enunciado (Scikit-Learn, FastAPI, Pytest).
3. Menor complexidade de pipeline, sem necessidade de `imblearn.pipeline.Pipeline`.

---

## 5. Limitações conhecidas

- **Dataset estático**: é uma foto única, sem dimensão temporal. Não captura
  sazonalidade, tendências de mercado, ou mudanças de comportamento ao longo do tempo.
- **Baixa precisão na classe Churn (0.50)**: metade dos clientes sinalizados como
  "vai cancelar" na verdade não cancelaria. Isso é uma consequência intencional do
  `class_weight="balanced"` (priorizamos não perder churners reais), mas significa
  que a equipe de retenção gastará esforço em falsos alarmes.
- **Dados de uma única operadora fictícia** (amostra pública da IBM). Não há garantia
  de generalização para outra operadora sem retreino com dados próprios.
- **`TotalCharges` imputado como 0** para 11 clientes com `tenure=0` (recém-chegados
  sem ciclo de cobrança fechado). É uma decisão defensável, mas pode subestimar o
  risco desses clientes novos.
- **Diferença entre os candidatos está dentro do ruído** ⚠️: executamos validação
  cruzada estratificada (5 folds) para verificar a robustez da escolha. Resultado:

  | Modelo | ROC-AUC médio | Desvio padrão | Mín | Máx |
  |---|---|---|---|---|
  | Regressão Logística | 0.8450 | ±0.0134 | 0.8247 | 0.8627 |
  | MLPClassifier | 0.8394 | ±0.0155 | 0.8188 | 0.8625 |
  | Random Forest | 0.8251 | ±0.0115 | 0.8073 | 0.8384 |

  A diferença entre Regressão Logística e MLPClassifier (0.0056) é **menor que o
  desvio padrão entre folds** (~0.014). Estatisticamente, os dois estão empatados —
  a vitória da Regressão Logística no split único não é robusta. O Random Forest,
  esse sim, fica consistentemente atrás nos 5 folds.

  Isso valida em retrospecto a regra `MIN_IMPROVEMENT = 0.01` adotada em
  `select_champion()`: ela impede a troca de campeão por diferenças de ruído.

  *Ressalva:* a validação cruzada do MLPClassifier rodou **sem** o oversampling
  (que precisaria ser reaplicado dentro de cada fold para ser metodologicamente
  correto). O número dele é, portanto, levemente pessimista — o que apenas reforça
  a conclusão de empate técnico.
- **Sem tuning de hiperparâmetros**: todos os modelos usam configurações
  próximas do padrão. Um `GridSearchCV`/`RandomizedSearchCV` poderia mudar o ranking.

---

## 6. Vieses potenciais

**Medição executada** (`python -m churn_prediction.evaluation`) sobre o conjunto de
teste, comparando Recall e taxa de falso positivo (FPR) entre subgrupos. Limite de
disparidade definido no ML Canvas: 10 pontos percentuais.

### Performance por subgrupo

| Atributo | Grupo | n | Taxa de churn | Recall | FPR |
|---|---|---|---|---|---|
| gender | Female | 687 | 0.2809 | 0.7772 | 0.2692 |
| gender | Male | 722 | 0.2507 | 0.7901 | 0.2865 |
| SeniorCitizen | 0 | 1187 | 0.2325 | 0.7319 | 0.2492 |
| SeniorCitizen | 1 | 222 | 0.4414 | 0.9286 | 0.4919 |
| Partner | No | 736 | 0.3370 | 0.8226 | 0.3750 |
| Partner | Yes | 673 | 0.1872 | 0.7063 | 0.1920 |
| Dependents | No | 978 | 0.3098 | 0.8251 | 0.3570 |
| Dependents | Yes | 431 | 0.1647 | 0.6056 | 0.1306 |

### Disparidades detectadas

| Atributo | Gap de Recall | Gap de FPR | Sinalizado |
|---|---|---|---|
| `gender` | 0.0129 | 0.0173 | ❌ Não |
| `Partner` | 0.1163 | 0.1830 | ⚠️ Sim |
| `SeniorCitizen` | 0.1967 | 0.2427 | ⚠️ Sim |
| `Dependents` | 0.2195 | 0.2264 | ⚠️ Sim |

### Interpretação

**`gender` não apresenta viés detectável.** O gap é de apenas ~1,5 pontos percentuais
em ambas as métricas — bem abaixo do limite. Isso também sugere que a variável carrega
pouco sinal preditivo; sendo um atributo sensível sem contribuição clara, permanece
candidata a remoção em versão futura, por princípio de minimização de dados.

**Gap de Recall é parcialmente explicável pela taxa base.** Clientes idosos cancelam
44% das vezes contra 23% dos demais; sem dependentes, 31% contra 16%. É esperado que
o modelo detecte mais churners em grupos onde há proporcionalmente mais churners.
Recall desigual, isoladamente, não prova tratamento injusto.

**O gap de FPR é o achado que exige atenção.** Entre clientes que **não** iriam
cancelar:
- **49% dos idosos** foram falsamente sinalizados como risco, contra 25% dos não-idosos
- **36% dos clientes sem dependentes**, contra 13% dos com dependentes
- **38% dos clientes sem parceiro(a)**, contra 19% dos com parceiro(a)

Na prática: clientes idosos, solteiros ou sem dependentes que são fiéis à operadora
têm aproximadamente o **dobro de chance** de receber uma abordagem de retenção
desnecessária. A taxa base não justifica esse gap — trata-se de clientes que o modelo
classificou erradamente.

### Severidade e mitigação

O dano concreto neste caso de uso é limitado: um falso positivo resulta em contato
comercial ou oferta de desconto não solicitada, não em negação de serviço, alteração
de preço ou qualquer decisão adversa. Ainda assim, é tratamento desigual mensurável e
deve ser monitorado.

Mitigações a considerar em versão futura:
1. Remover `gender` (sem custo de performance aparente) e avaliar impacto de remover
   `SeniorCitizen`, atributo protegido em diversas jurisdições.
2. Calibrar o threshold de decisão por subgrupo, equalizando o FPR entre grupos.
3. Definir alerta de monitoramento em produção quando o gap de FPR exceder o limite.

### Risco adicional — viés de feedback

Se as ações de retenção geradas pelo modelo afetarem quem cancela ou não, os dados
futuros de treino serão contaminados pela própria intervenção: um cliente retido por
causa do modelo aparecerá como "não churn", mascarando que o modelo o havia
identificado corretamente. Isso degrada a qualidade do retreino ao longo do tempo e
requer estratégia de mitigação (ex.: grupo de controle que não recebe intervenção).

## 6.1 Investigação conduzida — mitigação do gap em `Dependents`

Após a auditoria detectar disparidade em `Dependents` (recall de 0.6056, ou seja,
~40% dos churners desse grupo não detectados), conduzimos uma investigação sobre
viabilidade de correção. Documentamos aqui o resultado, as opções avaliadas e a
justificativa de **não implementar mitigação nesta versão**.

### Achado 1 — A amostra do subgrupo é pequena

| Grupo | n | Churners reais | Detectados | Perdidos |
|---|---|---|---|---|
| Dependents=No | 978 | 303 | 250 | 53 |
| Dependents=Yes | 431 | **71** | 43 | 28 |

Bootstrap (2.000 reamostragens) do Recall:

| Grupo | Recall | IC 95% |
|---|---|---|
| Dependents=No | 0.8251 | [0.7822, 0.8680] |
| Dependents=Yes | 0.6056 | **[0.4930, 0.7183]** |

O gap é real — os intervalos não se sobrepõem — mas a **magnitude** tem incerteza
considerável no grupo minoritário. Qualquer threshold calibrado sobre 71 casos tem
risco elevado de não generalizar.

### Achado 2 — Recall e FPR são sintomas da mesma causa

O modelo é uniformemente **mais conservador** com quem tem dependentes: sinaliza menos
clientes desse grupo, portanto detecta menos churners (Recall 0.61 vs 0.83) **e**
incomoda menos clientes fiéis (FPR 0.13 vs 0.36). A causa é a taxa base menor de churn
(16,5% vs 31,0%), que desloca a distribuição de probabilidades do grupo para baixo —
e o corte fixo de 0.5 acaba seccionando cada distribuição em pontos diferentes.

Não são duas falhas independentes: é uma só, manifestada em duas métricas.

### Achado 3 — Um threshold específico por grupo corrigiria ambos os gaps

Simulamos aplicar threshold de **0.25** apenas para `Dependents=Yes`, mantendo 0.5
para os demais:

| Configuração | Gap de Recall | Gap de FPR |
|---|---|---|
| Threshold 0.5 global (atual) | 0.2194 | 0.2265 |
| Threshold 0.25 para `Dependents=Yes` | **0.0059** | **0.0570** |

O Recall equaliza (0.8310 vs 0.8251) e o gap de FPR **também** diminui — porque o grupo
partia de uma posição conservadora em ambas as dimensões. O custo real da mudança não é
um trade-off entre métricas de equidade, e sim operacional: o FPR do grupo sobe de 0.13
para 0.30, gerando mais ofertas de retenção desperdiçadas nesse segmento.

### Achado 4 — A otimização global expõe um problema nas premissas de custo

Otimizando o threshold **global** pela função de custo da seção 3, o valor ótimo é
**0.11**, reduzindo o custo total de R$ 111.600 para R$ 39.650 (queda de 64%) e
colapsando o gap de Recall para 0.02.

Porém, nessa configuração o FPR chega a **0.7585** no grupo majoritário — sinalizar
três de cada quatro clientes fiéis como risco de churn é inviável operacionalmente.

Isso não é falha do método, e sim indício de que a razão de custo adotada
(R$ 1.200 / R$ 50 = **24:1**) é extrema demais e distorce a otimização. Os valores são
placeholders ilustrativos definidos no ML Canvas por não termos acesso a dados reais de
LTV e custo de retenção. **Recomendação: revisar essas premissas com stakeholders antes
de usar a função de custo para qualquer decisão de deployment.**

### Decisão: não implementar mitigação nesta versão

Avaliamos `fairlearn.postprocessing.ThresholdOptimizer` como ferramenta de correção
pós-processamento. Optamos por não adotá-lo, por três razões:

1. **Dependência externa nova** — o mesmo critério que nos levou a rejeitar
   `imbalanced-learn` em favor do oversampling manual (seção 4) se aplica aqui;
   adotá-lo agora seria incoerente com essa decisão.
2. **Exige o atributo protegido em tempo de inferência** — a API precisaria receber
   `Dependents` e aplicar threshold diferenciado por grupo. Usar um atributo protegido
   *explicitamente na regra de decisão* tem implicações jurídicas mais delicadas do que
   mantê-lo como uma feature entre outras.
3. **Risco de overfitting** — conforme o Achado 1, a calibração recairia sobre 71 casos
   com IC de recall de [0.49, 0.72].

### Recomendações para versão futura

1. Reavaliar as premissas de custo (Achado 4) antes de otimizar qualquer threshold.
2. Coletar volume maior de dados no subgrupo antes de calibrar mitigação específica.
3. Se a mitigação for adotada, validar o threshold por grupo em conjunto de validação
   separado — nunca no mesmo conjunto usado para medir o gap.
4. Avaliar a alternativa mais simples: remover `Dependents` e `Partner` do conjunto de
   features e verificar se a performance geral se mantém.

## 7. Cenários de falha

- **Categoria não vista em produção** (ex.: um novo `PaymentMethod`): o
  `OneHotEncoder` está configurado com `handle_unknown="ignore"`, então a API não
  quebra — mas a predição para esse cliente será menos confiável, pois a informação
  daquela categoria é simplesmente descartada.
- **Modelo ausente**: se `models/champion_model.joblib` não existir, a API retorna
  HTTP 503 com instrução de rodar o treino (comportamento coberto por teste).
- **Drift de distribuição**: mudanças de preço, novos planos, ou mudança de perfil de
  cliente degradam a performance silenciosamente — o modelo continua respondendo,
  só que pior. Requer monitoramento (ver ML Canvas, bloco *Monitoring*).
- **Payload inválido**: campos fora dos valores esperados são rejeitados pela
  validação Pydantic com HTTP 422 (comportamento coberto por teste).

---

## 8. Uso pretendido

**Para que serve:** gerar uma lista priorizada de clientes com risco de cancelamento,
consumida pela equipe de retenção/CRM para ação proativa (desconto, upgrade, contato).

**Para que NÃO serve:**
- Decisão automatizada sem revisão humana — dada a precisão de 0.50 na classe Churn,
  metade dos sinalizados são falsos alarmes.
- Negar serviço, alterar preço, ou qualquer decisão adversa ao cliente.
- Inferir características pessoais além do risco de churn.

**Threshold de decisão:** 0.5 (`churn_prediction.inference.DEFAULT_CHURN_THRESHOLD`).
Este valor não foi otimizado — dada a análise de custo da seção 3, ajustar o threshold
é a alavanca mais direta para reduzir custo de negócio sem trocar de modelo.

---

## 9. Reprodutibilidade

```bash
pip install -e ".[dev]"
python -m churn_prediction.train   # gera champion_model.joblib e model_comparison.csv
pytest -v                           # 15 testes
ruff check src tests                # linting
```

Seeds fixas (`random_state=42`) em split e em todos os estimadores. Ambiente
declarado em `pyproject.toml` (e `uv.lock`, opcional).

---

## 10. Referências

- ML Canvas do projeto: `docs/ml_canvas.md` (PT) / `docs/ml_canvas_en.md` (EN)
- Notebook de EDA: `notebooks/01_eda.ipynb`
- Notebook de comparação: `notebooks/02_model_comparison.ipynb`
- Dataset: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)
