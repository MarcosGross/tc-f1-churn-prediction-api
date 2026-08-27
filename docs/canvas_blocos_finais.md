# ML Canvas — blocos finais (versão validada)

Textos prontos para colar no template OWNML. Iteração 2 · Data: atualizar.
Todas as marcações "(rascunho)" foram removidas e o conteúdo reflete o que foi
efetivamente entregue.

---

## PREDICTION TASK

Classificação binária supervisionada. Entidade: cliente individual
(`customerID`). Resultados possíveis: Churn = Sim / Não. Neste dataset o
resultado é observado como snapshot estático; em produção seria observado no
evento de cancelamento/não renovação do contrato, registrado pelo sistema de
billing/CRM.

---

## DECISIONS

*(sem alteração — bloco já estava validado)*

As previsões geram um score de risco de churn por cliente. Clientes acima do
limiar de risco definido são adicionados a uma lista de "em risco" consumida
pela equipe de retenção/CRM. A ação específica de retenção (desconto, upgrade,
contato proativo) é escolhida manualmente pela equipe com base no segmento do
cliente — a seleção da ação está fora do escopo do modelo neste MVP.

---

## VALUE PROPOSITION

*(sem alteração — bloco já estava validado)*

---

## DATA COLLECTION

Dados vindos do dataset público IBM Telco Customer Churn, usado como proxy para
um extrato real de produção de billing + CRM. Snapshot estático e único neste
projeto acadêmico — sem atualização contínua implementada. Em uma implantação
real, um job de ETL periódico (ex.: diário/semanal) do banco de billing
atualizaria o dataset; documentado como trabalho futuro.

---

## DATA SOURCES

Telco Customer Churn (IBM), dataset tabular público. Fonte:
github.com/IBM/telco-customer-churn-on-icp4d, `data/Telco-Customer-Churn.csv`.
7.043 registros, 21 colunas (confirmado na EDA).

---

## FEATURES

Features utilizadas: `tenure`, tipo de contrato, forma de pagamento, serviços
assinados (internet, streaming, suporte técnico), cobranças mensais e totais,
além dos atributos demográficos do dataset. Representação: numéricas
padronizadas com `StandardScaler`; categóricas em one-hot com
`handle_unknown="ignore"`; binárias Sim/Não mapeadas para 0/1. Dataset já
agregado no nível de cliente — sem agregação entre entidades. `customerID` é
descartado por ser identificador sem poder preditivo.

---

## BUILDING MODELS

4 candidatos comparados, todos em Scikit-Learn — Regressão Logística
(baseline), Random Forest (ensemble), MLPClassifier (rede neural simples) e KNN
(candidato adicional). O de melhor ROC-AUC é escolhido como campeão e salvo com
`joblib`, exigindo margem mínima de 0.01 sobre o baseline para evitar troca por
ruído. Campeão: **Regressão Logística (ROC-AUC 0.8415)**. Rastreamento de
experimentos leve — tabela comparativa em CSV; MLflow não utilizado.
Retreinamento fora de escopo neste dataset estático; recomendado
mensal/trimestral em produção. Treinos rodam em sessão padrão de notebook/CI,
apenas CPU.

---

## MAKING PREDICTIONS

Endpoint de API síncrono em tempo real (FastAPI `/predict`), sob demanda a cada
interação com o cliente, com threshold de decisão em 0.5. Rollout em batch
(diário/semanal) permanece como alternativa viável para a lista de clientes em
risco. Orçamento de tempo: p95 < 300ms incluindo featurização. Computação: uma
instância pequena de CPU, sem necessidade de GPU. A API é containerizada
(Docker) para facilitar o deploy.

---

## IMPACT SIMULATION

Valores de custo/ganho (ilustrativos, a substituir por números reais de
negócio): oferta de retenção ≈ R$50/cliente; cliente perdido (LTV) ≈
R$1.200/cliente. Custo de negócio = (falsos negativos × custo de perder um
cliente) + (falsos positivos × custo de oferta desperdiçada), simulado no
conjunto de teste held-out. Critério de deploy: superar as baselines nessa
função de custo, não apenas em AUC-ROC.

Restrição de fairness: monitorar a diferença de recall e de taxa de falso
positivo (FPR) entre subgrupos de `gender`, `SeniorCitizen`, `Partner` e
`Dependents`; sinalizar disparidade > 10pp. **Auditoria executada**: `gender`
sem disparidade relevante; `SeniorCitizen`, `Partner` e `Dependents`
sinalizados, com gap de FPR como achado principal. Mitigação analisada e
adiada — ver Model Card, seções 6 e 6.1.

---

## MONITORING

SLOs: latência p95 < 300ms, `/health` retornando 200 continuamente, throughput
dimensionado para a base ativa de clientes. KPIs pós-deploy: AUC-ROC / PR-AUC /
F1 ao vivo conforme novos rótulos chegam, métrica de custo de negócio, data
drift em `tenure` e cobranças mensais, volume de previsões, e gap de FPR entre
subgrupos sensíveis. Revisão mensal, ou quando um novo lote rotulado chegar;
alertar se AUC-ROC cair > 0.03 ou se o gap de FPR exceder 10pp.
