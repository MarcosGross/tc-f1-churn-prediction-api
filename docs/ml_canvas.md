# ML Canvas — Previsão de Churn

> Preenchido em grupo. Este é um RASCUNHO inicial — discutam e ajustem cada bloco.
> Objetivo: alinhar TODO o time sobre o "porquê" antes de escrever qualquer linha de código de modelo.

## 1. Problema de Negócio (Value Proposition)
Uma operadora de telecomunicações está perdendo clientes (churn) em ritmo acelerado.
A diretoria precisa identificar, ANTES do cancelamento, quais clientes têm alto risco
de sair, para que a equipe de retenção possa agir (desconto, upgrade, contato proativo).

## 2. Stakeholders
- **Diretoria comercial**: quer reduzir a taxa de churn mensal.
- **Time de retenção/CRM**: vai consumir a lista de clientes em risco.
- **Engenharia de dados**: mantém o pipeline de dados dos clientes.
- **Grupo (vocês)**: constrói, valida e disponibiliza o modelo via API.

## 3. Dataset
- **Fonte**: Telco Customer Churn (IBM) — dataset público, tabular.
- **Volume**: ~7.043 registros, ~21 colunas (ajustar depois da EDA real).
- **Variável alvo**: `Churn` (Yes/No) → binária.
- **Features candidatas**: tempo de contrato (tenure), tipo de contrato, forma de
  pagamento, serviços contratados (internet, streaming, suporte técnico), cobrança
  mensal e total.

## 4. Métricas Técnicas (Model Performance)
- **Métrica principal**: AUC-ROC (robusta a desbalanceamento).
- **Métricas secundárias**: PR-AUC (importante pois churn costuma ser minoritário),
  F1-score, Recall da classe "Churn" (não queremos deixar de detectar quem vai sair).
- **Baseline mínimo**: Regressão Logística é o próprio baseline; Random Forest e
  MLPClassifier (Scikit-Learn) precisam superá-la para virar modelo campeão.

## 5. Métrica de Negócio
- **Custo de churn evitado**: se reter um cliente custa R$X (ex: desconto) e perder
  um cliente custa R$Y (LTV perdido), o modelo deve minimizar custo total =
  (falsos negativos × custo de perda) + (falsos positivos × custo de retenção
  desperdiçado).
- Ajustar threshold de classificação em função desse trade-off, não só usar 0.5.

## 6. SLOs (Service Level Objectives) da API
- Latência de inferência: p95 < 300ms por requisição.
- Disponibilidade: /health respondendo 200 continuamente.
- Throughput mínimo esperado: definir com base no volume de clientes ativos.

## 7. Riscos e Vieses Potenciais
- Dataset pode não refletir sazonalidade real (é uma foto estática).
- Risco de viés por variáveis sensíveis (ex: gênero, dependentes) — documentar no
  Model Card e avaliar se devem ser removidas ou tratadas com cautela.
- Risco de "data leakage" se alguma feature só existir depois do churn acontecer.

## 8. Como o modelo será usado (Decisions)
- **Batch** (ex: diário/semanal) alimentando uma lista de "clientes em risco" para
  o time de CRM, OU
- **Real-time** via API síncrona para decisão no momento de interação com o cliente.
- *(Decidir e justificar formalmente na Etapa 4 — arquitetura de deploy)*

---
**Próximos passos do grupo:**
1. Cada membro lê este canvas e sugere ajustes via Pull Request ou comentário.
2. Validar hipóteses de negócio (item 5) — mesmo que sejam suposições, documentem
   as premissas usadas.
3. Depois de aprovado, seguir para a EDA (notebook 01_eda.ipynb).
