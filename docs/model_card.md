# Model Card — Churn Prediction

> ⚠️ **Template.** Este documento tem a estrutura pronta, mas nenhum campo foi
> preenchido ainda — o grupo ainda não treinou nenhum modelo. Preencher de
> verdade só faz sentido após a Etapa 2 (comparação de modelos) e a escolha do
> modelo campeão.

## Visão geral do modelo
- **Tipo de tarefa**: classificação binária (previsão de churn de clientes).
- **Modelo campeão**: _a definir na Etapa 2, após comparar Regressão
  Logística (baseline), Random Forest/ensemble e MLPClassifier._
- **Dataset de treino**: Telco Customer Churn (IBM), ~7.043 registros, split
  80/20 estratificado, seed fixa (`random_state=42`).

## Performance
_A preencher após o treino e a comparação de modelos (Etapas 1 e 2)._

| Métrica | Valor |
|---|---|
| ROC-AUC | — |
| PR-AUC | — |
| F1-score | — |
| Recall (classe Churn) | — |

## Limitações conhecidas
- Dataset é uma foto estática — não captura sazonalidade real de churn ao
  longo do tempo.
- Modelo treinado em dados de uma única empresa fictícia (IBM sample) — pode
  não generalizar para outras operadoras sem retreino.
- _Demais limitações a documentar conforme o comportamento observado durante
  o treino real (Etapa 1/2)._

## Vieses potenciais
- _A avaliar depois do treino_: diferença de recall/taxa de falso positivo
  entre grupos (ex.: `gender`, `SeniorCitizen`), conforme constraint de
  fairness definida no ML Canvas (bloco Impact Simulation).

## Cenários de falha
- Clientes com combinações de features nunca vistas em treino (ex.: um novo
  `PaymentMethod`) — a API trata isso com `handle_unknown="ignore"` no
  encoder, mas a predição pode ficar menos confiável nesses casos.
- _Demais cenários a identificar durante a avaliação do modelo treinado._

## Como o modelo é usado
- Ver bloco **Decisions** e **Making Predictions** do ML Canvas
  (`docs/ml_canvas.md` / `docs/ml_canvas_en.md`).
