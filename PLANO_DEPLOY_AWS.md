# 🚀 Plano de Deploy na AWS — API de Inferência (Container)
### Tech Challenge Fase 1 — FIAP | Estratégia: Imagem Docker + Container gerenciado

> **Decisão de arquitetura:** em vez de Lambda (limitações de cold start, tamanho de
> pacote e timeout para modelos ML) ou EC2 (gestão de servidor, patching, custo ocioso),
> vamos empacotar a API FastAPI em uma **imagem Docker** e rodar em **container serverless**.

---

## 🎯 Qual serviço de container escolher?

| Opção | Complexidade | Quando usar | Recomendação |
|---|---|---|---|
| **AWS App Runner** | ⭐ Baixa | HTTPS automático, auto-scaling, zero infra | ✅ **Comece por aqui** (deploy em ~10 min) |
| **ECS + Fargate** | ⭐⭐ Média | Controle fino, ALB, VPC, produção robusta | ✅ Evolua para cá quando precisar de ALB/rede |
| ECS + EC2 | ⭐⭐⭐ Alta | Precisa gerenciar instâncias | ❌ Fura o objetivo (é EC2) |
| Lambda (container) | ⭐⭐ Média | Cargas esporádicas e leves | ⚠️ Não ideal p/ sklearn + latência |

**Recomendação prática:** faça o **App Runner** primeiro (entrega e demo do vídeo STAR),
e deixe o **ECS Fargate** documentado como evolução de produção. Ambos usam a **mesma
imagem no ECR** — o esforço de construir a imagem é reaproveitado 100%.

```
        ┌──────────────┐   docker push   ┌──────────────┐   pull    ┌───────────────┐
        │  Sua máquina │ ──────────────▶ │  Amazon ECR  │ ────────▶ │  App Runner    │
        │  (Docker)    │                 │  (registry)  │           │  ou ECS Fargate│
        └──────────────┘                 └──────────────┘           └───────┬───────┘
                                                                            │ HTTPS
                                                                     https://.../predict
```

---

## ✅ Pré-requisitos (uma vez só)

1. **Conta AWS** com acesso programático (IAM user ou SSO).
2. **AWS CLI v2** instalado e configurado:
   ```bash
   aws --version            # deve ser 2.x
   aws configure            # Access Key, Secret, região (us-east-1), output json
   aws sts get-caller-identity   # valida se autenticou
   ```
3. **Docker** instalado e rodando (`docker info`).
4. Estrutura do projeto com `src/main.py` (objeto `app` do FastAPI) e `models/*.pkl`.

---

## 📋 Passo a passo prático

### Etapa 0 — Preparar o projeto para conteinerização
Copie para a raiz do repositório os arquivos deste kit:
- `Dockerfile`
- `.dockerignore`
- `requirements.txt` (fixe as versões que você validou)
- `docker-compose.yml`

> ⚠️ Ajuste no `Dockerfile` o `CMD` para o caminho real do seu app
> (`src.main:app`). Confirme que o `/health` e `/predict` existem (critério do desafio).

---

### Etapa 1 — Testar a imagem LOCALMENTE (antes de gastar nuvem)
```bash
docker compose up --build

# Em outro terminal:
curl http://localhost:8000/health
# Teste o predict com um payload de exemplo do seu dataset Telco:
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":1,"PhoneService":"No","MultipleLines":"No phone service","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":29.85,"TotalCharges":29.85}'
```
✔️ Só avance quando `/health` responder `200` e `/predict` retornar a propensão.

---

### Etapa 2 — Publicar a imagem no Amazon ECR
```bash
cd scripts
chmod +x *.sh
./01_build_and_push_ecr.sh
```
O script cria o repositório (com **scan de vulnerabilidades**), autentica o Docker,
faz o build em `linux/amd64` (evita erro em Mac Apple Silicon) e sobe a imagem
com **tag por timestamp** (rastreabilidade) + `latest`.

---

### Etapa 3A — Deploy no App Runner (recomendado / mais simples)
```bash
./02_deploy_apprunner.sh

# Acompanhe e pegue a URL pública:
aws apprunner list-services --region us-east-1
```
- HTTPS já vem pronto. Acesse `https://<ServiceUrl>/docs` (Swagger) e `/health`.
- `AutoDeploymentsEnabled: true` → todo push de `latest` no ECR **redeploya sozinho**.

### Etapa 3B — Deploy no ECS Fargate (produção / mais controle)
```bash
# Ajuste ecs/task-definition.json se quiser (CPU/memória)
./03_deploy_ecs_fargate.sh
```
Cria cluster Fargate, registra a task definition, sobe o service com IP público e
logs no CloudWatch (`/ecs/churn-api`). Em produção real, coloque um **ALB** na
frente (porta 443) em vez de expor a 8000 direto.

---

### Etapa 4 — Validar o deploy em nuvem
```bash
# App Runner
curl https://<ServiceUrl>/health

# Predição em produção
curl -X POST https://<ServiceUrl>/predict \
     -H "Content-Type: application/json" \
     -d '{"tenure": 2, "MonthlyCharges": 95.0, "Contract": "Month-to-month"}'
```
Abra o Swagger em `/docs` — ótimo para gravar o **vídeo STAR** mostrando a API no ar.

---

## 🔍 Observabilidade e operação
- **Logs:** CloudWatch Logs (App Runner e ECS geram automaticamente).
- **Métricas/scaling:** App Runner escala por concorrência; ECS via *Service Auto Scaling*.
- **Rollback:** publique tags versionadas; para voltar, aponte o serviço à tag anterior.
- **Custo:** App Runner cobra por vCPU/memória ativos; pausar o serviço zera o custo de compute.

## 🔐 Boas práticas aplicadas neste kit
- Imagem **multi-stage** (menor e sem `build-essential` no runtime).
- Container roda como **usuário não-root**.
- `.dockerignore` exclui `notebooks/`, `data/`, `.env` (nada de segredo/dado bruto na imagem).
- **Healthcheck** nativo alinhado ao endpoint `/health` do desafio.
- **Scan de vulnerabilidades** no ECR ligado (`scanOnPush=true`).
- Versões **fixadas** no `requirements.txt` (reprodutibilidade — critério avaliado).

## 🧹 Limpeza (evitar cobrança após a entrega)
```bash
# App Runner
aws apprunner delete-service --service-arn <ARN> --region us-east-1
# ECS
aws ecs update-service --cluster churn-cluster --service churn-api-svc --desired-count 0
aws ecs delete-service --cluster churn-cluster --service churn-api-svc --force
aws ecs delete-cluster --cluster churn-cluster
# ECR
aws ecr delete-repository --repository-name churn-api --force
```

---

## 📁 Conteúdo do kit
```
deploy-aws/
├── Dockerfile                       # imagem multi-stage da API
├── .dockerignore                    # exclusões de build
├── requirements.txt                 # dependências fixadas
├── docker-compose.yml               # teste local
├── PLANO_DEPLOY_AWS.md              # este documento
├── scripts/
│   ├── 01_build_and_push_ecr.sh     # build + push ECR
│   ├── 02_deploy_apprunner.sh       # deploy App Runner (simples)
│   └── 03_deploy_ecs_fargate.sh     # deploy ECS Fargate (produção)
└── ecs/
    └── task-definition.json         # definição da task Fargate
```
