# ⚙️ CI/CD no GitHub Actions — Churn API (AWS)

Esteira completa: **CI** (lint + testes + gate de qualidade do modelo) → **CD** (build → push ECR → deploy App Runner → smoke test `/health`).
Autenticação **OIDC** (sem chaves estáticas na nuvem).

```
 git push main
      │
      ▼
┌─────────────────┐        ┌──────────────────────────────┐
│  CI: build-test │        │ CD: deploy (só na main)       │
│  • ruff (lint)  │  ✅──▶ │ • OIDC: assume IAM Role        │
│  • pytest API   │        │ • docker build --platform amd64│
│  • gate ΔF1≤0.02│        │ • push ECR (tag SHA + latest)  │
│  • docker build │        │ • App Runner start-deployment  │
└─────────────────┘        │ • curl /health até 200         │
                           └──────────────────────────────┘
```

---

## 🔑 Como funciona o OIDC (por que não usamos Access Key)
Em vez de guardar `AWS_ACCESS_KEY_ID`/`SECRET` como secret (risco de vazamento), o GitHub
emite um **token OIDC de curta duração** a cada execução. A AWS confia nesse token e entrega
credenciais temporárias — apenas para **o seu repositório**, **no branch main**. Mais seguro
e alinhado à sua preferência de least privilege / separação de escopo.

---

## 📋 Setup passo a passo (uma vez só)

### 1) Criar a role OIDC na AWS
```bash
cd scripts
# Edite GITHUB_ORG e GITHUB_REPO no topo do script antes de rodar
./04_setup_github_oidc.sh
```
O script cria o **OIDC Provider**, a **IAM Role** `GitHubActionsDeployRole` (com trust
restrito ao seu repo) e anexa permissões mínimas de **ECR + App Runner**.

### 2) Cadastrar o secret no GitHub
No repositório: **Settings → Secrets and variables → Actions → New repository secret**
| Nome | Valor |
|---|---|
| `AWS_ACCOUNT_ID` | (o número da conta que o script imprime no final) |

> É o **único** secret necessário. Nada de chave de acesso. 🎉

### 3) Garantir que o serviço App Runner existe (primeiro deploy)
O workflow **atualiza** um serviço existente. Na primeiríssima vez, crie-o manualmente:
```bash
./01_build_and_push_ecr.sh      # sobe a primeira imagem
./02_deploy_apprunner.sh        # cria o service 'churn-api'
```
Depois disso, todo `git push` na `main` faz o deploy automático.

### 4) Colocar os arquivos no repositório
```
seu-repo/
├── .github/workflows/ci-cd.yml
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
├── src/main.py            # sua API (app FastAPI)
├── models/model.pkl       # modelo campeão treinado
├── models/baseline_metrics.json   # {"f1": 0.78}  <- da Etapa 2
├── data/test.parquet      # hold-out p/ o gate de qualidade
└── tests/
    ├── test_api.py
    └── test_model_contract.py
```

### 5) Disparar
```bash
git add .
git commit -m "feat: esteira CI/CD para deploy da Churn API na AWS"
git push origin main
```
Acompanhe em **Actions** no GitHub. Ao final, o job imprime a URL: `https://<ServiceUrl>/docs`.

---

## 🧪 O que cada etapa valida
| Etapa | O que faz | Bloqueia deploy se... |
|---|---|---|
| `ruff` | Lint do código | (não bloqueante) |
| `pytest tests/` | `/health` e `/predict` respondem | teste falha |
| **gate ΔF1** | modelo campeão vs. baseline | `F1 < 0.75` ou regressão `> 0.02` |
| `docker build` (CI) | imagem compila | build quebra |
| OIDC | assume role na AWS | trust/secret errado |
| smoke `/health` | API respondeu 200 pós-deploy | não subiu a tempo |

---

## 🛡️ Boas práticas já embutidas
- **`environment: production`** → você pode exigir *aprovação manual* antes do deploy
  (Settings → Environments → production → Required reviewers).
- **PR roda só CI** (não faz deploy) — valida antes do merge.
- **`concurrency`** cancela runs antigos do mesmo branch (economiza minutos).
- **Tag por SHA** (`$GITHUB_SHA`) → rastreabilidade e rollback fácil (aponte para a tag anterior).
- **Permissões mínimas** na role (só ECR + App Runner).

## 🔁 Rollback rápido
```bash
# Liste as tags no ECR e reaponte o App Runner para a imagem anterior:
aws ecr describe-images --repository-name churn-api --region us-east-1 \
  --query 'sort_by(imageDetails,&imagePushedAt)[*].imageTags' --output table
# Depois, retag a versão boa como 'latest' e faça start-deployment.
```

## 🧯 Troubleshooting comum
- **`Not authorized to perform sts:AssumeRoleWithWebIdentity`** → confira `GITHUB_ORG/REPO`
  na trust policy (rode o script 04 de novo) e o secret `AWS_ACCOUNT_ID`.
- **`Service não encontrado`** no deploy → faça o 1º deploy manual (script 02).
- **Erro de arquitetura no container** → o build usa `--platform linux/amd64` de propósito
  (evita imagem ARM em Mac Apple Silicon).
