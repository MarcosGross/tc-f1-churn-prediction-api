#!/usr/bin/env bash
# =========================================================================
# 04 - Configurar OIDC entre GitHub Actions e AWS (SEM chave estática)
# Cria: (1) OIDC Provider do GitHub, (2) IAM Role que o workflow vai assumir.
# Rode UMA vez. Requer AWS CLI com permissão de IAM.
# =========================================================================
set -euo pipefail

# ---- AJUSTE estas 2 variáveis ----
export GITHUB_ORG="SEU_USUARIO_OU_ORG"     # ex.: marcosmoraes
export GITHUB_REPO="SEU_REPOSITORIO"       # ex.: tc-f1-churn-prediction-api

export AWS_REGION="us-east-1"
ROLE_NAME="GitHubActionsDeployRole"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
GITHUB_THUMBPRINT="6938fd4d98bab03faadb97b34396831e3780aea1"  # thumbprint padrão do GitHub OIDC

echo ">> Conta: ${ACCOUNT_ID} | Repo: ${GITHUB_ORG}/${GITHUB_REPO}"

# 1) Cria o OIDC Provider do GitHub (idempotente)
PROVIDER_ARN="arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"
if ! aws iam get-open-id-connect-provider --open-id-connect-provider-arn "${PROVIDER_ARN}" >/dev/null 2>&1; then
  aws iam create-open-id-connect-provider \
    --url "https://token.actions.githubusercontent.com" \
    --client-id-list "sts.amazonaws.com" \
    --thumbprint-list "${GITHUB_THUMBPRINT}"
  echo ">> OIDC Provider criado."
else
  echo ">> OIDC Provider já existe."
fi

# 2) Trust policy: só o SEU repo, no branch main, pode assumir a role
cat > /tmp/trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "${PROVIDER_ARN}"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {"token.actions.githubusercontent.com:aud": "sts.amazonaws.com"},
      "StringLike": {"token.actions.githubusercontent.com:sub": "repo:${GITHUB_ORG}/${GITHUB_REPO}:*"}
    }
  }]
}
EOF

# 3) Cria/atualiza a role
if aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  aws iam update-assume-role-policy --role-name "${ROLE_NAME}" --policy-document file:///tmp/trust-policy.json
  echo ">> Role atualizada."
else
  aws iam create-role --role-name "${ROLE_NAME}" \
    --assume-role-policy-document file:///tmp/trust-policy.json \
    --description "Role assumida pelo GitHub Actions via OIDC para deploy da Churn API"
  echo ">> Role criada."
fi

# 4) Permissões mínimas: push no ECR + gerenciar App Runner
cat > /tmp/deploy-permissions.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECRPushPull",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer"
      ],
      "Resource": "*"
    },
    {
      "Sid": "AppRunnerDeploy",
      "Effect": "Allow",
      "Action": [
        "apprunner:ListServices",
        "apprunner:DescribeService",
        "apprunner:StartDeployment",
        "apprunner:UpdateService"
      ],
      "Resource": "*"
    }
  ]
}
EOF

aws iam put-role-policy --role-name "${ROLE_NAME}" \
  --policy-name "ChurnDeployPolicy" \
  --policy-document file:///tmp/deploy-permissions.json

echo ""
echo "=================================================================="
echo ">> PRONTO! Configure no seu repositório GitHub:"
echo "   Settings > Secrets and variables > Actions > New repository secret"
echo "   Nome:  AWS_ACCOUNT_ID"
echo "   Valor: ${ACCOUNT_ID}"
echo ""
echo ">> A role que o workflow assume é:"
echo "   arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"
echo "=================================================================="
