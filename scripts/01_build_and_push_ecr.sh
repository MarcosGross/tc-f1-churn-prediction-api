#!/usr/bin/env bash
# =========================================================================
# 01 - Build da imagem e push para o Amazon ECR
# Pré-requisitos: AWS CLI v2 configurado (aws configure) + Docker rodando
# =========================================================================
set -euo pipefail

# ---- Ajuste estas variáveis ----
export AWS_REGION="us-east-1"
export ECR_REPO="churn-api"
export IMAGE_TAG="$(date +%Y%m%d-%H%M)"   # tag com timestamp (rastreabilidade)

# Descobre o Account ID automaticamente
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"

echo ">> Conta AWS: ${ACCOUNT_ID} | Região: ${AWS_REGION}"
echo ">> Imagem alvo: ${ECR_URI}:${IMAGE_TAG}"

# 1) Cria o repositório ECR (idempotente) com scan de vulnerabilidades ligado
aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository \
        --repository-name "${ECR_REPO}" \
        --image-scanning-configuration scanOnPush=true \
        --region "${AWS_REGION}"

# 2) Autentica o Docker no ECR
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# 3) Build (força linux/amd64 — importante se você usa Mac M1/M2/M3)
docker build --platform linux/amd64 -t "${ECR_REPO}:${IMAGE_TAG}" .

# 4) Tag + Push (tag versionada e 'latest')
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}:${IMAGE_TAG}"
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_URI}:latest"
docker push "${ECR_URI}:${IMAGE_TAG}"
docker push "${ECR_URI}:latest"

echo ""
echo ">> PUSH concluido. Guarde esta URI para o deploy:"
echo "   ${ECR_URI}:${IMAGE_TAG}"
