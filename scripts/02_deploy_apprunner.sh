#!/usr/bin/env bash
# =========================================================================
# 02 - Deploy via AWS App Runner (CAMINHO MAIS SIMPLES / RECOMENDADO)
# App Runner: container serverless, HTTPS automatico, auto-scaling.
# Sem gerenciar cluster, VPC, load balancer nem EC2.
# =========================================================================
set -euo pipefail

export AWS_REGION="us-east-1"
export ECR_REPO="churn-api"
export SERVICE_NAME="churn-api"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"

# 1) Role que permite ao App Runner puxar a imagem do ECR (idempotente)
ROLE_NAME="AppRunnerECRAccessRole"
if ! aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  aws iam create-role --role-name "${ROLE_NAME}" \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "build.apprunner.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }'
  aws iam attach-role-policy --role-name "${ROLE_NAME}" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
  echo ">> Aguardando propagacao da role IAM..."
  sleep 15
fi
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/${ROLE_NAME}"

# 2) Cria o serviço App Runner apontando para a imagem no ECR
aws apprunner create-service \
  --service-name "${SERVICE_NAME}" \
  --region "${AWS_REGION}" \
  --source-configuration '{
    "AuthenticationConfiguration": {"AccessRoleArn": "'"${ROLE_ARN}"'"},
    "AutoDeploymentsEnabled": true,
    "ImageRepository": {
      "ImageIdentifier": "'"${ECR_URI}"'",
      "ImageRepositoryType": "ECR",
      "ImageConfiguration": {
        "Port": "8000",
        "RuntimeEnvironmentVariables": {"ENV": "production"}
      }
    }
  }' \
  --instance-configuration '{"Cpu": "1024", "Memory": "2048"}' \
  --health-check-configuration '{
    "Protocol": "HTTP",
    "Path": "/health",
    "Interval": 10,
    "Timeout": 5,
    "HealthyThreshold": 1,
    "UnhealthyThreshold": 5
  }'

echo ""
echo ">> Servico criado. Acompanhe o status e pegue a URL publica com:"
echo "   aws apprunner list-services --region ${AWS_REGION}"
echo "   (a URL fica no campo 'ServiceUrl' — acesse https://<ServiceUrl>/docs)"
