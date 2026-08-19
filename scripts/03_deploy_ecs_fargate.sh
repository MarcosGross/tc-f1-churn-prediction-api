#!/usr/bin/env bash
# =========================================================================
# 03 - Deploy via Amazon ECS + Fargate (CAMINHO PRODUCAO / MAIS CONTROLE)
# Use quando precisar de ALB, VPC dedicada, auto-scaling fino, etc.
# Requer: task-definition.json (na pasta ecs/) com <ACCOUNT_ID> ajustado.
# =========================================================================
set -euo pipefail

export AWS_REGION="us-east-1"
export CLUSTER_NAME="churn-cluster"
export SERVICE_NAME="churn-api-svc"

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

# 0) Substitui o placeholder <ACCOUNT_ID> na task definition
sed "s/<ACCOUNT_ID>/${ACCOUNT_ID}/g" ../ecs/task-definition.json > /tmp/task-def.json

# 1) Garante a role de execucao do ECS (idempotente)
if ! aws iam get-role --role-name ecsTaskExecutionRole >/dev/null 2>&1; then
  aws iam create-role --role-name ecsTaskExecutionRole \
    --assume-role-policy-document '{
      "Version":"2012-10-17",
      "Statement":[{"Effect":"Allow","Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]
    }'
  aws iam attach-role-policy --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
  sleep 10
fi

# 2) Cria o cluster Fargate (idempotente)
aws ecs describe-clusters --clusters "${CLUSTER_NAME}" --region "${AWS_REGION}" \
  --query 'clusters[0].status' --output text 2>/dev/null | grep -q ACTIVE \
  || aws ecs create-cluster --cluster-name "${CLUSTER_NAME}" --region "${AWS_REGION}"

# 3) Registra a task definition
TASK_DEF_ARN="$(aws ecs register-task-definition \
  --cli-input-json file:///tmp/task-def.json \
  --region "${AWS_REGION}" \
  --query 'taskDefinition.taskDefinitionArn' --output text)"
echo ">> Task definition registrada: ${TASK_DEF_ARN}"

# 4) Descobre rede default (subnets publicas + security group)
VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text --region ${AWS_REGION})"
SUBNETS="$(aws ec2 describe-subnets --filters Name=vpc-id,Values=${VPC_ID} \
  --query 'Subnets[].SubnetId' --output text --region ${AWS_REGION} | tr '\t' ',')"
SG_ID="$(aws ec2 describe-security-groups --filters Name=vpc-id,Values=${VPC_ID} Name=group-name,Values=default \
  --query 'SecurityGroups[0].GroupId' --output text --region ${AWS_REGION})"

# Libera a porta 8000 no security group (para teste; em prod use um ALB na 443)
aws ec2 authorize-security-group-ingress --group-id "${SG_ID}" \
  --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region "${AWS_REGION}" 2>/dev/null || true

# 5) Cria o service (mantem 1 tarefa rodando, IP publico habilitado)
aws ecs create-service \
  --cluster "${CLUSTER_NAME}" \
  --service-name "${SERVICE_NAME}" \
  --task-definition "${TASK_DEF_ARN}" \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[${SUBNETS}],securityGroups=[${SG_ID}],assignPublicIp=ENABLED}" \
  --region "${AWS_REGION}"

echo ""
echo ">> Service criado no cluster ${CLUSTER_NAME}."
echo ">> Pegue o IP publico da tarefa (apos ficar RUNNING):"
echo "   aws ecs list-tasks --cluster ${CLUSTER_NAME} --region ${AWS_REGION}"
echo "   depois: aws ecs describe-tasks ... e busque a ENI -> IP publico"
echo ">> Teste: http://<IP_PUBLICO>:8000/health"
