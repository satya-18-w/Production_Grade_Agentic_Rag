#!/usr/bin/env bash
# ============================================================
# DEPLOY.SH — Azure VM Code + Env (Re)Deployment Script
# ============================================================

set -euo pipefail

RESOURCE_GROUP="rg-agentic-rag-prod"
VM_NAME="vm-agentic-rag"
VM_USER="azureuser"
PROJECT_DIR="~/Production_Grade_Agentic_Rag"
VM_PROJECT_DIR="/home/${VM_USER}/Production_Grade_Agentic_Rag"
ENV_FILE="${1:-.env}"
REBUILD="${2:---build}"
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "❌ Env file not found: ${ENV_FILE}"
  echo "   Pass a path: ./deploy.sh /path/to/prod.env [--build|--no-build]"
  exit 1
fi

if [[ "${REBUILD}" != "--build" && "${REBUILD}" != "--no-build" ]]; then
  echo "❌ Invalid rebuild flag: ${REBUILD}"
  echo "   Usage: ./deploy.sh [env-file] [--build|--no-build]"
  exit 1
fi

echo "🚀 Ensuring Azure VM is running..."
az vm start --resource-group "${RESOURCE_GROUP}" --name "${VM_NAME}" --no-wait
az vm wait --resource-group "${RESOURCE_GROUP}" --name "${VM_NAME}" --created

PUBLIC_IP=$(az vm list-ip-addresses \
  --resource-group "${RESOURCE_GROUP}" \
  --name "${VM_NAME}" \
  --query "[0].virtualMachine.network.publicIpAddresses[0].ipAddress" \
  -o tsv)

if [[ -z "${PUBLIC_IP}" ]]; then
  echo "❌ Could not resolve VM public IP."
  exit 1
fi

echo "✅ VM is running: ${PUBLIC_IP}"
echo "📦 Syncing project code to VM (excluding generated/secrets directories)..."
rsync -az \
  --delete \
  --exclude '.git' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude '.venv/' \
  --exclude 'venv/' \
  --exclude '__pycache__/' \
  --exclude 'processed_data/' \
  --exclude 'DATA/' \
  --exclude '.logfire/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "${LOCAL_PROJECT_DIR}/" "${VM_USER}@${PUBLIC_IP}:${VM_PROJECT_DIR}/"

echo "🔐 Uploading env file to VM (persistent in filesystem)..."
scp -A -o StrictHostKeyChecking=no "${ENV_FILE}" "${VM_USER}@${PUBLIC_IP}:${VM_PROJECT_DIR}/.env"

if [[ "${REBUILD}" == "--build" ]]; then
  REBUILD_CMD="docker compose down --remove-orphans && docker compose up -d --build"
else
  REBUILD_CMD="docker compose down --remove-orphans && docker compose up -d"
fi

echo "🐳 Re-deploying stack..."
ssh -A -o StrictHostKeyChecking=no "${VM_USER}@${PUBLIC_IP}" "cd ${PROJECT_DIR} && ${REBUILD_CMD}"

echo ""
echo "============================================================"
echo "🎉 Deployment complete."
echo "   Environment persisted at: ${VM_PROJECT_DIR}/.env"
echo "------------------------------------------------------------"
echo "🌐 Streamlit Web UI: http://${PUBLIC_IP}:8501"
echo "⚙️ FastAPI Backend:  http://${PUBLIC_IP}:8000"
echo "============================================================"
