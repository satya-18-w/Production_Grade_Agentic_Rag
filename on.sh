#!/bin/bash
# ============================================================
# ON.SH — Azure VM & App Stack Startup Script
# ============================================================

RESOURCE_GROUP="rg-agentic-rag-prod"
VM_NAME="vm-agentic-rag"
VM_USER="azureuser"

echo "🚀 Starting Azure Virtual Machine: ${VM_NAME}..."
az vm start --resource-group "${RESOURCE_GROUP}" --name "${VM_NAME}" --no-wait

echo "⏳ Waiting for VM power state to become VM running..."
az vm wait --resource-group "${RESOURCE_GROUP}" --name "${VM_NAME}" --created

PUBLIC_IP=$(az vm list-ip-addresses --resource-group "${RESOURCE_GROUP}" --name "${VM_NAME}" --query "[0].virtualMachine.network.publicIpAddresses[0].ipAddress" -o tsv)

if [ -z "${PUBLIC_IP}" ]; then
  echo "❌ Could not resolve public IP for VM ${VM_NAME}."
  exit 1
fi

echo "✅ VM is running at Public IP: ${PUBLIC_IP}"

echo "🐳 Verifying Docker compose stack status on VM..."
ssh -A -o StrictHostKeyChecking=no "${VM_USER}@${PUBLIC_IP}" "cd ~/Production_Grade_Agentic_Rag && docker compose up -d"

echo ""
echo "============================================================"
echo "🎉 Enterprise Agentic RAG is ONLINE!"
echo "------------------------------------------------------------"
echo "🌐 Streamlit Web UI: http://${PUBLIC_IP}:8501"
echo "⚙️ FastAPI Backend:  http://${PUBLIC_IP}:8000"
echo "============================================================"
