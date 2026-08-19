#!/bin/bash
# ============================================================
# OFF.SH — Azure VM Status Check & Deallocation Script
# ============================================================

RESOURCE_GROUP="rg-agentic-rag-prod"
VM_NAME="vm-agentic-rag"
VM_USER="azureuser"

echo "🔍 Checking Azure VM status..."
VM_STATUS=$(az vm get-instance-view --resource-group "${RESOURCE_GROUP}" --name "${VM_NAME}" --query "instanceView.statuses[1].displayStatus" -o tsv 2>/dev/null)

if [ -z "${VM_STATUS}" ]; then
  VM_STATUS="Unknown"
fi

echo "📊 Current VM Status: ${VM_STATUS}"

if [[ "${VM_STATUS}" == "VM running" ]]; then
  PUBLIC_IP=$(az vm list-ip-addresses --resource-group "${RESOURCE_GROUP}" --name "${VM_NAME}" --query "[0].virtualMachine.network.publicIpAddresses[0].ipAddress" -o tsv 2>/dev/null)
  echo "🌐 Public IP: ${PUBLIC_IP}"
  
  echo "🐳 Checking running Docker containers..."
  ssh -A -o ConnectTimeout=5 -o StrictHostKeyChecking=no "${VM_USER}@${PUBLIC_IP}" "docker compose -f ~/Production_Grade_Agentic_Rag/docker-compose.yml ps" 2>/dev/null || echo "Unable to query SSH container status."
  
  echo ""
  echo "🛑 Stopping Docker containers cleanly on VM..."
  ssh -A -o ConnectTimeout=5 -o StrictHostKeyChecking=no "${VM_USER}@${PUBLIC_IP}" "cd ~/Production_Grade_Agentic_Rag && docker compose stop" 2>/dev/null
  
  echo "⚡ Deallocating Azure VM to stop all compute charges..."
  az vm deallocate --resource-group "${RESOURCE_GROUP}" --name "${VM_NAME}"
  
  echo ""
  echo "============================================================"
  echo "💤 Enterprise Agentic RAG is OFF & DEALLOCATED."
  echo "   (Compute billing has stopped completely)"
  echo "============================================================"
else
  echo "ℹ️ VM is already stopped/deallocated (Status: ${VM_STATUS}). No compute charges active."
fi
